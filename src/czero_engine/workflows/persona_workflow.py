"""Persona workflow for CZero Engine - Interact with specialized AI personas."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from ..client import CZeroEngineClient
from ..models import PersonaListResponse, PersonaChatResponse, ConversationMessage

console = Console()


@dataclass
class ConversationContext:
    """Context for maintaining conversation with a persona."""
    persona_id: str
    persona_name: str
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    turn_count: int = 0
    max_history: int = 10  # Keep last N messages for context


class PersonaWorkflow:
    """
    Workflow for interacting with AI personas in CZero Engine.
    
    Personas provide specialized interaction styles and expertise:
    - Gestalt: Adaptive general assistant
    - Sage: Research and analysis expert
    - Pioneer: Innovation and creative solutions
    
    Each persona maintains conversation context for coherent dialogue.
    """
    
    def __init__(self, client: Optional[CZeroEngineClient] = None, verbose: bool = True):
        """
        Initialize Persona workflow.
        
        Args:
            client: Optional CZeroEngineClient instance
            verbose: Enable verbose output
        """
        self.client = client
        self._owns_client = client is None
        self.verbose = verbose
        self.available_personas: Optional[PersonaListResponse] = None
        self.active_persona: Optional[ConversationContext] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        if self._owns_client:
            self.client = CZeroEngineClient(verbose=self.verbose)
            await self.client.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._owns_client and self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
            
    async def list_personas(self, refresh: bool = False) -> PersonaListResponse:
        """
        List available AI personas.
        
        Args:
            refresh: Force refresh of persona list
            
        Returns:
            List of available personas
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        if refresh or not self.available_personas:
            self.available_personas = await self.client.list_personas()
            
        if self.verbose:
            table = Table(title="Available Personas")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Specialty", style="yellow")
            table.add_column("Tagline", style="dim")
            
            for persona in self.available_personas.personas:
                table.add_row(
                    persona.id,
                    persona.name,
                    persona.specialty,
                    persona.tagline or ""
                )
                
            console.print(table)
            
        return self.available_personas
        
    async def select_persona(self, persona_id: str) -> ConversationContext:
        """
        Select a persona for conversation.
        
        Args:
            persona_id: ID of the persona to select
            
        Returns:
            Conversation context for the selected persona
        """
        if not self.available_personas:
            await self.list_personas()
            
        # Find persona info
        persona_info = None
        for persona in self.available_personas.personas:
            if persona.id == persona_id:
                persona_info = persona
                break
                
        if not persona_info:
            raise ValueError(f"Persona not found: {persona_id}")
            
        # Create conversation context
        self.active_persona = ConversationContext(
            persona_id=persona_id,
            persona_name=persona_info.name
        )
        
        if self.verbose:
            print(f"\n=== Active Persona ===")
            print(f"Name: {persona_info.name}")
            print(f"Specialty: {persona_info.specialty}")
            print(f"Tagline: {persona_info.tagline}")
            print("=" * 20)
            
        return self.active_persona
        
    async def chat(
        self,
        message: str,
        persona_id: Optional[str] = None,
        model_id: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        maintain_history: bool = True
    ) -> PersonaChatResponse:
        """
        Chat with a persona.
        
        Args:
            message: Message to send
            persona_id: Optional persona ID (uses active if not specified)
            model_id: Optional specific model to use
            system_prompt_template: Optional custom system prompt
            max_tokens: Maximum tokens in response
            temperature: Generation temperature
            maintain_history: Whether to maintain conversation history
            
        Returns:
            Persona's response
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        # Determine which persona to use
        if persona_id:
            if not self.active_persona or self.active_persona.persona_id != persona_id:
                await self.select_persona(persona_id)
        elif not self.active_persona:
            # Default to Gestalt
            await self.select_persona("gestalt-default")
            
        if not self.active_persona:
            raise ValueError("No persona selected")
            
        # Prepare conversation history
        history = None
        if maintain_history and self.active_persona.conversation_history:
            # Keep only recent history
            recent_history = self.active_persona.conversation_history[-self.active_persona.max_history:]
            history = [msg.model_dump() for msg in recent_history]
            
        # Send message
        if self.verbose:
            console.print(f"\n[cyan]You:[/cyan] {message}")
            
        response = await self.client.persona_chat(
            persona_id=self.active_persona.persona_id,
            message=message,
            model_id=model_id,
            system_prompt_template=system_prompt_template,
            conversation_history=history,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Update conversation history
        if maintain_history:
            self.active_persona.conversation_history.append(
                ConversationMessage(role="user", content=message)
            )
            self.active_persona.conversation_history.append(
                ConversationMessage(role="assistant", content=response.response)
            )
            self.active_persona.turn_count += 1
            
        # Display response
        if self.verbose:
            print(f"\n{self.active_persona.persona_name}:")
            print(response.response)
            print("-" * 40)
            
        return response
        
    async def multi_persona_discussion(
        self,
        topic: str,
        persona_ids: List[str],
        rounds: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Have multiple personas discuss a topic.
        
        Each persona provides their perspective based on their specialty.
        
        Args:
            topic: Topic to discuss
            persona_ids: List of persona IDs to participate
            rounds: Number of discussion rounds
            
        Returns:
            List of responses from each persona
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        if self.verbose:
            console.print(Panel(
                f"[bold cyan]Multi-Persona Discussion[/bold cyan]\n"
                f"Topic: {topic}\n"
                f"Participants: {', '.join(persona_ids)}\n"
                f"Rounds: {rounds}",
                expand=False
            ))
            
        discussion_log = []
        current_topic = topic
        
        for round_num in range(rounds):
            if self.verbose:
                console.print(f"\n[bold]Round {round_num + 1}[/bold]")
                
            round_responses = []
            
            for persona_id in persona_ids:
                # Build prompt with previous responses
                if round_responses:
                    previous = "\n".join([
                        f"{r['persona']}: {r['response'][:200]}..."
                        for r in round_responses
                    ])
                    prompt = f"""Topic: {current_topic}

Previous responses in this round:
{previous}

Please provide your perspective on this topic."""
                else:
                    prompt = f"Please provide your perspective on: {current_topic}"
                    
                # Get persona response
                response = await self.chat(
                    message=prompt,
                    persona_id=persona_id,
                    maintain_history=False  # Don't maintain history for discussions
                )
                
                round_responses.append({
                    "persona": persona_id,
                    "response": response.response,
                    "round": round_num + 1
                })
                
            discussion_log.extend(round_responses)
            
            # Generate next round's topic based on responses
            if round_num < rounds - 1:
                synthesis = " ".join([r["response"][:100] for r in round_responses])
                next_topic_prompt = f"""Based on these perspectives on "{current_topic}":
{synthesis}

What follow-up question or aspect should be explored next? Provide only the question."""
                
                next_response = await self.client.chat(
                    message=next_topic_prompt,
                    use_rag=False,
                    max_tokens=100
                )
                current_topic = next_response.response
                
                if self.verbose:
                    console.print(f"\n[dim]Next topic: {current_topic}[/dim]")
                    
        return discussion_log
        
    async def persona_comparison(
        self,
        question: str,
        persona_ids: Optional[List[str]] = None
    ) -> Dict[str, PersonaChatResponse]:
        """
        Compare how different personas respond to the same question.
        
        Args:
            question: Question to ask all personas
            persona_ids: List of personas to compare (uses all if not specified)
            
        Returns:
            Dictionary of persona responses
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        # Get all personas if not specified
        if not persona_ids:
            personas = await self.list_personas()
            persona_ids = [p.id for p in personas.personas]
            
        if self.verbose:
            console.print(Panel(
                f"[bold cyan]Persona Comparison[/bold cyan]\n"
                f"Question: {question}",
                expand=False
            ))
            
        responses = {}
        
        for persona_id in persona_ids:
            if self.verbose:
                console.print(f"\n[cyan]Asking {persona_id}...[/cyan]")
                
            response = await self.chat(
                message=question,
                persona_id=persona_id,
                maintain_history=False
            )
            
            responses[persona_id] = response
            
        return responses
        
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current conversation.
        
        Returns:
            Conversation statistics and recent history
        """
        if not self.active_persona:
            return {"error": "No active conversation"}
            
        summary = {
            "persona": self.active_persona.persona_name,
            "persona_id": self.active_persona.persona_id,
            "turn_count": self.active_persona.turn_count,
            "message_count": len(self.active_persona.conversation_history),
            "recent_messages": [
                msg.model_dump() 
                for msg in self.active_persona.conversation_history[-6:]
            ]
        }
        
        if self.verbose:
            console.print(Panel(
                f"[bold]Conversation Summary[/bold]\n"
                f"Persona: {summary['persona']}\n"
                f"Turns: {summary['turn_count']}\n"
                f"Messages: {summary['message_count']}",
                expand=False
            ))
            
        return summary
        
    def reset_conversation(self):
        """Reset the current conversation context."""
        if self.active_persona:
            self.active_persona.conversation_history = []
            self.active_persona.turn_count = 0
            
            if self.verbose:
                console.print("[yellow]Conversation reset[/yellow]")


# Example usage
async def example_personas():
    """Example of using personas workflow."""
    
    async with PersonaWorkflow() as workflow:
        # List available personas
        await workflow.list_personas()
        
        # Chat with Gestalt (default persona)
        response = await workflow.chat(
            "Hello! What makes you unique as an AI assistant?"
        )
        
        # Continue conversation
        response = await workflow.chat(
            "Can you help me understand semantic search?"
        )
        
        # Switch to Sage persona
        await workflow.select_persona("sage")
        response = await workflow.chat(
            "What are the philosophical implications of AI?"
        )
        
        # Multi-persona discussion
        discussion = await workflow.multi_persona_discussion(
            topic="The future of human-AI collaboration",
            persona_ids=["gestalt-default", "sage", "pioneer"],
            rounds=2
        )
        
        # Compare persona responses
        comparison = await workflow.persona_comparison(
            "How should we approach learning new technologies?"
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_personas())