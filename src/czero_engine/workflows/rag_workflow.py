"""RAG (Retrieval Augmented Generation) workflow for CZero Engine."""

from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from ..client import CZeroEngineClient
from ..models import ChatResponse, SemanticSearchResponse

console = Console()


class RAGWorkflow:
    """
    Workflow for Retrieval Augmented Generation using CZero Engine.
    
    This workflow combines semantic search with LLM generation to provide
    accurate, context-aware responses based on your document knowledge base.
    """
    
    def __init__(self, client: Optional[CZeroEngineClient] = None, verbose: bool = True):
        """
        Initialize RAG workflow.
        
        Args:
            client: Optional CZeroEngineClient instance
            verbose: Enable verbose output
        """
        self.client = client
        self._owns_client = client is None
        self.verbose = verbose
        self.last_response: Optional[ChatResponse] = None
        self.last_search: Optional[SemanticSearchResponse] = None
        
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
            
    async def ask(
        self,
        question: str,
        similarity_threshold: float = 0.7,
        chunk_limit: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        workspace_filter: Optional[str] = None
    ) -> ChatResponse:
        """
        Ask a question using RAG - retrieves relevant context then generates response.
        
        Args:
            question: The question to ask
            similarity_threshold: Minimum similarity for context chunks
            chunk_limit: Maximum number of context chunks to use
            max_tokens: Maximum tokens in response
            temperature: LLM temperature (0.0-1.0)
            system_prompt: Optional system prompt
            workspace_filter: Optional workspace to search
            
        Returns:
            Chat response with answer and context used
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        if self.verbose:
            console.print(Panel(
                f"[bold cyan]RAG Query[/bold cyan]\n{question}",
                expand=False
            ))
            
        # Use the chat endpoint with RAG enabled
        response = await self.client.chat(
            message=question,
            use_rag=True,
            system_prompt=system_prompt or "You are a helpful assistant. Answer based on the provided context.",
            max_tokens=max_tokens,
            temperature=temperature,
            similarity_threshold=similarity_threshold,
            chunk_limit=chunk_limit
        )
        
        self.last_response = response
        
        if self.verbose:
            # Display response
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Panel(Markdown(response.response), expand=False))
            
            # Display context used if available
            if response.context_used:
                console.print(f"\n[dim]Context: {len(response.context_used)} chunks used[/dim]")
                
                context_table = Table(title="Context Sources")
                context_table.add_column("Score", style="cyan", width=10)
                context_table.add_column("Content", style="dim", width=80)
                
                for chunk in response.context_used[:3]:
                    content_preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                    context_table.add_row(
                        f"{chunk.similarity:.3f}",
                        content_preview
                    )
                    
                console.print(context_table)
                
        return response
        
    async def search_then_ask(
        self,
        search_query: str,
        question: Optional[str] = None,
        search_limit: int = 10,
        use_top_n: int = 5,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        First search for relevant content, then ask a question about it.
        
        Useful when you want to see the search results before generating an answer.
        
        Args:
            search_query: Query for semantic search
            question: Optional different question to ask (uses search_query if not provided)
            search_limit: Number of search results to retrieve
            use_top_n: Number of top results to use for context
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary with search results and generated answer
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        # Step 1: Search
        if self.verbose:
            console.print(f"\n[cyan]🔍 Searching: {search_query}[/cyan]")
            
        search_results = await self.client.semantic_search(
            query=search_query,
            limit=search_limit,
            similarity_threshold=similarity_threshold
        )
        
        self.last_search = search_results
        
        if self.verbose:
            console.print(f"[green]✓[/green] Found {len(search_results.results)} results")
            
        if not search_results.results:
            console.print("[yellow]No relevant content found[/yellow]")
            return {
                "search_results": search_results.model_dump(),
                "answer": None
            }
            
        # Step 2: Build context from top results
        context_chunks = []
        for result in search_results.results[:use_top_n]:
            context_chunks.append(result.content)
            
        context = "\n\n".join(context_chunks)
        
        # Step 3: Ask question with context
        final_question = question or search_query
        prompt_with_context = f"""Based on the following context, answer this question: {final_question}

Context:
{context}

Answer:"""
        
        if self.verbose:
            console.print(f"\n[cyan]💬 Generating answer using top {use_top_n} results...[/cyan]")
            
        # Use chat without RAG since we already have context
        response = await self.client.chat(
            message=prompt_with_context,
            use_rag=False,  # We already have the context
            max_tokens=1024
        )
        
        if self.verbose:
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Panel(Markdown(response.response), expand=False))
            
        return {
            "search_results": search_results.model_dump(),
            "answer": response.model_dump(),
            "context_used": use_top_n
        }
        
    async def iterative_refinement(
        self,
        initial_question: str,
        max_iterations: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[ChatResponse]:
        """
        Iteratively refine answers by asking follow-up questions.
        
        Args:
            initial_question: Starting question
            max_iterations: Maximum refinement iterations
            similarity_threshold: Minimum similarity for context
            
        Returns:
            List of responses from each iteration
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        responses = []
        current_question = initial_question
        
        for iteration in range(max_iterations):
            if self.verbose:
                console.print(f"\n[cyan]Iteration {iteration + 1}/{max_iterations}[/cyan]")
                console.print(f"Question: {current_question}")
                
            # Get response
            response = await self.ask(
                current_question,
                similarity_threshold=similarity_threshold
            )
            responses.append(response)
            
            # Generate follow-up question if not last iteration
            if iteration < max_iterations - 1:
                follow_up_prompt = f"""Based on this answer: "{response.response}"
                
What follow-up question would help clarify or expand on this topic? 
Provide only the question, nothing else."""
                
                follow_up_response = await self.client.chat(
                    message=follow_up_prompt,
                    use_rag=False,
                    max_tokens=100
                )
                
                current_question = follow_up_response.response
                
        return responses
        
    async def compare_with_without_rag(
        self,
        question: str,
        **kwargs
    ) -> Dict[str, ChatResponse]:
        """
        Compare responses with and without RAG for the same question.
        
        Useful for demonstrating the value of RAG.
        
        Args:
            question: Question to ask
            **kwargs: Additional arguments for chat
            
        Returns:
            Dictionary with both responses
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        if self.verbose:
            console.print(Panel(
                f"[bold cyan]RAG Comparison[/bold cyan]\n{question}",
                expand=False
            ))
            
        # Response with RAG
        if self.verbose:
            console.print("\n[cyan]With RAG (using knowledge base):[/cyan]")
            
        with_rag = await self.client.chat(
            message=question,
            use_rag=True,
            **kwargs
        )
        
        if self.verbose:
            console.print(Panel(
                Markdown(with_rag.response),
                title="With RAG",
                border_style="green"
            ))
            if with_rag.context_used:
                console.print(f"[dim]Used {len(with_rag.context_used)} context chunks[/dim]")
                
        # Response without RAG
        if self.verbose:
            console.print("\n[cyan]Without RAG (LLM only):[/cyan]")
            
        without_rag = await self.client.chat(
            message=question,
            use_rag=False,
            **kwargs
        )
        
        if self.verbose:
            console.print(Panel(
                Markdown(without_rag.response),
                title="Without RAG",
                border_style="yellow"
            ))
            
        return {
            "with_rag": with_rag,
            "without_rag": without_rag
        }


# Example usage
async def example_rag():
    """Example of using RAG workflow."""
    
    async with RAGWorkflow() as workflow:
        # Simple RAG question
        response = await workflow.ask(
            "What is CZero Engine and what are its main features?",
            chunk_limit=5
        )
        
        # Search then ask
        result = await workflow.search_then_ask(
            search_query="vector embeddings semantic search",
            question="How does semantic search work with embeddings?",
            use_top_n=3
        )
        
        # Compare with and without RAG
        comparison = await workflow.compare_with_without_rag(
            "Explain the document processing pipeline",
            max_tokens=500
        )
        
        # Iterative refinement
        responses = await workflow.iterative_refinement(
            "What is RAG?",
            max_iterations=2
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_rag())