"""Examples of interacting with AI personas."""

import asyncio
from czero_engine.workflows import PersonaWorkflow


async def persona_examples():
    """Demonstrate persona interactions."""
    
    async with PersonaWorkflow() as workflow:
        
        # 1. List available personas
        print("1. Available Personas:")
        print("="*50)
        await workflow.list_personas()
        print()
        
        # 2. Chat with different personas
        print("2. Individual Persona Chats:")
        print("="*50)
        
        # Chat with Gestalt
        print("\n--- Gestalt (Adaptive Assistant) ---")
        await workflow.select_persona("gestalt-default")
        
        response = await workflow.chat(
            "Hello! Can you introduce yourself and explain what makes you unique?"
        )
        
        # Continue conversation
        await workflow.chat(
            "How would you help someone learn about AI and machine learning?"
        )
        
        # Chat with Sage
        print("\n--- Sage (Research & Analysis) ---")
        await workflow.select_persona("sage")
        
        await workflow.chat(
            "What are the philosophical implications of AGI (Artificial General Intelligence)?"
        )
        
        # Chat with Pioneer
        print("\n--- Pioneer (Innovation) ---")
        await workflow.select_persona("pioneer")
        
        await workflow.chat(
            "What innovative applications could combine AR/VR with AI?"
        )
        
        # 3. Multi-persona discussion
        print("\n3. Multi-Persona Discussion:")
        print("="*50)
        
        discussion = await workflow.multi_persona_discussion(
            topic="The role of AI in education: opportunities and challenges",
            persona_ids=["gestalt-default", "sage", "pioneer"],
            rounds=2
        )
        
        print("\nDiscussion Summary:")
        for entry in discussion:
            print(f"\nRound {entry['round']} - {entry['persona']}:")
            print(f"  {entry['response'][:200]}...")
        
        # 4. Persona comparison on same question
        print("\n4. Persona Comparison:")
        print("="*50)
        
        question = "How should we approach the ethics of AI development?"
        print(f"\nQuestion: {question}\n")
        
        responses = await workflow.persona_comparison(
            question=question,
            persona_ids=["gestalt-default", "sage", "pioneer"]
        )
        
        for persona_id, response in responses.items():
            print(f"\n{persona_id}:")
            print(f"  {response.response[:250]}...")
        
        # 5. Get conversation summary
        print("\n5. Conversation Summary:")
        print("="*50)
        
        # Switch back to Gestalt to check conversation history
        await workflow.select_persona("gestalt-default")
        summary = workflow.get_conversation_summary()
        
        print(f"\nActive persona: {summary['persona']}")
        print(f"Total turns: {summary['turn_count']}")
        print(f"Message count: {summary['message_count']}")
        
        if summary['recent_messages']:
            print("\nRecent messages:")
            for msg in summary['recent_messages'][-4:]:
                role = msg['role'].capitalize()
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"  {role}: {content}")


async def interactive_persona_chat():
    """Interactive chat example with a persona."""
    
    async with PersonaWorkflow() as workflow:
        print("Starting interactive chat with Gestalt...")
        print("="*50)
        
        await workflow.select_persona("gestalt-default")
        
        # Simulate a conversation
        messages = [
            "Hello! I'm interested in learning about vector databases.",
            "What makes them different from traditional databases?",
            "Can you give me a practical example of when to use one?",
            "How do they relate to RAG systems?",
            "Thank you for the explanation!"
        ]
        
        for message in messages:
            print(f"\nYou: {message}")
            response = await workflow.chat(
                message=message,
                maintain_history=True
            )
            # Response is printed by the workflow if verbose=True
            
            # Small delay to simulate conversation flow
            await asyncio.sleep(0.5)
        
        # Show final conversation summary
        print("\n" + "="*50)
        summary = workflow.get_conversation_summary()
        print(f"Conversation ended with {summary['turn_count']} turns")


if __name__ == "__main__":
    print("Running persona examples...")
    asyncio.run(persona_examples())
    
    print("\n\n" + "="*70)
    print("Running interactive chat example...")
    print("="*70)
    asyncio.run(interactive_persona_chat())