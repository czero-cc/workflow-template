"""AI Persona Interactions Example - CZero Engine

This example demonstrates:
- Listing and discovering AI personas
- Chatting with specific personas
- Multi-persona discussions
- Persona comparison on same topics
- Maintaining conversation history
"""

import asyncio
from czero_engine import CZeroEngineClient
from czero_engine.workflows import PersonaWorkflow


async def persona_examples():
    """Demonstrate persona interactions and capabilities."""
    
    print("\n🤖 AI Persona Examples")
    print("=" * 50)
    
    # 1. Discover available personas using direct API
    print("\n1. Available Personas")
    print("-" * 30)
    
    async with CZeroEngineClient() as client:
        personas = await client.list_personas()
        print(f"Found {len(personas.personas)} personas:\n")
        
        for persona in personas.personas[:5]:  # Show first 5
            print(f"  📌 {persona.name}")
            print(f"     Specialty: {persona.specialty}")
            if persona.tagline:
                print(f"     Tagline: {persona.tagline[:80]}...")
            print()
    
    # 2. Chat with Gestalt persona  
    print("\n2. Gestalt Persona Chat")
    print("-" * 30)
    
    async with PersonaWorkflow(verbose=False) as workflow:
        # Chat with Gestalt (Adaptive Intelligence)
        print("\n💬 Gestalt - Adaptive Intelligence")
        await workflow.select_persona("gestalt-default")
        
        # Ask multiple questions to show versatility
        questions = [
            "Explain quantum computing in simple terms",
            "What are the latest breakthroughs in AI safety research?",
            "What innovative applications could combine blockchain with AI?"
        ]
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            response = await workflow.chat(
                message=question,
                max_tokens=100  # Moderate response length
            )
            print(f"💡 Response: {response.response[:250]}...")
    
    # 3. Conversation with context
    print("\n3. Contextual Conversation")
    print("-" * 30)
    
    async with PersonaWorkflow(verbose=False) as workflow:
        # Have a multi-turn conversation with Gestalt
        print("\n🎭 Multi-turn conversation with Gestalt\n")
        
        await workflow.select_persona("gestalt-default")
        
        # Simulate a conversation about a specific topic
        conversation_flow = [
            "I want to learn about machine learning",
            "What are neural networks?",
            "How do they learn from data?"
        ]
        
        for i, message in enumerate(conversation_flow, 1):
            print(f"Turn {i} - You: {message}")
            response = await workflow.chat(
                message=message,
                maintain_history=True,  # Keep conversation context
                max_tokens=100  # Moderate response
            )
            print(f"Turn {i} - Gestalt: {response.response[:200]}...")
            print()
    
    # 4. Different conversation styles with Gestalt
    print("\n4. Exploring Gestalt's Versatility")
    print("-" * 30)
    
    async with PersonaWorkflow(verbose=False) as workflow:
        print("\n❓ Testing different types of queries with Gestalt\n")
        
        await workflow.select_persona("gestalt-default")
        
        # Different types of queries to show Gestalt's adaptability
        query_types = [
            ("Technical", "How should we balance AI innovation with ethical considerations?"),
            ("Creative", "Write a haiku about artificial intelligence"),
            ("Analytical", "What are the pros and cons of remote work?")
        ]
        
        for query_type, question in query_types:
            print(f"🔹 {query_type} Query: {question}")
            response = await workflow.chat(
                message=question,
                maintain_history=False,  # Fresh context for each
                max_tokens=100  # Shorter responses for variety
            )
            print(f"   Response: {response.response[:200]}...")
            print()


async def interactive_chat_example():
    """Demonstrate interactive conversation with context."""
    
    print("\n5. Interactive Conversation")
    print("-" * 30)
    
    async with PersonaWorkflow(verbose=False) as workflow:
        print("\n💬 Starting conversation with Gestalt...\n")
        
        await workflow.select_persona("gestalt-default")
        
        # Simulate a multi-turn conversation
        conversation = [
            "I'm building a RAG system. What are the key components I need?",
            "How do I choose the right embedding model?",
            "What chunk size and overlap should I use?",
            "How can I evaluate the quality of my RAG responses?",
        ]
        
        for i, message in enumerate(conversation, 1):
            print(f"👤 You: {message}")
            
            response = await workflow.chat(
                message=message,
                maintain_history=True,
                max_tokens=100  # Moderate response
            )
            
            print(f"🤖 Gestalt: {response.response[:300]}...")
            print()
            
            # Small delay for readability
            await asyncio.sleep(0.3)
        
        # Get conversation summary
        summary = workflow.get_conversation_summary()
        print(f"📊 Conversation Summary:")
        print(f"   Total turns: {summary['turn_count']}")
        print(f"   Messages: {summary['message_count']}")
        print(f"   Active persona: {summary['persona']}")


async def persona_with_rag():
    """Demonstrate personas using RAG context."""
    
    print("\n6. Persona + RAG Integration")
    print("-" * 30)
    
    async with CZeroEngineClient() as client:
        # First, list workspaces to find one with documents
        workspaces = await client.list_workspaces()
        workspace_id = None
        
        if workspaces.workspaces:
            # Use the first available workspace
            workspace_id = workspaces.workspaces[0].id
            print(f"📁 Using workspace: {workspaces.workspaces[0].name}")
        else:
            print("⚠️ No workspaces found. Creating a sample workspace...")
            # Create a sample workspace if none exist
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                workspace = await client.create_workspace(
                    name="Sample Workspace",
                    path=temp_dir
                )
                workspace_id = workspace.id
        
        # Use persona chat with RAG context from workspace
        print("\n🔍 Asking Gestalt with document context...\n")
        
        response = await client.persona_chat(
            persona_id="gestalt-default",  # Use real persona
            message="Based on the documents, what are the key features of CZero Engine?",
            workspace_filter=workspace_id,  # Enable RAG with this workspace
            max_tokens=100  # Moderate response
        )
        
        print(f"Response: {response.response[:400]}...")
        print(f"Timestamp: {response.timestamp}")


async def main():
    """Run all persona examples with error handling."""
    try:
        await persona_examples()
        await interactive_chat_example()
        await persona_with_rag()
        print("\n✅ All persona examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure CZero Engine is running")
        print("2. Check API server is active")
        print("3. Verify personas are loaded")
        print("4. Check LLM models are available")


if __name__ == "__main__":
    asyncio.run(main())