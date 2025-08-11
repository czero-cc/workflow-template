"""Basic Usage Example - CZero Engine Python SDK

This example demonstrates fundamental API operations:
- Health checks and status monitoring
- Chat interactions with LLM
- Embedding generation
- Workspace management
- Persona discovery
"""

import asyncio
from czero_engine import CZeroEngineClient


async def basic_examples():
    """Demonstrate basic SDK usage."""
    
    # Initialize client with default settings (localhost:1421)
    async with CZeroEngineClient(verbose=True) as client:
        
        # 1. Health check - verify API and models are ready
        print("\n1. Health Check")
        print("=" * 40)
        health = await client.health_check()
        print(f"✅ Status: {health.status}")
        print(f"   Version: {health.version}")
        print(f"   Service: {health.service}")
        
        # 2. Simple chat without RAG - direct LLM interaction
        print("\n2. Chat (No RAG)")
        print("=" * 40)
        response = await client.chat(
            message="Explain quantum computing in one sentence.",
            use_rag=False,
            max_tokens=100
        )
        print(f"Response: {response.response}")
        print(f"Model: {response.model_used}")
        
        # 3. Generate embeddings for semantic similarity
        print("\n3. Embedding Generation")
        print("=" * 40)
        embedding = await client.generate_embedding(
            text="Advanced AI document processing with semantic search"
        )
        print(f"✅ Generated {len(embedding.embedding)}-dimensional embedding")
        print(f"   Model: {embedding.model_used}")
        print(f"   Sample values: [{embedding.embedding[0]:.4f}, {embedding.embedding[1]:.4f}, ...]")
        
        # 4. Create a workspace for document organization
        print("\n4. Workspace Creation")
        print("=" * 40)
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = await client.create_workspace(
                name="Demo Workspace",
                path=temp_dir,
                description="Example workspace for SDK demonstration"
            )
            print(f"✅ Created workspace: {workspace.name}")
            print(f"   ID: {workspace.id}")
            print(f"   Path: {workspace.path}")
        
        # 5. Discover available AI personas
        print("\n5. AI Personas")
        print("=" * 40)
        personas = await client.list_personas()
        print(f"Found {len(personas.personas)} personas:")
        for persona in personas.personas[:5]:  # Show first 5
            print(f"  • {persona.name}: {persona.specialty}")
            if persona.tagline:
                print(f"    {persona.tagline[:60]}...")


async def main():
    """Run basic examples with error handling."""
    try:
        await basic_examples()
        print("\n✅ All basic examples completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. CZero Engine is running")
        print("2. API server is started (port 1421)")
        print("3. Models are loaded in the app")


if __name__ == "__main__":
    asyncio.run(main())