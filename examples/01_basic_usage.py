"""Basic usage examples for CZero Engine Python SDK."""

import asyncio
from czero_engine import CZeroEngineClient


async def basic_examples():
    """Demonstrate basic SDK usage."""
    
    # Initialize client
    async with CZeroEngineClient() as client:
        
        # 1. Health check
        print("1. Checking API health...")
        health = await client.health_check()
        print(f"   Status: {health['status']}")
        print()
        
        # 2. Simple chat without RAG
        print("2. Chat without RAG...")
        response = await client.chat(
            message="What is machine learning?",
            use_rag=False
        )
        print(f"   Response: {response.response[:200]}...")
        print()
        
        # 3. Create a workspace
        print("3. Creating workspace...")
        workspace = await client.create_workspace(
            name="Example Workspace",
            path="./documents",
            description="Test workspace for examples"
        )
        print(f"   Workspace ID: {workspace.id}")
        print()
        
        # 4. Generate embedding
        print("4. Generating embedding...")
        embedding = await client.generate_embedding(
            text="CZero Engine is a powerful document processing system"
        )
        print(f"   Embedding dimensions: {len(embedding.embedding)}")
        print(f"   First 5 values: {embedding.embedding[:5]}")
        print()
        
        # 5. List personas
        print("5. Available personas...")
        personas = await client.list_personas()
        for persona in personas.personas:
            print(f"   - {persona.name}: {persona.specialty}")


if __name__ == "__main__":
    asyncio.run(basic_examples())