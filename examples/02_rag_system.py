"""RAG (Retrieval-Augmented Generation) Example - CZero Engine

This example demonstrates:
- Document processing and chunking
- Semantic search with hierarchical support
- Chat with RAG context
- Similarity-based recommendations
- Comparing responses with and without RAG
"""

import asyncio
from pathlib import Path
from czero_engine import CZeroEngineClient
from czero_engine.workflows import KnowledgeBaseWorkflow, RAGWorkflow


async def rag_example():
    """Build and use a RAG system with enhanced features."""
    
    print("\n🚀 RAG System Example")
    print("=" * 50)
    
    # Step 1: Create knowledge base from documents
    print("\n1. Creating Knowledge Base")
    print("-" * 30)
    async with KnowledgeBaseWorkflow() as kb_workflow:
        
        # Ensure we have a documents directory
        docs_dir = Path("./sample_docs")
        docs_dir.mkdir(exist_ok=True)
        
        # Create some sample documents
        (docs_dir / "ai_basics.txt").write_text("""
        Artificial Intelligence (AI) refers to the simulation of human intelligence 
        in machines. Machine learning is a subset of AI that enables systems to 
        learn from data. Deep learning uses neural networks with multiple layers 
        to process complex patterns. Natural language processing helps computers 
        understand human language.
        """)
        
        (docs_dir / "czero_engine.md").write_text("""
        # CZero Engine Overview
        
        CZero Engine is a comprehensive document processing and RAG system. 
        It provides:
        - Document extraction and chunking
        - Vector embeddings for semantic search
        - Integration with multiple LLM backends
        - AI personas for specialized interactions
        - Workspace management for organizing documents
        
        The system uses ONNX Runtime for efficient model inference and supports
        GPU acceleration for faster processing.
        """)
        
        (docs_dir / "semantic_search.txt").write_text("""
        Semantic search goes beyond keyword matching to understand the meaning 
        and intent behind queries. It uses vector embeddings to represent text 
        as high-dimensional vectors. Similar content has vectors that are close 
        together in the vector space. This enables finding relevant information 
        even when exact keywords don't match.
        """)
        
        # Create knowledge base
        result = await kb_workflow.create_knowledge_base(
            name="AI Documentation",
            directory_path=str(docs_dir),
            file_patterns=["*.txt", "*.md"],
            chunk_size=500,
            chunk_overlap=50
        )
        
        print(f"✅ Created workspace: {result['workspace']['name']}")
        print(f"   ID: {result['workspace']['id']}")
        print(f"   Processed {result['files_processed']} files")
        print(f"   Created {result['chunks_created']} chunks")
    
    # Step 2: Demonstrate hierarchical search
    print("\n2. Hierarchical Semantic Search")
    print("-" * 30)
    async with CZeroEngineClient() as client:
        # Search with hierarchy support
        results = await client.semantic_search(
            query="How does AI and machine learning work?",
            limit=3,
            include_hierarchy=True,
            hierarchy_level=None  # Search all levels
        )
        
        print(f"Found {len(results.results)} results with hierarchy:")
        for i, res in enumerate(results.results, 1):
            print(f"\n  {i}. Score: {res.similarity:.3f}")
            print(f"     {res.content[:100]}...")
            if res.parent_chunk:
                print(f"     ↳ Has parent context")
    
    # Step 3: Use RAG for Q&A
    print("\n3. RAG-Enhanced Q&A")
    print("-" * 30)
    async with RAGWorkflow() as rag_workflow:
        
        # Ask questions with RAG
        questions = [
            "What is CZero Engine and what are its main features?",
            "How does semantic search work?",
            "What's the difference between AI, machine learning, and deep learning?",
            "Does CZero Engine support GPU acceleration?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Q{i}: {question}")
            response = await rag_workflow.ask(
                question=question,
                chunk_limit=3,
                similarity_threshold=0.5
            )
            print(f"💡 A{i}: {response.response[:250]}...")
            
            if response.context_used:
                print(f"   📚 Used {len(response.context_used)} context chunks")
                for j, ctx in enumerate(response.context_used[:2], 1):
                    print(f"      {j}. {ctx.content[:60]}...")
        
    # Step 4: Compare with and without RAG
    print("\n4. RAG vs Non-RAG Comparison")
    print("-" * 30)
    comparison_q = "What document processing features does CZero Engine provide?"
    
    async with RAGWorkflow() as rag_workflow:
        comparison = await rag_workflow.compare_with_without_rag(
            question=comparison_q
        )
        
        print(f"\n🤔 Question: {comparison_q}")
        print("\n❌ Without RAG (generic response):")
        print(f"   {comparison['without_rag'][:200]}...")
        print("\n✅ With RAG (context-aware):")
        print(f"   {comparison['with_rag'][:200]}...")
        print(f"\n📊 Statistics:")
        print(f"   Context chunks used: {comparison['chunks_used']}")
        print(f"   Improvement: More specific and accurate with RAG")
    
    # Step 5: Find similar content
    print("\n5. Similarity Search")
    print("-" * 30)
    async with CZeroEngineClient() as client:
        # Get all chunks first
        search_res = await client.semantic_search(
            query="semantic search",
            limit=1
        )
        
        if search_res.results:
            chunk_id = search_res.results[0].chunk_id
            similar = await client.similarity_search(
                chunk_id=chunk_id,
                limit=3
            )
            
            print(f"Content similar to chunk '{chunk_id[:20]}...':\n")
            for i, res in enumerate(similar.results, 1):
                print(f"  {i}. Score: {res.similarity:.3f}")
                print(f"     {res.content[:80]}...")


async def main():
    """Run RAG examples with error handling."""
    try:
        await rag_example()
        print("\n✅ RAG examples completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure CZero Engine is running")
        print("2. Check that API server is active")
        print("3. Verify embedding models are loaded")
        print("4. Confirm vector database is initialized")


if __name__ == "__main__":
    asyncio.run(main())