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
    
    workspace_id = None
    
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
        
        workspace_id = result['workspace']['id']
        print(f"✅ Created workspace: {result['workspace']['name']}")
        print(f"   ID: {workspace_id}")
        print(f"   Processed {result['files_processed']} files")
        print(f"   Created {result['chunks_created']} chunks")
        
        # Important: Give the system time to index the documents
        print("\n⏳ Waiting for indexing to complete...")
        await asyncio.sleep(5)  # Wait 5 seconds for indexing
    
    # Step 2: Verify documents are indexed by searching
    print("\n2. Verifying Document Indexing")
    print("-" * 30)
    async with CZeroEngineClient() as client:
        # Search for content we just indexed, using workspace filter
        test_results = await client.semantic_search(
            query="artificial intelligence",
            limit=3,
            similarity_threshold=0.3,  # Low threshold to ensure we find something
            workspace_filter=workspace_id  # Search only in our workspace
        )
        
        if test_results.results:
            print(f"✅ Found {len(test_results.results)} indexed documents")
            for i, res in enumerate(test_results.results, 1):
                print(f"  {i}. Score: {res.similarity:.3f}")
                print(f"     {res.content[:100]}...")
        else:
            print("⚠️  Documents may still be indexing. Continuing with example...")
    
    # Step 3: Use RAG for Q&A
    print("\n3. RAG-Enhanced Q&A")
    print("-" * 30)
    print("Note: RAG searches across all workspaces, not just the one we created.")
    async with RAGWorkflow() as rag_workflow:
        
        # Ask questions that match our indexed documents
        questions = [
            "What is artificial intelligence and machine learning?",
            "How does semantic search work?",
            "What features does CZero Engine provide?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Q{i}: {question}")
            response = await rag_workflow.ask(
                question=question,
                chunk_limit=3,
                similarity_threshold=0.3  # Lower threshold to be more inclusive
            )
            print(f"💡 A{i}: {response.response[:250]}...")
            
            if response.context_used:
                print(f"   📚 Used {len(response.context_used)} context chunks")
                for j, ctx in enumerate(response.context_used[:2], 1):
                    print(f"      {j}. {ctx.content[:60]}...")
        
    # Step 4: Compare with and without RAG
    print("\n4. RAG vs Non-RAG Comparison")
    print("-" * 30)
    comparison_q = "What is machine learning and how does it relate to AI?"
    
    async with RAGWorkflow() as rag_workflow:
        comparison = await rag_workflow.compare_with_without_rag(
            question=comparison_q
        )
        
        print(f"\n🤔 Question: {comparison_q}")
        print("\n❌ Without RAG (generic response):")
        print(f"   {comparison['without_rag'].response[:200]}...")
        print("\n✅ With RAG (context-aware):")
        print(f"   {comparison['with_rag'].response[:200]}...")
        print(f"\n📊 Statistics:")
        chunks_used = len(comparison['with_rag'].context_used) if comparison['with_rag'].context_used else 0
        print(f"   Context chunks used: {chunks_used}")
        print(f"   Improvement: More specific and accurate with RAG")
    
    # Step 5: Find similar content
    print("\n5. Similarity Search")
    print("-" * 30)
    async with CZeroEngineClient() as client:
        # Search in our workspace for semantic search content
        search_res = await client.semantic_search(
            query="semantic search",
            limit=1,
            workspace_filter=workspace_id
        )
        
        if search_res.results:
            chunk_id = search_res.results[0].chunk_id
            similar = await client.similarity_search(
                chunk_id=chunk_id,
                limit=3,
                similarity_threshold=0.3  # Lower threshold
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