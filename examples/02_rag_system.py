"""RAG (Retrieval Augmented Generation) system example."""

import asyncio
from pathlib import Path
from czero_engine.workflows import KnowledgeBaseWorkflow, RAGWorkflow


async def rag_example():
    """Build and use a RAG system."""
    
    # Step 1: Create knowledge base from documents
    print("Step 1: Creating knowledge base...")
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
        
        print(f"   Created workspace: {result['workspace_id']}")
        print(f"   Processed {result['files_processed']} files")
        print(f"   Created {result['chunks_created']} chunks")
        print()
    
    # Step 2: Use RAG for Q&A
    print("Step 2: Using RAG for questions...")
    async with RAGWorkflow() as rag_workflow:
        
        # Ask questions with RAG
        questions = [
            "What is CZero Engine and what are its main features?",
            "How does semantic search work?",
            "What's the difference between AI, machine learning, and deep learning?",
            "Does CZero Engine support GPU acceleration?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            response = await rag_workflow.ask(
                question=question,
                chunk_limit=3,
                similarity_threshold=0.6
            )
            print(f"A{i}: {response.response[:300]}...")
            
            if response.context_used:
                print(f"   (Used {len(response.context_used)} context chunks)")
        
        print("\n" + "="*50)
        
        # Compare with and without RAG
        print("\nStep 3: Comparing with/without RAG...")
        comparison_q = "What document processing features does CZero Engine provide?"
        
        comparison = await rag_workflow.compare_with_without_rag(
            question=comparison_q
        )
        
        print(f"\nQuestion: {comparison_q}")
        print("\nWithout RAG:")
        print(f"  {comparison['without_rag'][:200]}...")
        print("\nWith RAG:")
        print(f"  {comparison['with_rag'][:200]}...")
        print(f"\n  Context chunks used: {comparison['chunks_used']}")


if __name__ == "__main__":
    asyncio.run(rag_example())