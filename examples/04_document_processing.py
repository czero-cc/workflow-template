"""Advanced document processing examples."""

import asyncio
from pathlib import Path
from czero_engine.workflows import DocumentProcessingWorkflow


async def document_processing_example():
    """Demonstrate document processing capabilities."""
    
    # Create sample project structure
    print("Setting up sample project structure...")
    project_root = Path("./sample_project")
    
    # Create directories
    (project_root / "src").mkdir(parents=True, exist_ok=True)
    (project_root / "docs").mkdir(parents=True, exist_ok=True)
    (project_root / "tests").mkdir(parents=True, exist_ok=True)
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    
    # Create sample files
    (project_root / "README.md").write_text("""
    # Sample Project
    
    This is a sample project for demonstrating CZero Engine's document processing.
    
    ## Features
    - Document extraction
    - Text chunking
    - Vector embeddings
    - Semantic search
    """)
    
    (project_root / "src" / "main.py").write_text("""
    # Main application file
    
    def process_documents(path):
        '''Process documents in the given path.'''
        print(f"Processing documents in {path}")
        # Implementation here
        return True
    
    def search(query, limit=10):
        '''Search for documents matching the query.'''
        results = []
        # Search implementation
        return results
    """)
    
    (project_root / "src" / "utils.py").write_text("""
    # Utility functions
    
    def chunk_text(text, chunk_size=1000, overlap=200):
        '''Split text into overlapping chunks.'''
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
    
    def calculate_similarity(vec1, vec2):
        '''Calculate cosine similarity between vectors.'''
        # Similarity calculation
        return 0.95
    """)
    
    (project_root / "docs" / "api.md").write_text("""
    # API Documentation
    
    ## Endpoints
    
    ### POST /api/process
    Process documents and create embeddings.
    
    ### GET /api/search
    Search for similar documents.
    
    ### POST /api/chat
    Chat with AI using document context.
    """)
    
    (project_root / "docs" / "architecture.txt").write_text("""
    System Architecture
    
    The system consists of three main components:
    1. Document Processor - Extracts and chunks text
    2. Embedding Service - Generates vector embeddings
    3. Search Engine - Performs semantic search
    
    Data flows from documents through the processor to the embedding service,
    and finally into the vector database for searching.
    """)
    
    (project_root / "tests" / "test_main.py").write_text("""
    import unittest
    from src.main import process_documents, search
    
    class TestDocumentProcessing(unittest.TestCase):
        def test_process_documents(self):
            result = process_documents("./test_data")
            self.assertTrue(result)
        
        def test_search(self):
            results = search("test query")
            self.assertIsInstance(results, list)
    """)
    
    print("Sample project structure created.\n")
    
    # Process documents
    async with DocumentProcessingWorkflow(verbose=True) as workflow:
        
        # 1. Discover files with filtering
        print("\n1. File Discovery:")
        print("="*50)
        
        all_files = workflow.discover_files(
            directory=str(project_root),
            patterns=["*.py", "*.md", "*.txt"],
            max_size_mb=10
        )
        
        print(f"\nFound {len(all_files)} files total")
        
        # 2. Process specific file types
        print("\n2. Processing Python Files:")
        print("="*50)
        
        python_files = [f for f in all_files if f.suffix == ".py"]
        
        if python_files:
            stats = await workflow.process_documents(
                files=python_files,
                workspace_name="Python Code",
                chunk_size=500,
                chunk_overlap=100,
                batch_size=5
            )
            
            print(f"\nPython files processed: {stats.processed_files}")
            print(f"Success rate: {stats.success_rate:.1f}%")
        
        # 3. Process documentation files
        print("\n3. Processing Documentation:")
        print("="*50)
        
        doc_files = [f for f in all_files if f.suffix in [".md", ".txt"]]
        
        if doc_files:
            stats = await workflow.process_documents(
                files=doc_files,
                workspace_name="Documentation",
                chunk_size=800,
                chunk_overlap=200
            )
            
            print(f"\nDoc files processed: {stats.processed_files}")
            print(f"Chunks created: {stats.total_chunks}")
        
        # 4. Process entire directory tree with organization
        print("\n4. Processing Directory Tree (Organized by Type):")
        print("="*50)
        
        workspace_stats = await workflow.process_directory_tree(
            root_directory=str(project_root),
            workspace_prefix="organized",
            organize_by_type=True,
            chunk_size=600,
            batch_size=3
        )
        
        print("\nWorkspace Summary:")
        total_processed = sum(s.processed_files for s in workspace_stats.values())
        total_chunks = sum(s.total_chunks for s in workspace_stats.values())
        print(f"  Total workspaces created: {len(workspace_stats)}")
        print(f"  Total files processed: {total_processed}")
        print(f"  Total chunks created: {total_chunks}")
        
        # 5. Generate embeddings for custom content
        print("\n5. Custom Embedding Generation:")
        print("="*50)
        
        custom_texts = [
            "CZero Engine provides powerful document processing capabilities",
            "Vector embeddings enable semantic understanding of text",
            "RAG systems combine retrieval with generation for accurate responses"
        ]
        
        embeddings = await workflow.generate_embeddings_for_text(custom_texts)
        
        print(f"\nGenerated {len(embeddings)} embeddings")
        for i, (text, emb) in enumerate(zip(custom_texts, embeddings), 1):
            print(f"  {i}. Text: '{text[:50]}...'")
            print(f"     Dimensions: {len(emb.embedding)}")


async def batch_processing_example():
    """Example of batch processing with progress tracking."""
    
    print("\nBatch Processing Example:")
    print("="*70)
    
    # Create a larger set of test files
    test_dir = Path("./batch_test")
    test_dir.mkdir(exist_ok=True)
    
    # Generate multiple test files
    for i in range(25):
        file_path = test_dir / f"document_{i:03d}.txt"
        content = f"""
        Document {i}
        
        This is test document number {i}. It contains sample text for processing.
        The document discusses various topics related to AI and machine learning.
        
        Key concepts covered:
        - Neural networks and deep learning
        - Natural language processing
        - Computer vision applications
        - Reinforcement learning strategies
        
        Each document is unique but shares common themes to test the processing
        and chunking capabilities of the system.
        """
        file_path.write_text(content * 3)  # Make files larger
    
    print(f"Created 25 test documents in {test_dir}")
    
    async with DocumentProcessingWorkflow(verbose=True) as workflow:
        
        # Discover all files
        files = workflow.discover_files(
            directory=str(test_dir),
            patterns=["*.txt"]
        )
        
        print(f"\nProcessing {len(files)} files in batches...")
        
        # Process in batches with progress
        stats = await workflow.process_documents(
            files=files,
            workspace_name="Batch Processing Test",
            batch_size=5,  # Process 5 files at a time
            chunk_size=300,
            chunk_overlap=50
        )
        
        print("\nBatch Processing Results:")
        print(f"  Total files: {stats.total_files}")
        print(f"  Successfully processed: {stats.processed_files}")
        print(f"  Failed: {stats.failed_files}")
        print(f"  Success rate: {stats.success_rate:.1f}%")
        print(f"  Total chunks: {stats.total_chunks}")
        print(f"  Processing time: {stats.processing_time:.2f} seconds")
        
        if stats.processing_time > 0:
            throughput = stats.total_size_bytes / (1024 * 1024) / stats.processing_time
            print(f"  Throughput: {throughput:.2f} MB/s")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print(f"\nCleaned up test directory: {test_dir}")


if __name__ == "__main__":
    print("Running document processing examples...")
    asyncio.run(document_processing_example())
    
    print("\n\n" + "="*70)
    asyncio.run(batch_processing_example())