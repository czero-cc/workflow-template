"""Document Processing Example - CZero Engine

This example demonstrates:
- Workspace creation and management
- Document extraction and chunking
- Batch processing with progress tracking
- File discovery and filtering
- Hierarchical document organization (SmallToBig chunking)
- Custom embedding generation

IMPORTANT: CZero Engine uses SmallToBig hierarchical chunking by default.
This means each document is processed into BOTH parent chunks (e.g., paragraphs)
AND child chunks (smaller segments within each parent). This creates multiple
chunks per file for better semantic search - typically 4x the number of files.

The 'files_processed' count returned by the API actually represents the total
number of chunking operations (parent + child chunks), not the number of files.
This is why you may see "400%" success rates - it's counting all chunks created.
"""

import asyncio
from pathlib import Path
import tempfile
import shutil
from czero_engine import CZeroEngineClient
from czero_engine.workflows import DocumentProcessingWorkflow


async def basic_document_processing():
    """Demonstrate basic document processing pipeline."""
    
    print("\n📄 Document Processing Example")
    print("=" * 50)
    
    async with CZeroEngineClient() as client:
        # Create a temporary directory with sample documents
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample documents
            print("\n1. Creating Sample Documents")
            print("-" * 30)
            
            # Technical document
            tech_doc = temp_path / "technical_guide.md"
            tech_doc.write_text("""
            # Technical Implementation Guide
            
            ## Architecture Overview
            The system uses a microservices architecture with the following components:
            - API Gateway for request routing
            - Processing Service for document analysis
            - Vector Database for semantic search
            - LLM Service for text generation
            
            ## Key Features
            1. Real-time document processing
            2. Semantic search with sub-second latency
            3. Multi-model support (LLM and embeddings)
            4. Horizontal scaling capabilities
            
            ## Performance Metrics
            - Document processing: 100 docs/minute
            - Search latency: <200ms p99
            - Embedding generation: 1000 tokens/second
            """)
            
            # Business document
            business_doc = temp_path / "business_report.txt"
            business_doc.write_text("""
            Q4 2024 Business Report
            
            Executive Summary:
            This quarter demonstrated strong growth in AI adoption across enterprises.
            Key achievements include launching three new AI products and expanding
            our customer base by 45%. Revenue increased by 32% year-over-year.
            
            Market Analysis:
            The AI market continues to expand rapidly with enterprises investing
            heavily in automation and intelligent document processing. Our RAG
            solution has gained significant traction in the financial sector.
            
            Future Outlook:
            We expect continued growth driven by demand for enterprise AI solutions.
            Investment in R&D will focus on improving model accuracy and reducing
            operational costs through optimization.
            """)
            
            # Code sample
            code_doc = temp_path / "example_code.py"
            code_doc.write_text('''
            def process_documents(documents, chunk_size=500):
                """Process a list of documents into chunks."""
                all_chunks = []
                for doc in documents:
                    chunks = create_chunks(doc.content, chunk_size)
                    all_chunks.extend(chunks)
                return all_chunks
            
            def create_chunks(text, size):
                """Split text into overlapping chunks."""
                chunks = []
                words = text.split()
                for i in range(0, len(words), size):
                    chunk = ' '.join(words[i:i+size])
                    chunks.append(chunk)
                return chunks
            ''')
            
            print(f"✅ Created 3 sample documents in {temp_dir}")
            
            # 2. Create workspace and process documents
            print("\n2. Creating Workspace")
            print("-" * 30)
            
            workspace = await client.create_workspace(
                name="Document Processing Demo",
                path=str(temp_path),
                description="Demonstration of document processing capabilities"
            )
            
            print(f"✅ Created workspace: {workspace.name}")
            print(f"   ID: {workspace.id}")
            
            # 3. Process the documents
            print("\n3. Processing Documents")
            print("-" * 30)
            
            files = [str(tech_doc), str(business_doc), str(code_doc)]
            result = await client.process_files(
                workspace_id=workspace.id,
                files=files
            )
            
            print(f"✅ Processing complete:")
            print(f"   Files submitted: {len(files)}")
            print(f"   Total chunks created: {result.chunks_created}")  # Includes parent + child chunks
            print(f"   Processing operations: {result.files_processed}")  # Total chunking operations
            print(f"   Processing time: {result.processing_time:.2f}s")
            print(f"\n   Note: CZero Engine uses hierarchical SmallToBig chunking by default,")
            print(f"   creating both parent and child chunks for better semantic search.")
            
            # 4. Search the processed documents
            print("\n4. Searching Documents")
            print("-" * 30)
            
            queries = [
                "microservices architecture components",
                "business growth revenue",
                "document processing chunks"
            ]
            
            for query in queries:
                results = await client.semantic_search(
                    query=query,
                    limit=2,
                    similarity_threshold=0.3
                )
                
                print(f"\n🔍 Query: '{query}'")
                print(f"   Found {len(results.results)} matches:")
                for i, res in enumerate(results.results, 1):
                    print(f"   {i}. Score: {res.similarity:.3f}")
                    print(f"      {res.content[:100]}...")


async def advanced_processing_workflow():
    """Demonstrate advanced document processing with workflow."""
    
    print("\n📚 Advanced Document Processing")
    print("=" * 50)
    
    # Create a project structure
    project_dir = Path("./demo_project")
    project_dir.mkdir(exist_ok=True)
    
    try:
        # Create subdirectories
        (project_dir / "docs").mkdir(exist_ok=True)
        (project_dir / "src").mkdir(exist_ok=True)
        (project_dir / "data").mkdir(exist_ok=True)
        
        # Create various file types
        print("\n1. Creating Project Structure")
        print("-" * 30)
        
        # README
        (project_dir / "README.md").write_text("""
        # Demo Project
        
        This project demonstrates CZero Engine's capabilities for:
        - Multi-format document processing
        - Intelligent chunking strategies
        - Hierarchical organization
        - Batch processing
        """)
        
        # Documentation
        (project_dir / "docs" / "api_guide.md").write_text("""
        # API Guide
        
        ## Authentication
        All API requests require authentication via API key.
        
        ## Endpoints
        - POST /api/process - Process documents
        - GET /api/search - Semantic search
        - POST /api/chat - Chat with context
        """)
        
        # Source code
        (project_dir / "src" / "processor.py").write_text("""
        class DocumentProcessor:
            def __init__(self, chunk_size=1000):
                self.chunk_size = chunk_size
            
            def process(self, document):
                # Extract text from document
                text = self.extract_text(document)
                # Create chunks
                chunks = self.create_chunks(text)
                return chunks
        """)
        
        # Data file
        (project_dir / "data" / "config.json").write_text("""
        {
            "processing": {
                "chunk_size": 500,
                "overlap": 100,
                "max_tokens": 8192
            },
            "models": {
                "embedding": "all-MiniLM-L6-v2",
                "llm": "gpt-4"
            }
        }
        """)
        
        print(f"✅ Created project structure in {project_dir}")
        
        # Use DocumentProcessingWorkflow for advanced features
        async with DocumentProcessingWorkflow(verbose=True) as workflow:
            
            # 2. Discover and categorize files
            print("\n2. File Discovery")
            print("-" * 30)
            
            all_files = workflow.discover_files(
                directory=str(project_dir),
                patterns=["*.md", "*.py", "*.json", "*.txt"],
                max_size_mb=10
            )
            
            print(f"📁 Discovered {len(all_files)} files:")
            for file in all_files:
                size_kb = file.stat().st_size / 1024
                print(f"   - {file.relative_to(project_dir)} ({size_kb:.1f} KB)")
            
            # 3. Process by file type
            print("\n3. Processing by Type")
            print("-" * 30)
            
            # Process markdown files
            md_files = [f for f in all_files if f.suffix == ".md"]
            if md_files:
                stats = await workflow.process_documents(
                    files=md_files,
                    workspace_name="Documentation"
                )
                print(f"\n📝 Markdown files:")
                print(f"   Files submitted: {stats.total_files}")
                print(f"   Chunking operations: {stats.processed_files}")  # Parent + child chunks
                print(f"   Total chunks created: {stats.total_chunks}")
            
            # Process Python files
            py_files = [f for f in all_files if f.suffix == ".py"]
            if py_files:
                stats = await workflow.process_documents(
                    files=py_files,
                    workspace_name="Source Code"
                )
                print(f"\n🐍 Python files:")
                print(f"   Files submitted: {stats.total_files}")
                print(f"   Chunking operations: {stats.processed_files}")
                print(f"   Total chunks created: {stats.total_chunks}")
            
            # 4. Batch processing example
            print("\n4. Batch Processing")
            print("-" * 30)
            
            # Create more files for batch demo
            batch_dir = project_dir / "batch_docs"
            batch_dir.mkdir(exist_ok=True)
            
            for i in range(10):
                doc = batch_dir / f"doc_{i:02d}.txt"
                doc.write_text(f"""
                Document {i}: Sample Content
                
                This is document number {i} in our batch processing demo.
                It contains sample text to demonstrate parallel processing
                capabilities of the CZero Engine document processor.
                
                Topics covered: AI, ML, NLP, Vector Databases, RAG Systems
                """)
            
            batch_files = list(batch_dir.glob("*.txt"))
            
            stats = await workflow.process_documents(
                files=batch_files,
                workspace_name="Batch Demo",
                batch_size=3  # Process 3 files at a time
            )
            
            print(f"\n⚡ Batch processing results:")
            print(f"   Files submitted: {stats.total_files}")
            print(f"   Chunking operations: {stats.processed_files}")  # Hierarchical chunks
            print(f"   Total chunks created: {stats.total_chunks}")
            print(f"   Time: {stats.processing_time:.2f}s")
            print(f"   Throughput: {stats.total_chunks/stats.processing_time:.1f} chunks/s")
            print(f"\n   Note: With SmallToBig hierarchical chunking, each file generates")
            print(f"   multiple parent and child chunks for optimal retrieval.")
    
    finally:
        # Cleanup
        if project_dir.exists():
            shutil.rmtree(project_dir)
            print(f"\n🧹 Cleaned up {project_dir}")


async def hierarchical_processing():
    """Demonstrate hierarchical document processing."""
    
    print("\n🏗️ Hierarchical Document Processing")
    print("=" * 50)
    
    async with CZeroEngineClient() as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a document with clear hierarchy
            doc_path = Path(temp_dir) / "hierarchical_doc.md"
            doc_path.write_text("""
            # Chapter 1: Introduction to AI
            
            ## Section 1.1: What is AI?
            Artificial Intelligence refers to computer systems that can perform
            tasks typically requiring human intelligence.
            
            ## Section 1.2: History of AI
            AI research began in the 1950s with pioneers like Alan Turing.
            
            # Chapter 2: Machine Learning
            
            ## Section 2.1: Supervised Learning
            Supervised learning uses labeled data to train models.
            
            ## Section 2.2: Unsupervised Learning
            Unsupervised learning finds patterns in unlabeled data.
            """)
            
            # Create workspace and process
            workspace = await client.create_workspace(
                name="Hierarchical Demo",
                path=temp_dir
            )
            
            result = await client.process_files(
                workspace_id=workspace.id,
                files=[str(doc_path)]
            )
            
            print(f"✅ Hierarchical processing complete:")
            print(f"   Total chunks created: {result.chunks_created}")
            print(f"   (SmallToBig creates parent + child chunks for each section)")
            
            # Search at different hierarchy levels
            print("\n🔍 Searching at different levels:")
            
            # Search sections (level 0)
            section_results = await client.semantic_search(
                query="What is machine learning?",
                hierarchy_level="0",  # Sections
                limit=2
            )
            print(f"\n📑 Section-level results: {len(section_results.results)}")
            
            # Search paragraphs (level 1)
            paragraph_results = await client.semantic_search(
                query="supervised learning with labeled data",
                hierarchy_level="1",  # Paragraphs
                limit=2
            )
            print(f"📝 Paragraph-level results: {len(paragraph_results.results)}")
            
            # Search all levels with hierarchy
            all_results = await client.semantic_search(
                query="AI and machine learning",
                include_hierarchy=True,
                limit=3
            )
            print(f"🔗 All levels with context: {len(all_results.results)}")


async def main():
    """Run all document processing examples."""
    try:
        await basic_document_processing()
        await advanced_processing_workflow()
        await hierarchical_processing()
        
        print("\n✅ All document processing examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure CZero Engine is running")
        print("2. Check API server is active")
        print("3. Verify embedding models are loaded")
        print("4. Ensure sufficient disk space for processing")


if __name__ == "__main__":
    asyncio.run(main())