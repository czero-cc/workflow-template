# CZero Engine Python SDK

Official Python SDK and workflow templates for CZero Engine API - a powerful document processing and RAG (Retrieval Augmented Generation) system.

## 🚀 Features

CZero Engine provides:
- **Document Processing**: Extract, chunk, and embed documents (PDFs, text, code, etc.)
- **Vector Search**: Semantic search across your knowledge base
- **RAG System**: Context-aware LLM responses using your documents
- **AI Personas**: Specialized AI assistants (Gestalt, Sage, Pioneer)
- **Workspace Management**: Organize documents into searchable workspaces

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- CZero Engine running locally (port 1421)
- UV package manager (optional but recommended)

### Install with pip
```bash
pip install czero-engine-python
```

### Install from source with UV
```bash
git clone https://github.com/czero/workflow-template.git
cd workflow-template

# Using UV (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 🎯 Quick Start

### 1. Check CZero Engine is Running
```python
import asyncio
from czero_engine import CZeroEngineClient

async def check_health():
    async with CZeroEngineClient() as client:
        health = await client.health_check()
        print(f"Status: {health.status}")
        print(f"Version: {health.version}")

asyncio.run(check_health())
```

### 2. Create a Knowledge Base
```python
from czero_engine.workflows import KnowledgeBaseWorkflow

async def create_kb():
    async with KnowledgeBaseWorkflow() as workflow:
        # Process documents into a searchable knowledge base
        result = await workflow.create_knowledge_base(
            name="My Documentation",
            directory_path="./docs",
            file_patterns=["*.pdf", "*.md", "*.txt"],
            chunk_size=1000,
            chunk_overlap=200
        )
        print(f"Processed {result['files_processed']} files")
        print(f"Created {result['chunks_created']} chunks")

asyncio.run(create_kb())
```

### 3. Use RAG for Q&A
```python
from czero_engine.workflows import RAGWorkflow

async def ask_question():
    async with RAGWorkflow() as workflow:
        response = await workflow.ask(
            question="What is semantic search and how does it work?",
            chunk_limit=5,
            similarity_threshold=0.7
        )
        print(response.response)

asyncio.run(ask_question())
```

## 📚 Workflows

### Knowledge Base Workflow
Build and query document knowledge bases:

```python
from czero_engine.workflows import KnowledgeBaseWorkflow

async with KnowledgeBaseWorkflow() as kb:
    # Create knowledge base from documents
    await kb.create_knowledge_base(
        name="Technical Docs",
        directory_path="./documents"
    )
    
    # Search the knowledge base
    results = await kb.query("How does vector search work?")
    
    # Find similar content
    similar = await kb.find_related(chunk_id="chunk_123")
    
    # Get recommendations
    recs = await kb.get_recommendations(
        positive_examples=["chunk_1", "chunk_2"]
    )
```

### RAG Workflow
Retrieval Augmented Generation for accurate Q&A:

```python
from czero_engine.workflows import RAGWorkflow

async with RAGWorkflow() as rag:
    # Ask with RAG
    response = await rag.ask("Explain document embeddings")
    
    # Search then ask
    result = await rag.search_then_ask(
        search_query="vector embeddings",
        question="How are they generated?"
    )
    
    # Compare with/without RAG
    comparison = await rag.compare_with_without_rag(
        "What is CZero Engine?"
    )
```

### Persona Workflow
Interact with specialized AI personas:

```python
from czero_engine.workflows import PersonaWorkflow

async with PersonaWorkflow() as personas:
    # List available personas
    await personas.list_personas()
    
    # Chat with Gestalt (adaptive assistant)
    response = await personas.chat(
        "Help me understand RAG systems",
        persona_id="gestalt-default"
    )
    
    # Multi-persona discussion
    discussion = await personas.multi_persona_discussion(
        topic="Future of AI",
        persona_ids=["gestalt-default", "sage", "pioneer"],
        rounds=3
    )
```

### Document Processing Workflow
Advanced document processing capabilities:

```python
from czero_engine.workflows import DocumentProcessingWorkflow

async with DocumentProcessingWorkflow() as processor:
    # Discover files with filtering
    files = processor.discover_files(
        directory="./docs",
        patterns=["*.pdf", "*.md"],
        max_size_mb=10
    )
    
    # Process documents
    stats = await processor.process_documents(
        files=files,
        workspace_name="Research",
        chunk_size=800
    )
    
    # Process entire directory tree
    await processor.process_directory_tree(
        root_directory="./project",
        organize_by_type=True
    )
```

## 🔧 API Client

Low-level client for direct API access:

```python
from czero_engine import CZeroEngineClient

async with CZeroEngineClient() as client:
    # Chat with optional RAG
    response = await client.chat(
        message="What is CZero Engine?",
        use_rag=True,
        chunk_limit=5
    )
    
    # Semantic search
    results = await client.semantic_search(
        query="document processing",
        limit=10,
        similarity_threshold=0.7
    )
    
    # Generate embeddings
    embedding = await client.generate_embedding(
        text="Sample text to embed"
    )
    
    # Create workspace
    workspace = await client.create_workspace(
        name="My Workspace",
        path="./workspace"
    )
    
    # Process files
    result = await client.process_files(
        workspace_id=workspace.id,
        files=["doc1.pdf", "doc2.txt"],
        chunk_size=1000
    )
```

## 📋 CLI Usage

The SDK includes a CLI for common operations:

```bash
# Check API health
czero health

# Create knowledge base
czero create-kb ./documents --name "My KB" --chunk-size 1000

# Search
czero search "query text" --limit 10

# Ask with RAG
czero ask "Your question here" --use-rag

# Chat with persona
czero chat --persona gestalt-default

# Process documents
czero process ./docs --workspace "Research"
```

## 🏗️ Architecture

### API Endpoints Used

The SDK uses these CZero Engine API endpoints:

- **`POST /api/chat/send`** - LLM text generation with optional RAG
- **`POST /api/vector/search/semantic`** - Semantic search across documents
- **`POST /api/vector/search/similarity`** - Find similar chunks
- **`POST /api/vector/recommendations`** - Get content recommendations
- **`GET /api/documents`** - List all documents
- **`POST /api/embeddings/generate`** - Generate text embeddings
- **`POST /api/workspaces/create`** - Create document workspace
- **`POST /api/workspaces/process`** - Process files into workspace
- **`GET /api/personas/list`** - List available AI personas
- **`POST /api/personas/chat`** - Chat with specific persona
- **`GET /api/health`** - Health check

### How It Works

1. **Document Processing**: Documents are extracted, chunked, and converted to vector embeddings
2. **Vector Storage**: Embeddings are stored in a vector database for fast similarity search
3. **RAG Pipeline**: 
   - User query → Generate embedding
   - Search for similar chunks → Retrieve context
   - Augment prompt with context → Generate response
4. **Personas**: Specialized system prompts and conversation management

## 🔬 Advanced Examples

### Building a Q&A System
```python
async def build_qa_system(docs_dir: str):
    # Step 1: Create knowledge base
    async with KnowledgeBaseWorkflow() as kb:
        await kb.create_knowledge_base(
            name="QA Knowledge",
            directory_path=docs_dir
        )
        workspace_id = kb.workspace_id
    
    # Step 2: Set up RAG for Q&A
    async with RAGWorkflow() as rag:
        while True:
            question = input("Ask a question (or 'quit'): ")
            if question.lower() == 'quit':
                break
                
            response = await rag.ask(
                question=question,
                workspace_filter=workspace_id
            )
            print(f"\nAnswer: {response.response}\n")
```

### Document Comparison
```python
async def compare_documents(doc1: str, doc2: str):
    async with CZeroEngineClient() as client:
        # Generate embeddings
        emb1 = await client.generate_embedding(doc1)
        emb2 = await client.generate_embedding(doc2)
        
        # Calculate similarity (cosine similarity)
        import numpy as np
        vec1 = np.array(emb1.embedding)
        vec2 = np.array(emb2.embedding)
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        print(f"Document similarity: {similarity:.3f}")
```

### Batch Processing with Progress
```python
from pathlib import Path

async def batch_process_with_progress(root_dir: str):
    async with DocumentProcessingWorkflow(verbose=True) as processor:
        # Discover all documents
        files = processor.discover_files(
            directory=root_dir,
            patterns=["*.pdf", "*.docx", "*.txt"]
        )
        
        print(f"Found {len(files)} files to process")
        
        # Process in batches of 20
        batch_size = 20
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}...")
            
            stats = await processor.process_documents(
                files=batch,
                workspace_name=f"Batch_{i//batch_size + 1}",
                chunk_size=1000
            )
            
            print(f"Success rate: {stats.success_rate:.1f}%")
```

## 🔐 Environment Configuration

Create a `.env` file:

```env
# CZero Engine API Configuration
CZERO_API_URL=http://localhost:1421

# Optional settings
CZERO_API_TIMEOUT=30.0
CZERO_VERBOSE=true
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

MIT License - see LICENSE file for details

## 📧 Support

- Email: info@czero.cc
- Documentation: https://docs.czero.cc
- Issues: https://github.com/czero/workflow-template/issues

## 🔗 Related

- [CZero Engine](https://github.com/czero/czero-engine) - Main engine repository
- [CZero Overlay](https://github.com/czero/czero-overlay) - Desktop application
- [API Documentation](https://api.czero.cc/docs) - Full API reference

---

Built with ❤️ by the CZero Team