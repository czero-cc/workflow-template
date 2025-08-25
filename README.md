# 🚀 CZero Engine Python SDK

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CZero Engine](https://img.shields.io/badge/CZero%20Engine-1.0%2B-purple?style=for-the-badge)](https://github.com/CZero/czero-engine)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6.4%2B-orange?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Documentation](https://img.shields.io/badge/Docs-Available-success?style=for-the-badge)](https://czero.cc/docs)

**Official Python SDK for CZero Engine**  
*Personal AI Interface: full local AI suite with document processing, semantic search, and RAG system with AI personas*

[Installation](#-installation) • [Quick Start](#-quick-start) • [Examples](#-examples) • [API Docs](https://docs.czero.cc) • [Contributing](CONTRIBUTING.md)

</div>

## ✨ Features

- **🔍 Semantic Search**: Vector-based search with hierarchical context support
- **📄 Document Processing**: Extract, chunk, and embed multiple file formats
- **🤖 RAG System**: Context-aware AI responses using your documents
- **🎭 AI Personas**: Gestalt (default assistant) + custom personas
- **📊 Workspace Management**: Organize and process documents efficiently (CZero Engine's main offering)
- **⚡ High Performance**: Batch processing, streaming responses, GPU acceleration
- **🔗 LangGraph Integration**: Build complex AI agents with CZero Engine as backend
- **☁️ Cloud AI Compatible**: Combine with OpenAI, Anthropic, Google AI, and more (langchain compatible)

## 📦 Installation

```bash
# From source (currently the only method)
git clone https://github.com/czero-cc/workflow-template.git
cd workflow-template
uv pip install -e .

# Or with pip
pip install -e .

# Install with optional dependencies
uv pip install -e ".[langgraph]"  # For LangGraph integration
```

**Requirements**: Python 3.11+ | CZero Engine running on port 1421

## 🎯 Quick Start

```python
import asyncio
from czero_engine import CZeroEngineClient

async def main():
    async with CZeroEngineClient() as client:
        # Check health
        health = await client.health_check()
        print(f"✅ API Status: {health.status}")
        
        # Chat with LLM
        response = await client.chat(
            message="Explain RAG systems",
            use_rag=True  # Use document context if available
        )
        print(response.response)

asyncio.run(main())
```

## 📚 Core Workflows

### 1. Knowledge Base Creation

```python
from czero_engine.workflows import KnowledgeBaseWorkflow

async with KnowledgeBaseWorkflow() as kb:
    result = await kb.create_knowledge_base(
        name="Technical Docs",
        directory_path="./documents"
    )
    print(f"Processed {result['files_processed']} files, created {result['chunks_created']} chunks")  # Hierarchical chunking creates multiple chunks per file
```

### 2. RAG-Enhanced Q&A

```python
# Use direct client for RAG-enhanced chat
async with CZeroEngineClient() as client:
    # Ask with document context
    response = await client.chat(
        message="What are the key features?",
        use_rag=True,
        chunk_limit=5,
        similarity_threshold=0.3  # Lower threshold for better recall
    )
    
    # Compare with/without RAG
    response_no_rag = await client.chat(
        message="Explain semantic search",
        use_rag=False
    )
    response_with_rag = await client.chat(
        message="Explain semantic search",
        use_rag=True,
        similarity_threshold=0.3
    )
```

### 3. Hierarchical Search

```python
# Search at different hierarchy levels
results = await client.semantic_search(
    query="machine learning concepts",
    hierarchy_level="0",  # Sections only
    include_hierarchy=True  # Include parent/child context
)
```

### 4. AI Persona Interactions

```python
# Use direct client for persona interactions
async with CZeroEngineClient() as client:
    # Chat with default Gestalt persona
    response = await client.chat_with_persona(
        persona_id="gestalt-default",  # default persona
        message="Analyze the implications of AGI"
    )
    
    # Or use regular chat (defaults to Gestalt if no persona specified)
    response = await client.chat(
        message="What are the key features of CZero Engine?",
        use_rag=True
    )
```

### 5. LangGraph Integration (NEW!)

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.language_models import BaseChatModel
from czero_engine import CZeroEngineClient

# Create a ChatModel wrapper for CZero Engine (simplified from example 05)
class CZeroLLM(BaseChatModel):
    client: Optional[CZeroEngineClient] = None
    use_rag: bool = True
    base_url: str = "http://localhost:1421"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.client:
            self.client = CZeroEngineClient(base_url=self.base_url)
    
    async def _agenerate(self, messages, **kwargs):
        # Convert messages to prompt for CZero Engine
        prompt = messages[-1].content if messages else ""
        
        # Use CZero Engine for generation with RAG
        response = await self.client.chat(
            message=prompt,
            use_rag=self.use_rag,
            max_tokens=1024
        )
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=response.response)
        )])
    
    @property
    def _llm_type(self):
        return "czero-engine"

# Use CZero Engine as LLM backend for LangGraph agents
async with CZeroLLM(use_rag=True) as llm:

# Build complex agent workflows with Command-based routing
workflow = StateGraph(MessagesState)
workflow.add_node("search", search_node)
workflow.add_node("analyze", analyze_node)
graph = workflow.compile()

# Combine with cloud AI providers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Use multiple LLMs in your workflow
cloud_llm = ChatOpenAI(model="gpt-4")  # Or Anthropic, Google, etc.
local_llm = CZeroEngineLLM()  # Your local CZero Engine

# The possibilities are endless! 🚀
```

## 🔧 Direct API Client

For fine-grained control:

```python
async with CZeroEngineClient(
    base_url="http://localhost:1421",
    timeout=60.0
) as client:
    # Create workspace
    workspace = await client.create_workspace(
        name="Research Papers",
        path="./papers"
    )
    
    # Process documents (uses SmallToBig hierarchical chunking automatically)
    result = await client.process_files(
        workspace_id=workspace.id,
        files=["paper1.pdf", "paper2.md"]
    )
    
    # Semantic search
    results = await client.semantic_search(
        query="neural networks",
        limit=10,
        include_hierarchy=True
    )
    
    # Generate embeddings
    embedding = await client.generate_embedding(
        text="Advanced AI concepts"
    )
```

## 📋 CLI Interface

```bash
# Check system health
uv run czero health

# Create knowledge base (uses hierarchical chunking automatically)
uv run czero create-kb ./docs --name "My KB"

# Search documents
uv run czero search "query text" --limit 10 --threshold 0.7

# Ask with RAG
uv run czero ask "Your question" --rag --chunks 5

# Chat with persona
uv run czero chat --persona gestalt-default

# List available personas
uv run czero personas

# List documents
uv run czero documents

# Process documents in directory
uv run czero process ./docs --workspace "My Docs" --batch-size 10

# Generate embeddings
uv run czero embed "some text" --output embedding.json

# Show version
uv run czero version
```

## 🏗️ API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/chat/send` | POST | LLM chat with optional RAG |
| `/api/vector/search/semantic` | POST | Semantic search with hierarchy |
| `/api/vector/search/similarity` | POST | Find similar chunks |
| `/api/embeddings/generate` | POST | Generate text embeddings |
| `/api/workspaces/create` | POST | Create workspace |
| `/api/workspaces/process` | POST | Process documents |
| `/api/personas/list` | GET | List AI personas |
| `/api/personas/chat` | POST | Chat with persona |
| `/api/documents` | GET | List all documents |

### 📊 Similarity Scoring with E5 Models

CZero Engine uses **E5 embedding models** which are known for their high-quality semantic representations. However, E5 models typically produce similarity scores that cluster above 70%, even for moderately related content. 

**Automatic Score Rescaling**: To provide more intuitive similarity scores, CZero Engine automatically rescales E5 similarity scores:
- Raw E5 scores of 70-100% are rescaled to 0-100%
- This provides better differentiation between content relevance
- Scores below 70% similarity are generally filtered out as irrelevant

Example rescaling:
- Raw E5: 70% → Rescaled: 0% (minimum threshold)
- Raw E5: 85% → Rescaled: 50% (moderate similarity)  
- Raw E5: 100% → Rescaled: 100% (exact match)

When using the API:
```python
# The similarity scores returned are already rescaled
results = await client.semantic_search(
    query="your search query",
    similarity_threshold=0.5  # This is post-rescaling (85% raw E5)
)
```

### Request/Response Models

All models are fully typed with Pydantic:

```python
from czero_engine.models import (
    ChatRequest, ChatResponse,
    SemanticSearchRequest, SearchResult,
    WorkspaceCreate, ProcessingResult,
    PersonaChat, PersonaResponse
)
```

## 📖 Examples

### Building a Q&A System

```python
async def build_qa_system(docs_dir: str):
    async with CZeroEngineClient() as client:
        # 1. Create workspace and process documents
        workspace = await client.create_workspace(
            name="QA KB", 
            path=docs_dir
        )
        
        # Process documents in the directory
        files = list(Path(docs_dir).glob("**/*"))
        file_paths = [str(f) for f in files if f.is_file()]
        await client.process_files(
            workspace_id=workspace.id,
            files=file_paths
        )
        
        # 2. Interactive Q&A
        while True:
            q = input("Question: ")
            if q == 'quit': break
            
            response = await client.chat(
                message=q,
                use_rag=True,
                workspace_filter=workspace.id,
                similarity_threshold=0.3
            )
            print(f"Answer: {response.response}\n")
```

### Document Similarity Analysis

```python
async def analyze_similarity(doc1: str, doc2: str):
    async with CZeroEngineClient() as client:
        # Generate embeddings
        emb1 = await client.generate_embedding(doc1)
        emb2 = await client.generate_embedding(doc2)
        
        # Calculate cosine similarity
        import numpy as np
        v1, v2 = np.array(emb1.embedding), np.array(emb2.embedding)
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        print(f"Similarity: {similarity:.3f}")
```

### Batch Processing with Progress

```python
async with CZeroEngineClient() as client:
    # Create workspace
    workspace = await client.create_workspace(
        name="Batch Process",
        path="./docs"
    )
    
    # Discover files
    files = list(Path("./docs").glob("**/*.pdf"))
    file_paths = [str(f) for f in files]
    
    # Process in batches of 10
    import math
    batch_size = 10
    total_chunks = 0
    
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        result = await client.process_files(
            workspace_id=workspace.id,
            files=batch
        )
        total_chunks += result.chunks_created
        print(f"Batch {i//batch_size + 1}: {result.chunks_created} chunks")
    
    print(f"Files submitted: {len(file_paths)}")
    print(f"Total chunks created: {total_chunks}")  # Hierarchical chunks
```

## 🌐 Cloud AI Integration

While CZero Engine is fully self-contained with local LLMs and RAG, you can seamlessly integrate cloud AI providers through LangChain when needed:

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from czero_engine import CZeroEngineClient

# Build a hybrid workflow: Local RAG + Cloud LLMs
workflow = StateGraph(MessagesState)

# Use CZero for local RAG and embeddings
async def search_local(state):
    async with CZeroEngineClient() as client:
        # CZero handles document search locally
        results = await client.semantic_search(state["query"])
        return {"context": results}

# Use cloud LLM for specific tasks if needed
async def generate_cloud(state):
    cloud_llm = ChatOpenAI(model="gpt-4")  # or Anthropic, Google, etc.
    response = await cloud_llm.ainvoke(state["messages"])
    return {"messages": [response]}

# Or use CZero Engine for everything
async def generate_local(state):
    async with CZeroEngineClient() as client:
        response = await client.chat(
            message=state["messages"][-1].content,
            use_rag=True
        )
        return {"messages": [AIMessage(content=response.response)]}

# Mix and match as needed - the choice is yours!
workflow.add_node("search", search_local)
workflow.add_node("generate", generate_local)  # or generate_cloud
```

The beauty of LangChain compatibility means you can start fully local and add cloud services only when needed!

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_integration.py::test_hierarchical_search

# Run with coverage
uv run pytest --cov=czero_engine
```

## 🔐 Configuration

Environment variables (`.env` file):

```env
CZERO_API_URL=http://localhost:1421
CZERO_API_TIMEOUT=60.0
CZERO_VERBOSE=false
```

## 📊 Performance Tips

1. **Batch Processing**: Process multiple files in parallel
2. **Hierarchical Chunking**: System automatically uses SmallToBig chunking for optimal retrieval
3. **Hierarchy**: Use hierarchical search for structured documents
4. **Models**: Ensure LLM and embedding models are pre-loaded
5. **Local LLM Performance**: Expect 10-30 second response times for chat operations
6. **Similarity Thresholds**: Use 0.3 or lower for broader results with E5 models

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🚢 Version Control & Branch Management

### For Contributors:

**Branch Strategy:**
- `main` - Stable releases only
- `develop` - Active development branch
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

**Workflow:**
1. Fork the repository
2. Create feature branch from `develop`
3. Make changes and test
4. Submit PR to `develop` branch
5. After review, we'll merge to `develop`
6. Periodically, `develop` → `main` for releases

**Versioning:**
- We follow [Semantic Versioning](https://semver.org/)
- Format: `MAJOR.MINOR.PATCH`
- Example: `1.2.3`

### Release Process:
```bash
# Tag a release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Create release on GitHub with changelog
```

## 📜 License

MIT License - see [LICENSE](LICENSE) file

## 🤝 Support

- 📧 Email: info@czero.cc
- 💬 Discord: [Join our community](https://discord.gg/yjEUkUTEak)
- 🐛 Issues: [GitHub Issues](https://github.com/czero-cc/workflow-template/issues)
- 📚 Docs: [Documentation](https://docs.czero.cc)

---

<div align="center">
  <h3>🌟 If you find this useful, please give us a star!</h3>
  
  <br>
  
  Made with ❤️ by the CZero Team
  
  <br><br>
  
  <!-- GitHub badges will appear once repo is public -->
  <a href="https://github.com/czero-cc/workflow-template">
    <img src="https://img.shields.io/badge/GitHub-View%20on%20GitHub-181717?style=for-the-badge&logo=github" alt="View on GitHub">
  </a>
</div>