"""Official Python client for CZero Engine API."""

import asyncio
from typing import Any, Dict, List, Optional, Union
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

from .models import (
    ChatRequest, ChatResponse,
    SemanticSearchRequest, SemanticSearchResponse,
    SimilaritySearchRequest, RecommendationsRequest,
    DocumentsResponse, DocumentMetadata,
    EmbeddingRequest, EmbeddingResponse,
    WorkspaceCreateRequest, WorkspaceResponse, WorkspaceListResponse, WorkspaceInfo,
    ProcessFilesRequest, ProcessFilesResponse, ProcessingConfig,
    PersonaListResponse, PersonaChatRequest, PersonaChatResponse,
    HealthResponse,
)

console = Console()


class CZeroEngineClient:
    """
    Official Python client for CZero Engine API.
    
    CZero Engine provides:
    - LLM text generation with RAG (Retrieval Augmented Generation)
    - Semantic vector search across documents
    - Document processing and workspace management
    - Embedding generation for text
    - AI personas for specialized interactions
    
    All methods are async and return typed responses.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:1421",
        timeout: float = 60.0,
        verbose: bool = False
    ):
        """
        Initialize CZero Engine client.
        
        Args:
            base_url: Base URL for CZero Engine API (default: http://localhost:1421)
            timeout: Request timeout in seconds (default: 60s for LLM operations)
            verbose: Enable verbose logging
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.client = httpx.AsyncClient(timeout=timeout)
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        
    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)  # Use plain print instead of Rich console
            
    # ==================== Health Check ====================
    
    async def health_check(self) -> HealthResponse:
        """
        Check if CZero Engine API is healthy.
        
        Returns:
            HealthResponse with service status
        """
        self._log("Checking API health...")
        response = await self.client.get(f"{self.base_url}/api/health")
        response.raise_for_status()
        return HealthResponse(**response.json())
        
    # ==================== Chat/LLM Endpoints ====================
        
    async def chat(
        self,
        message: str,
        use_rag: bool = True,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        similarity_threshold: float = 0.7,
        chunk_limit: int = 5,
        use_web_search: bool = False,
        workspace_filter: Optional[str] = None
    ) -> ChatResponse:
        """
        Send a chat message to CZero Engine LLM with optional RAG.
        
        This endpoint combines LLM generation with semantic search to provide
        context-aware responses based on your document knowledge base.
        
        Args:
            message: The user message/prompt
            use_rag: Whether to use RAG for context retrieval
            model_id: Optional specific model ID to use
            system_prompt: Optional system prompt to guide the response
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0-1.0)
            similarity_threshold: Minimum similarity for RAG chunks
            chunk_limit: Maximum number of context chunks to retrieve
            use_web_search: Whether to enable web search (if available)
            workspace_filter: Optional workspace ID to filter RAG context
            
        Returns:
            ChatResponse with generated text and optional context chunks
        """
        request = ChatRequest(
            message=message,
            model_id=model_id,
            use_rag=use_rag,
            rag_config={
                "similarity_threshold": similarity_threshold,
                "chunk_limit": chunk_limit
            } if use_rag else None,
            use_web_search=use_web_search,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            workspace_filter=workspace_filter
        )
        
        self._log(f"Sending chat request (RAG: {use_rag})...")
        response = await self.client.post(
            f"{self.base_url}/api/chat/send",
            json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return ChatResponse(**response.json())
        
    # ==================== Vector Search Endpoints ====================
        
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        include_content: bool = True,
        workspace_filter: Optional[str] = None,
        hierarchy_level: Optional[str] = None,
        include_hierarchy: bool = False
    ) -> SemanticSearchResponse:
        """
        Perform semantic search across your document knowledge base.
        
        Uses vector embeddings to find semantically similar content,
        not just keyword matches.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            include_content: Whether to include full content in results
            workspace_filter: Optional workspace ID to limit search
            hierarchy_level: Optional hierarchy level ("0" for sections, "1" for paragraphs)
            include_hierarchy: Whether to include parent chunks and hierarchy path
            
        Returns:
            SemanticSearchResponse with matching chunks
        """
        request = SemanticSearchRequest(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            include_content=include_content,
            workspace_filter=workspace_filter,
            hierarchy_level=hierarchy_level,
            include_hierarchy=include_hierarchy
        )
        
        self._log(f"Searching for: {query[:50]}...")
        response = await self.client.post(
            f"{self.base_url}/api/vector/search/semantic",
            json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return SemanticSearchResponse(**response.json())
        
    async def find_similar_chunks(
        self,
        chunk_id: str,
        limit: int = 5,
        similarity_threshold: float = 0.5
    ) -> SemanticSearchResponse:
        """
        Find chunks similar to a given chunk ID.
        
        Useful for finding related content or duplicates.
        
        Args:
            chunk_id: ID of the reference chunk
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            SemanticSearchResponse with similar chunks
        """
        request = SimilaritySearchRequest(
            chunk_id=chunk_id,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        self._log(f"Finding similar to chunk: {chunk_id}")
        response = await self.client.post(
            f"{self.base_url}/api/vector/search/similarity",
            json=request.model_dump()
        )
        response.raise_for_status()
        return SemanticSearchResponse(**response.json())
        
    async def get_recommendations(
        self,
        positive_chunk_ids: List[str],
        negative_chunk_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> SemanticSearchResponse:
        """
        Get content recommendations based on positive/negative examples.
        
        Uses vector math to find content similar to positive examples
        and dissimilar to negative examples.
        
        Args:
            positive_chunk_ids: Chunk IDs to find similar content to
            negative_chunk_ids: Chunk IDs to avoid similarity to
            limit: Maximum number of recommendations
            
        Returns:
            SemanticSearchResponse with recommended chunks
        """
        request = RecommendationsRequest(
            positive_chunk_ids=positive_chunk_ids,
            negative_chunk_ids=negative_chunk_ids or [],
            limit=limit
        )
        
        self._log(f"Getting recommendations based on {len(positive_chunk_ids)} positive examples")
        response = await self.client.post(
            f"{self.base_url}/api/vector/recommendations",
            json=request.model_dump()
        )
        response.raise_for_status()
        return SemanticSearchResponse(**response.json())
        
    # ==================== Document Management ====================
        
    async def list_documents(self) -> DocumentsResponse:
        """
        List all documents in the CZero Engine system.
        
        Returns:
            DocumentsResponse with document metadata
        """
        self._log("Fetching document list...")
        response = await self.client.get(f"{self.base_url}/api/documents")
        response.raise_for_status()
        return DocumentsResponse(**response.json())
        
    # ==================== Embedding Generation ====================
        
    async def generate_embedding(
        self,
        text: str,
        model_id: Optional[str] = None
    ) -> EmbeddingResponse:
        """
        Generate vector embeddings for text.
        
        These embeddings can be used for similarity comparisons
        or custom vector operations.
        
        Args:
            text: Text to generate embedding for
            model_id: Optional specific embedding model to use
            
        Returns:
            EmbeddingResponse with embedding vector
        """
        request = EmbeddingRequest(
            text=text,
            model_id=model_id
        )
        
        self._log(f"Generating embedding for text ({len(text)} chars)...")
        response = await self.client.post(
            f"{self.base_url}/api/embeddings/generate",
            json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return EmbeddingResponse(**response.json())
        
    # ==================== Workspace Management ====================
        
    async def create_workspace(
        self,
        name: str,
        path: str,
        description: Optional[str] = None
    ) -> WorkspaceResponse:
        """
        Create a new workspace for document organization.
        
        Workspaces allow you to organize documents and process them
        as separate knowledge bases.
        
        Args:
            name: Workspace name
            path: Filesystem path for the workspace
            description: Optional description
            
        Returns:
            WorkspaceResponse with workspace ID and details
        """
        request = WorkspaceCreateRequest(
            name=name,
            path=path,
            description=description
        )
        
        self._log(f"Creating workspace: {name}")
        response = await self.client.post(
            f"{self.base_url}/api/workspaces/create",
            json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return WorkspaceResponse(**response.json())
        
    async def list_workspaces(self) -> WorkspaceListResponse:
        """
        List all available workspaces.
        
        Returns a list of workspaces with their IDs, names, paths, and status.
        
        Returns:
            WorkspaceListResponse containing list of WorkspaceInfo objects
        """
        self._log("Listing workspaces...")
        response = await self.client.get(f"{self.base_url}/api/workspaces")
        response.raise_for_status()
        return WorkspaceListResponse(**response.json())
        
    async def process_files(
        self,
        workspace_id: str,
        files: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: Optional[str] = None
    ) -> ProcessFilesResponse:
        """
        Process files and add them to a workspace.
        
        This will:
        1. Extract text from documents
        2. Split into chunks
        3. Generate embeddings
        4. Store in vector database
        
        Args:
            workspace_id: ID of the workspace to add files to
            files: List of file paths to process
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
            embedding_model: Optional specific embedding model
            
        Returns:
            ProcessFilesResponse with processing statistics
        """
        config = ProcessingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model
        )
        
        request = ProcessFilesRequest(
            workspace_id=workspace_id,
            files=files,
            config=config
        )
        
        self._log(f"Processing {len(files)} files for workspace {workspace_id}")
        response = await self.client.post(
            f"{self.base_url}/api/workspaces/process",
            json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return ProcessFilesResponse(**response.json())
        
    # ==================== Persona Endpoints ====================
        
    async def list_personas(self) -> PersonaListResponse:
        """
        Get list of available AI personas.
        
        Personas provide specialized interaction styles and expertise.
        
        Returns:
            PersonaListResponse with available personas
        """
        self._log("Fetching persona list...")
        response = await self.client.get(f"{self.base_url}/api/personas/list")
        response.raise_for_status()
        return PersonaListResponse(**response.json())
        
    async def persona_chat(
        self,
        persona_id: str,
        message: str,
        model_id: Optional[str] = None,
        system_prompt_template: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> PersonaChatResponse:
        """
        Chat with a specific AI persona.
        
        Each persona has its own personality, expertise, and interaction style.
        
        Args:
            persona_id: ID of the persona to chat with
            message: User message
            model_id: Optional specific model to use
            system_prompt_template: Optional custom system prompt
            conversation_history: Optional conversation history for context
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            PersonaChatResponse with persona's response
        """
        request = PersonaChatRequest(
            persona_id=persona_id,
            message=message,
            model_id=model_id,
            system_prompt_template=system_prompt_template,
            conversation_history=conversation_history,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        self._log(f"Chatting with persona: {persona_id}")
        response = await self.client.post(
            f"{self.base_url}/api/personas/chat",
            json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return PersonaChatResponse(**response.json())
        
    # ==================== Utility Methods ====================
        
    def print_search_results(self, response: SemanticSearchResponse):
        """
        Pretty print search results to console.
        
        Args:
            response: Search response to display
        """
        if not response.results:
            console.print("[yellow]No results found[/yellow]")
            return
            
        table = Table(title="Search Results")
        table.add_column("Score", style="cyan", width=10)
        table.add_column("Doc ID", style="magenta", width=20)
        table.add_column("Content", style="green", width=60)
        
        for result in response.results[:10]:
            content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
            table.add_row(
                f"{result.similarity:.3f}",
                result.document_id[:20] + "..." if len(result.document_id) > 20 else result.document_id,
                content_preview
            )
            
        console.print(table)
        
    def print_documents(self, response: DocumentsResponse):
        """
        Pretty print document list to console.
        
        Args:
            response: Documents response to display
        """
        if not response.documents:
            console.print("[yellow]No documents found[/yellow]")
            return
            
        table = Table(title=f"Documents ({len(response.documents)} total)")
        table.add_column("Title", style="cyan", width=30)
        table.add_column("Type", style="yellow", width=15)
        table.add_column("Size", style="magenta", width=10)
        table.add_column("Workspace", style="green", width=20)
        
        for doc in response.documents[:20]:
            size_kb = doc.size / 1024
            table.add_row(
                doc.title[:30],
                doc.content_type,
                f"{size_kb:.1f} KB",
                doc.workspace_id or "None"
            )
            
        console.print(table)
        
        if len(response.documents) > 20:
            console.print(f"[dim]... and {len(response.documents) - 20} more[/dim]")


# Convenience function for quick usage
async def quick_chat(message: str, use_rag: bool = True) -> str:
    """
    Quick helper function to send a chat message.
    
    Args:
        message: Message to send
        use_rag: Whether to use RAG
        
    Returns:
        Response text
    """
    async with CZeroEngineClient() as client:
        response = await client.chat(message, use_rag=use_rag)
        return response.response