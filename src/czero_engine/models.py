"""Data models for CZero Engine API - matching actual Rust implementation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# Chat/LLM Models
class RagConfig(BaseModel):
    """RAG configuration for chat requests."""
    similarity_threshold: float = 0.7
    chunk_limit: int = 5


class ChatRequest(BaseModel):
    """Request model for /api/chat/send endpoint."""
    message: str
    model_id: Optional[str] = None
    use_rag: bool = True
    rag_config: Optional[RagConfig] = None
    use_web_search: bool = False
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    workspace_filter: Optional[str] = None


class ContextChunk(BaseModel):
    """Context chunk used in chat responses."""
    chunk_id: str
    content: str
    similarity: float


class ChatResponse(BaseModel):
    """Response model for /api/chat/send endpoint."""
    response: str
    model_used: str
    context_used: Optional[List[ContextChunk]] = None
    cacheable: bool = False


# Vector Search Models
class SemanticSearchRequest(BaseModel):
    """Request model for /api/vector/search/semantic endpoint."""
    query: str
    limit: int = 10
    similarity_threshold: float = 0.7
    include_content: bool = True
    workspace_filter: Optional[str] = None
    hierarchy_level: Optional[str] = None  # "0" for sections, "1" for paragraphs
    include_hierarchy: bool = False  # Include parent chunks and hierarchy path


class SearchResult(BaseModel):
    """Individual search result."""
    chunk_id: str
    document_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Optional hierarchical context
    parent_chunk: Optional['SearchResult'] = None
    hierarchy_path: Optional[List['SearchResult']] = None
    document_content: Optional[str] = None


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search endpoints."""
    results: List[SearchResult]


class SimilaritySearchRequest(BaseModel):
    """Request model for /api/vector/search/similarity endpoint."""
    chunk_id: str
    limit: int = 5
    similarity_threshold: float = 0.5


class RecommendationsRequest(BaseModel):
    """Request model for /api/vector/recommendations endpoint."""
    positive_chunk_ids: List[str]
    negative_chunk_ids: Optional[List[str]] = Field(default_factory=list)
    limit: int = 10


# Document Models
class DocumentMetadata(BaseModel):
    """Document metadata model."""
    id: str
    title: str
    path: str
    content_type: str
    size: int
    created_at: str
    updated_at: str
    workspace_id: Optional[str] = None


class DocumentsResponse(BaseModel):
    """Response model for /api/documents endpoint."""
    documents: List[DocumentMetadata]


# Embedding Models
class EmbeddingRequest(BaseModel):
    """Request model for /api/embeddings/generate endpoint."""
    text: str
    model_id: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Response model for /api/embeddings/generate endpoint."""
    embedding: List[float]
    model_used: str


# Workspace Models
class WorkspaceCreateRequest(BaseModel):
    """Request model for /api/workspaces/create endpoint."""
    name: str
    path: str
    description: Optional[str] = None


class WorkspaceResponse(BaseModel):
    """Response model for workspace creation."""
    id: str
    name: str
    path: str
    description: Optional[str] = None
    created_at: str


class WorkspaceInfo(BaseModel):
    """Information about a workspace."""
    id: str
    name: str
    path: str
    description: Optional[str] = None
    status: str
    created_at: str
    updated_at: str


class WorkspaceListResponse(BaseModel):
    """Response model for /api/workspaces endpoint."""
    workspaces: List[WorkspaceInfo]


class ProcessingConfig(BaseModel):
    """Configuration for file processing."""
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    embedding_model: Optional[str] = None


class ProcessFilesRequest(BaseModel):
    """Request model for /api/workspaces/process endpoint."""
    workspace_id: str
    files: List[str]
    config: Optional[ProcessingConfig] = None


class ProcessFilesResponse(BaseModel):
    """Response model for file processing."""
    status: str
    workspace_id: str
    files_processed: int
    files_failed: int
    chunks_created: int
    processing_time: float
    message: str


# Persona Models
class PersonaInfo(BaseModel):
    """Information about an AI persona."""
    id: str
    name: str
    tagline: Optional[str] = None
    specialty: str
    avatar: str
    is_default: bool = False
    is_custom: bool = False


class PersonaListResponse(BaseModel):
    """Response model for /api/personas/list endpoint."""
    personas: List[PersonaInfo]


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: str  # "user" or "assistant"
    content: str


class PersonaChatRequest(BaseModel):
    """Request model for /api/personas/chat endpoint."""
    persona_id: str
    message: str
    model_id: Optional[str] = None
    system_prompt_template: Optional[str] = None
    conversation_history: Optional[List[ConversationMessage]] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7


class PersonaChatResponse(BaseModel):
    """Response model for persona chat."""
    response: str
    persona_id: str
    model_used: str
    timestamp: str


# Health Check Model
class HealthResponse(BaseModel):
    """Response model for /api/health endpoint."""
    status: str
    service: str
    version: str
    timestamp: str