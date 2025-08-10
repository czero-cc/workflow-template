"""CZero Engine Python Client - Official Python SDK for CZero Engine API."""

__version__ = "0.1.0"

from .client import CZeroEngineClient
from .models import (
    ChatRequest,
    ChatResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    WorkspaceCreateRequest,
    WorkspaceResponse,
    ProcessFilesRequest,
    ProcessFilesResponse,
    PersonaChatRequest,
    PersonaChatResponse,
)
from .workflows import (
    KnowledgeBaseWorkflow,
    RAGWorkflow,
    PersonaWorkflow,
    DocumentProcessingWorkflow,
)

__all__ = [
    "CZeroEngineClient",
    "ChatRequest",
    "ChatResponse",
    "SemanticSearchRequest",
    "SemanticSearchResponse",
    "WorkspaceCreateRequest",
    "WorkspaceResponse",
    "ProcessFilesRequest",
    "ProcessFilesResponse",
    "PersonaChatRequest",
    "PersonaChatResponse",
    "KnowledgeBaseWorkflow",
    "RAGWorkflow",
    "PersonaWorkflow",
    "DocumentProcessingWorkflow",
]