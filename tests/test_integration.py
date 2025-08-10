"""Integration tests for CZero Engine Python SDK."""

import pytest
import asyncio
from pathlib import Path
from czero_engine import CZeroEngineClient
from czero_engine.workflows import (
    KnowledgeBaseWorkflow,
    RAGWorkflow,
    PersonaWorkflow,
    DocumentProcessingWorkflow
)


@pytest.mark.asyncio
async def test_client_health_check():
    """Test API health check."""
    async with CZeroEngineClient() as client:
        health = await client.health_check()
        assert health["status"] == "healthy"
        assert "version" in health


@pytest.mark.asyncio
async def test_workspace_creation():
    """Test workspace creation."""
    async with CZeroEngineClient() as client:
        workspace = await client.create_workspace(
            name="Test Workspace",
            path="./test_workspace"
        )
        assert workspace.id
        assert workspace.name == "Test Workspace"


@pytest.mark.asyncio
async def test_chat_without_rag():
    """Test chat without RAG."""
    async with CZeroEngineClient() as client:
        response = await client.chat(
            message="What is 2+2?",
            use_rag=False
        )
        assert response.response
        assert response.model_used


@pytest.mark.asyncio
async def test_embedding_generation():
    """Test embedding generation."""
    async with CZeroEngineClient() as client:
        embedding = await client.generate_embedding(
            text="Test text for embedding"
        )
        assert embedding.embedding
        assert len(embedding.embedding) > 0
        assert embedding.model_used


@pytest.mark.asyncio
async def test_persona_list():
    """Test listing personas."""
    async with CZeroEngineClient() as client:
        personas = await client.list_personas()
        assert personas.personas
        assert len(personas.personas) > 0
        
        # Check for expected personas
        persona_ids = [p.id for p in personas.personas]
        assert "gestalt-default" in persona_ids


@pytest.mark.asyncio
async def test_knowledge_base_workflow():
    """Test knowledge base workflow."""
    # Create test documents
    test_dir = Path("./test_kb_docs")
    test_dir.mkdir(exist_ok=True)
    
    (test_dir / "doc1.txt").write_text("This is a test document about AI.")
    (test_dir / "doc2.txt").write_text("Machine learning is a subset of AI.")
    
    try:
        async with KnowledgeBaseWorkflow() as workflow:
            result = await workflow.create_knowledge_base(
                name="Test KB",
                directory_path=str(test_dir),
                chunk_size=100
            )
            
            assert result["workspace_id"]
            assert result["files_processed"] >= 2
            assert result["chunks_created"] > 0
            
            # Test query
            results = await workflow.query("What is AI?")
            assert results
            
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)


@pytest.mark.asyncio
async def test_rag_workflow():
    """Test RAG workflow."""
    async with RAGWorkflow() as workflow:
        # Ask without RAG first
        response = await workflow.ask(
            question="What is CZero Engine?",
            chunk_limit=0  # No RAG
        )
        assert response.response
        assert not response.context_used
        
        # Note: With RAG would require documents to be indexed first


@pytest.mark.asyncio
async def test_persona_workflow():
    """Test persona workflow."""
    async with PersonaWorkflow(verbose=False) as workflow:
        # List personas
        personas = await workflow.list_personas()
        assert personas.personas
        
        # Select and chat with a persona
        context = await workflow.select_persona("gestalt-default")
        assert context.persona_id == "gestalt-default"
        
        response = await workflow.chat(
            message="Hello, how are you?",
            maintain_history=True
        )
        assert response.response
        assert response.persona_id == "gestalt-default"
        
        # Check conversation history
        summary = workflow.get_conversation_summary()
        assert summary["turn_count"] == 1
        assert summary["message_count"] == 2  # User + assistant


@pytest.mark.asyncio
async def test_document_processing_workflow():
    """Test document processing workflow."""
    # Create test files
    test_dir = Path("./test_processing")
    test_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        (test_dir / f"file_{i}.txt").write_text(f"Test content {i}" * 50)
    
    try:
        async with DocumentProcessingWorkflow(verbose=False) as workflow:
            # Discover files
            files = workflow.discover_files(str(test_dir))
            assert len(files) == 3
            
            # Process documents
            stats = await workflow.process_documents(
                files=files,
                workspace_name="Test Processing",
                chunk_size=100
            )
            
            assert stats.total_files == 3
            assert stats.processed_files > 0
            assert stats.total_chunks > 0
            
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)


@pytest.mark.asyncio
async def test_semantic_search():
    """Test semantic search functionality."""
    async with CZeroEngineClient() as client:
        # First create and process some content
        workspace = await client.create_workspace(
            name="Search Test",
            path="./search_test"
        )
        
        # Note: Would need actual documents processed first
        # This tests the API call structure
        try:
            results = await client.semantic_search(
                query="test query",
                limit=5
            )
            # Results might be empty if no documents are indexed
            assert hasattr(results, 'results')
        except Exception as e:
            # Expected if no documents are indexed
            pass


@pytest.mark.asyncio 
async def test_error_handling():
    """Test error handling."""
    async with CZeroEngineClient() as client:
        # Test with invalid workspace
        with pytest.raises(Exception):
            await client.process_files(
                workspace_id="invalid-workspace-id",
                files=["nonexistent.txt"]
            )


def test_sync_operations():
    """Test that sync operations raise appropriate errors."""
    client = CZeroEngineClient()
    
    # Should not be able to use client without async context
    with pytest.raises(RuntimeError):
        asyncio.run(client.health_check())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])