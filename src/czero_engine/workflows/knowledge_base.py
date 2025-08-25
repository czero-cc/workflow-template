"""Knowledge Base workflow for CZero Engine - Build and query document knowledge bases."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table

from ..client import CZeroEngineClient
from ..models import WorkspaceResponse, ProcessFilesResponse, SemanticSearchResponse

console = Console()


class KnowledgeBaseWorkflow:
    """
    Workflow for creating and querying knowledge bases in CZero Engine.
    
    This workflow helps you:
    1. Create workspaces for organizing documents
    2. Process documents (PDFs, text files, code, etc.)
    3. Build searchable knowledge bases with vector embeddings
    4. Query your knowledge base with semantic search
    """
    
    def __init__(self, client: Optional[CZeroEngineClient] = None, verbose: bool = True):
        """
        Initialize Knowledge Base workflow.
        
        Args:
            client: Optional CZeroEngineClient instance (creates one if not provided)
            verbose: Enable verbose output
        """
        self.client = client
        self._owns_client = client is None
        self.verbose = verbose
        self.workspace_id: Optional[str] = None
        self.workspace_name: Optional[str] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        if self._owns_client:
            self.client = CZeroEngineClient(verbose=self.verbose)
            await self.client.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._owns_client and self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
            
    async def create_knowledge_base(
        self,
        name: str,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a complete knowledge base from a directory of documents.
        
        Args:
            name: Name for the knowledge base/workspace
            directory_path: Path to directory containing documents
            file_patterns: Optional file patterns to include (e.g., ["*.pdf", "*.txt"])
            description: Optional description for the workspace
            
        Returns:
            Dictionary with workspace details and processing statistics
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        console.print(Panel(
            f"[bold cyan]Creating Knowledge Base: {name}[/bold cyan]",
            expand=False
        ))
        
        # Step 1: Create workspace
        if self.verbose:
            console.print("[cyan]📁 Creating workspace...[/cyan]")
            
        workspace = await self.client.create_workspace(
            name=name,
            path=directory_path,
            description=description or f"Knowledge base for {name}"
        )
        
        self.workspace_id = workspace.id
        self.workspace_name = name
        
        console.print(f"[green]✓[/green] Workspace created: {workspace.id}")
        
        # Step 2: Find files to process
        path = Path(directory_path)
        if not path.exists():
            raise ValueError(f"Directory not found: {directory_path}")
            
        patterns = file_patterns or ["*.txt", "*.md", "*.pdf", "*.docx", "*.py", "*.js", "*.json", "*.yaml", "*.yml"]
        files_to_process = []
        
        for pattern in patterns:
            files_to_process.extend(path.rglob(pattern))
            
        if not files_to_process:
            console.print(f"[yellow]⚠ No files found matching patterns: {patterns}[/yellow]")
            return {
                "workspace": workspace.model_dump(),
                "files_processed": 0,
                "chunks_created": 0
            }
            
        console.print(f"[cyan]📄 Found {len(files_to_process)} files to process[/cyan]")
        
        # Step 3: Process files in batches
        batch_size = 10
        total_processed = 0
        total_chunks = 0
        failed_files = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Processing {len(files_to_process)} files...",
                total=len(files_to_process)
            )
            
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i+batch_size]
                batch_paths = [str(f.absolute()) for f in batch]
                
                try:
                    # Process batch using hierarchical chunking
                    result = await self.client.process_files(
                        workspace_id=workspace.id,
                        files=batch_paths
                    )
                    
                    total_processed += result.files_processed
                    total_chunks += result.chunks_created
                    failed_files += result.files_failed
                    
                    progress.update(task, advance=len(batch))
                    
                    if self.verbose and result.files_failed > 0:
                        console.print(f"[yellow]⚠ {result.files_failed} files failed in batch[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]Error processing batch: {e}[/red]")
                    failed_files += len(batch)
                    progress.update(task, advance=len(batch))
                    
        # Step 4: Display summary
        console.print("\n[bold green]Knowledge Base Created Successfully![/bold green]")
        
        summary = Table(title="Knowledge Base Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")
        
        summary.add_row("Workspace ID", workspace.id)
        summary.add_row("Workspace Name", name)
        summary.add_row("Files Processed", str(total_processed))
        summary.add_row("Files Failed", str(failed_files))
        summary.add_row("Chunks Created", str(total_chunks))
        
        console.print(summary)
        
        return {
            "workspace": workspace.model_dump(),
            "files_processed": total_processed,
            "files_failed": failed_files,
            "chunks_created": total_chunks
        }
        
    async def query(
        self,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.7,
        workspace_id: Optional[str] = None
    ) -> SemanticSearchResponse:
        """
        Query the knowledge base with semantic search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            workspace_id: Optional workspace to search (uses current if not specified)
            
        Returns:
            Search results
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        workspace_to_search = workspace_id or self.workspace_id
        
        if self.verbose:
            console.print(f"\n[cyan]🔍 Searching: {query}[/cyan]")
            if workspace_to_search:
                console.print(f"[dim]Workspace: {workspace_to_search}[/dim]")
                
        results = await self.client.semantic_search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            workspace_filter=workspace_to_search
        )
        
        if self.verbose:
            self.client.print_search_results(results)
            
        return results
        
    async def find_related(
        self,
        chunk_id: str,
        limit: int = 5
    ) -> SemanticSearchResponse:
        """
        Find content related to a specific chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            limit: Maximum number of results
            
        Returns:
            Related content
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        if self.verbose:
            console.print(f"\n[cyan]🔗 Finding content related to chunk: {chunk_id}[/cyan]")
            
        results = await self.client.find_similar_chunks(
            chunk_id=chunk_id,
            limit=limit
        )
        
        if self.verbose:
            self.client.print_search_results(results)
            
        return results
        
    async def get_recommendations(
        self,
        positive_examples: List[str],
        negative_examples: Optional[List[str]] = None,
        limit: int = 10
    ) -> SemanticSearchResponse:
        """
        Get content recommendations based on examples.
        
        Args:
            positive_examples: Chunk IDs of content you like
            negative_examples: Chunk IDs of content to avoid
            limit: Maximum number of recommendations
            
        Returns:
            Recommended content
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        if self.verbose:
            console.print(f"\n[cyan]💡 Getting recommendations...[/cyan]")
            console.print(f"[dim]Based on {len(positive_examples)} positive examples[/dim]")
            if negative_examples:
                console.print(f"[dim]Avoiding {len(negative_examples)} negative examples[/dim]")
                
        results = await self.client.get_recommendations(
            positive_chunk_ids=positive_examples,
            negative_chunk_ids=negative_examples,
            limit=limit
        )
        
        if self.verbose:
            console.print("\n[bold]Recommendations:[/bold]")
            self.client.print_search_results(results)
            
        return results


# Example usage
async def example_knowledge_base():
    """Example of creating and querying a knowledge base."""
    
    async with KnowledgeBaseWorkflow() as workflow:
        # Create knowledge base from documents
        kb_info = await workflow.create_knowledge_base(
            name="Technical Documentation",
            directory_path="./docs",
            file_patterns=["*.md", "*.txt", "*.pdf"],
            description="Technical documentation and guides"
        )
        
        # Query the knowledge base
        search_results = await workflow.query(
            "How does vector search work?",
            limit=5
        )
        
        # Find related content
        if search_results.results:
            first_chunk_id = search_results.results[0].chunk_id
            related = await workflow.find_related(first_chunk_id, limit=3)
            
        # Get recommendations
        if len(search_results.results) >= 2:
            positive_ids = [r.chunk_id for r in search_results.results[:2]]
            recommendations = await workflow.get_recommendations(
                positive_examples=positive_ids,
                limit=5
            )


if __name__ == "__main__":
    asyncio.run(example_knowledge_base())