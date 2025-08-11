"""Document Processing workflow for CZero Engine - Advanced document handling."""

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import mimetypes
import asyncio
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

from ..client import CZeroEngineClient
from ..models import (
    WorkspaceResponse, 
    ProcessFilesResponse,
    DocumentsResponse,
    EmbeddingResponse
)

console = Console()


@dataclass
class ProcessingStats:
    """Statistics for document processing."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_size_bytes: int = 0
    processing_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate based on actual files (not chunks)."""
        if self.total_files == 0:
            return 0.0
        # For hierarchical chunking, processed_files counts total chunks
        # We need to estimate actual file success rate
        # Assuming average of 4 chunks per file (2 parent + 2 child)
        estimated_files = self.processed_files // 4 if self.processed_files > self.total_files else self.processed_files
        actual_success = min(estimated_files, self.total_files)  # Cap at 100%
        return (actual_success / self.total_files) * 100


class DocumentProcessingWorkflow:
    """
    Advanced document processing workflow for CZero Engine.
    
    This workflow provides:
    - Intelligent file discovery and filtering
    - Batch processing with progress tracking
    - Multiple workspace management
    - Document deduplication
    - Processing statistics and reporting
    """
    
    def __init__(self, client: Optional[CZeroEngineClient] = None, verbose: bool = True):
        """
        Initialize Document Processing workflow.
        
        Args:
            client: Optional CZeroEngineClient instance
            verbose: Enable verbose output
        """
        self.client = client
        self._owns_client = client is None
        self.verbose = verbose
        self.workspaces: Dict[str, WorkspaceResponse] = {}
        self.processing_stats = ProcessingStats()
        
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
            
    def discover_files(
        self,
        directory: str,
        patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_size_mb: Optional[float] = None,
        min_size_kb: Optional[float] = None
    ) -> List[Path]:
        """
        Discover files in a directory with advanced filtering.
        
        Args:
            directory: Directory to search
            patterns: File patterns to include (e.g., ["*.pdf", "*.txt"])
            exclude_patterns: Patterns to exclude
            max_size_mb: Maximum file size in MB
            min_size_kb: Minimum file size in KB
            
        Returns:
            List of discovered file paths
        """
        path = Path(directory)
        if not path.exists():
            console.print(f"[red]Directory not found: {directory}[/red]")
            return []
            
        # Default patterns for common document types
        if patterns is None:
            patterns = [
                "*.txt", "*.md", "*.pdf", "*.docx", "*.doc",
                "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
                "*.java", "*.cpp", "*.c", "*.h", "*.hpp",
                "*.json", "*.yaml", "*.yml", "*.xml",
                "*.rst", "*.tex", "*.html", "*.css"
            ]
            
        exclude = exclude_patterns or [
            "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll",
            "*.exe", "*.bin", "*.dat", "*.db", "*.sqlite",
            "*.jpg", "*.jpeg", "*.png", "*.gif", "*.mp4",
            "*.zip", "*.tar", "*.gz", "*.rar"
        ]
        
        discovered_files = []
        
        for pattern in patterns:
            for file_path in path.rglob(pattern):
                # Skip if matches exclude pattern
                if any(file_path.match(exc) for exc in exclude):
                    continue
                    
                # Skip if not a file
                if not file_path.is_file():
                    continue
                    
                # Check file size constraints
                try:
                    size_bytes = file_path.stat().st_size
                    
                    if max_size_mb and size_bytes > max_size_mb * 1024 * 1024:
                        continue
                        
                    if min_size_kb and size_bytes < min_size_kb * 1024:
                        continue
                        
                    discovered_files.append(file_path)
                    
                except Exception as e:
                    if self.verbose:
                        console.print(f"[yellow]Cannot access {file_path}: {e}[/yellow]")
                        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in discovered_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
                
        if self.verbose:
            console.print(f"[cyan]Discovered {len(unique_files)} files[/cyan]")
            
            # Show file type distribution
            type_counts = {}
            for f in unique_files:
                ext = f.suffix.lower()
                type_counts[ext] = type_counts.get(ext, 0) + 1
                
            if type_counts:
                table = Table(title="File Types")
                table.add_column("Extension", style="cyan")
                table.add_column("Count", style="green")
                
                for ext, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    table.add_row(ext or "(no extension)", str(count))
                    
                console.print(table)
                
        return unique_files
        
    async def process_documents(
        self,
        files: List[Path],
        workspace_name: str,
        workspace_path: Optional[str] = None,
        batch_size: int = 10,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        skip_existing: bool = False
    ) -> ProcessingStats:
        """
        Process a list of documents into a workspace.
        
        Args:
            files: List of file paths to process
            workspace_name: Name for the workspace
            workspace_path: Optional workspace path (uses first file's directory if not specified)
            batch_size: Number of files to process at once
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            skip_existing: Skip files that already exist in the workspace
            
        Returns:
            Processing statistics
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        if not files:
            console.print("[yellow]No files to process[/yellow]")
            return ProcessingStats()
            
        # Determine workspace path
        if not workspace_path:
            workspace_path = str(files[0].parent)
            
        # Create or get workspace
        if workspace_name not in self.workspaces:
            workspace = await self.client.create_workspace(
                name=workspace_name,
                path=workspace_path,
                description=f"Document workspace with {len(files)} files"
            )
            self.workspaces[workspace_name] = workspace
        else:
            workspace = self.workspaces[workspace_name]
            
        console.print(f"[green]✓[/green] Using workspace: {workspace.id}")
        
        # Get existing documents if skip_existing is enabled
        existing_files = set()
        if skip_existing:
            docs_response = await self.client.list_documents()
            for doc in docs_response.documents:
                if doc.workspace_id == workspace.id:
                    existing_files.add(doc.path)
                    
            if existing_files:
                console.print(f"[dim]Skipping {len(existing_files)} existing files[/dim]")
                files = [f for f in files if str(f) not in existing_files]
                
        # Initialize statistics
        stats = ProcessingStats(total_files=len(files))
        
        # Process files in batches
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Processing {len(files)} documents...",
                total=len(files)
            )
            
            for i in range(0, len(files), batch_size):
                batch = files[i:i+batch_size]
                batch_paths = [str(f.absolute()) for f in batch]
                
                # Calculate batch size
                batch_size_bytes = sum(f.stat().st_size for f in batch if f.exists())
                stats.total_size_bytes += batch_size_bytes
                
                try:
                    # Process batch
                    import time
                    start_time = time.time()
                    
                    result = await self.client.process_files(
                        workspace_id=workspace.id,
                        files=batch_paths,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    processing_time = time.time() - start_time
                    stats.processing_time += processing_time
                    
                    # Update statistics
                    # Note: files_processed actually returns total chunk operations for hierarchical processing
                    stats.processed_files += result.files_processed  # This is actually chunk count
                    stats.failed_files += result.files_failed
                    stats.total_chunks += result.chunks_created
                    
                    progress.update(task, advance=len(batch))
                    
                    if self.verbose and result.files_failed > 0:
                        console.print(f"[yellow]⚠ {result.files_failed} files failed in batch[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]Batch processing error: {e}[/red]")
                    stats.failed_files += len(batch)
                    progress.update(task, advance=len(batch))
                    
        # Display final statistics
        self._display_stats(stats)
        
        return stats
        
    async def process_directory_tree(
        self,
        root_directory: str,
        workspace_prefix: str = "workspace",
        organize_by_type: bool = True,
        **process_kwargs
    ) -> Dict[str, ProcessingStats]:
        """
        Process an entire directory tree, organizing into multiple workspaces.
        
        Args:
            root_directory: Root directory to process
            workspace_prefix: Prefix for workspace names
            organize_by_type: Create separate workspaces by file type
            **process_kwargs: Additional arguments for process_documents
            
        Returns:
            Dictionary of workspace names to processing statistics
        """
        # Discover all files
        all_files = self.discover_files(root_directory)
        
        if not all_files:
            console.print("[yellow]No files found to process[/yellow]")
            return {}
            
        workspace_stats = {}
        
        if organize_by_type:
            # Group files by type
            file_groups = {}
            for file_path in all_files:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type:
                    category = mime_type.split('/')[0]
                else:
                    category = file_path.suffix.lower() or "other"
                    
                if category not in file_groups:
                    file_groups[category] = []
                file_groups[category].append(file_path)
                
            # Process each group
            for category, files in file_groups.items():
                workspace_name = f"{workspace_prefix}_{category}"
                
                if self.verbose:
                    console.print(f"\n[bold]Processing {category} files[/bold]")
                    
                stats = await self.process_documents(
                    files=files,
                    workspace_name=workspace_name,
                    workspace_path=root_directory,
                    **process_kwargs
                )
                
                workspace_stats[workspace_name] = stats
                
        else:
            # Process all files into single workspace
            workspace_name = f"{workspace_prefix}_all"
            stats = await self.process_documents(
                files=all_files,
                workspace_name=workspace_name,
                workspace_path=root_directory,
                **process_kwargs
            )
            workspace_stats[workspace_name] = stats
            
        # Display summary
        self._display_summary(workspace_stats)
        
        return workspace_stats
        
    async def generate_embeddings_for_text(
        self,
        texts: List[str],
        model_id: Optional[str] = None
    ) -> List[EmbeddingResponse]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            model_id: Optional embedding model to use
            
        Returns:
            List of embedding responses
        """
        if not self.client:
            raise ValueError("Client not initialized")
            
        embeddings = []
        
        if self.verbose:
            console.print(f"[cyan]Generating embeddings for {len(texts)} texts...[/cyan]")
            
        for text in texts:
            try:
                embedding = await self.client.generate_embedding(
                    text=text,
                    model_id=model_id
                )
                embeddings.append(embedding)
            except Exception as e:
                console.print(f"[red]Failed to generate embedding: {e}[/red]")
                
        if self.verbose:
            console.print(f"[green]✓[/green] Generated {len(embeddings)} embeddings")
            
        return embeddings
        
    def _display_stats(self, stats: ProcessingStats):
        """Display processing statistics."""
        if not self.verbose:
            return
            
        table = Table(title="Processing Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Files", str(stats.total_files))
        table.add_row("Chunk Operations", str(stats.processed_files))  # Hierarchical chunks
        table.add_row("Failed Files", str(stats.failed_files))
        table.add_row("Est. Success Rate", f"{stats.success_rate:.1f}%")
        table.add_row("Total Chunks", str(stats.total_chunks))
        table.add_row("Total Size", f"{stats.total_size_bytes / (1024*1024):.2f} MB")
        table.add_row("Processing Time", f"{stats.processing_time:.2f} seconds")
        
        if stats.processing_time > 0:
            throughput = stats.total_size_bytes / (1024 * 1024) / stats.processing_time
            table.add_row("Throughput", f"{throughput:.2f} MB/s")
            
        console.print(table)
        
    def _display_summary(self, workspace_stats: Dict[str, ProcessingStats]):
        """Display summary of multiple workspace processing."""
        if not self.verbose or not workspace_stats:
            return
            
        console.print("\n[bold]Processing Summary[/bold]")
        
        tree = Tree("Workspaces")
        
        total_files = 0
        total_chunks = 0
        total_time = 0
        
        for workspace_name, stats in workspace_stats.items():
            branch = tree.add(f"{workspace_name}")
            branch.add(f"Files Submitted: {stats.total_files}")
            branch.add(f"Total Chunks: {stats.total_chunks}")
            branch.add(f"Chunk Operations: {stats.processed_files}")
            branch.add(f"Est. Success: {stats.success_rate:.1f}%")
            
            total_files += stats.processed_files
            total_chunks += stats.total_chunks
            total_time += stats.processing_time
            
        console.print(tree)
        
        console.print(f"\n[bold green]Total:[/bold green]")
        console.print(f"  Files processed: {total_files}")
        console.print(f"  Chunks created: {total_chunks}")
        console.print(f"  Total time: {total_time:.2f} seconds")


# Example usage
async def example_document_processing():
    """Example of document processing workflow."""
    
    async with DocumentProcessingWorkflow() as workflow:
        # Discover files with filtering
        files = workflow.discover_files(
            directory="./documents",
            patterns=["*.pdf", "*.txt", "*.md"],
            max_size_mb=10,
            min_size_kb=1
        )
        
        # Process documents into a workspace
        if files:
            stats = await workflow.process_documents(
                files=files[:20],  # Process first 20 files
                workspace_name="Technical Docs",
                chunk_size=1000,
                chunk_overlap=200,
                skip_existing=True
            )
            
        # Process entire directory tree with organization
        workspace_stats = await workflow.process_directory_tree(
            root_directory="./project",
            workspace_prefix="project",
            organize_by_type=True,
            chunk_size=800,
            batch_size=5
        )
        
        # Generate embeddings for custom texts
        embeddings = await workflow.generate_embeddings_for_text([
            "CZero Engine is a powerful document processing system",
            "It provides semantic search and RAG capabilities",
            "Documents are processed into vector embeddings"
        ])


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_document_processing())