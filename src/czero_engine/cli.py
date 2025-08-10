"""CLI for CZero Engine Python SDK."""

import asyncio
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import json

from .client import CZeroEngineClient
from .workflows import (
    KnowledgeBaseWorkflow,
    RAGWorkflow,
    PersonaWorkflow,
    DocumentProcessingWorkflow
)

app = typer.Typer(help="CZero Engine CLI - Interact with CZero Engine API")
console = Console()


@app.command()
def health():
    """Check CZero Engine API health status."""
    async def check():
        async with CZeroEngineClient() as client:
            try:
                result = await client.health_check()
                console.print(Panel(
                    f"[green]✓[/green] CZero Engine API is healthy\n"
                    f"Status: {result['status']}\n"
                    f"Version: {result.get('version', 'Unknown')}\n"
                    f"Service: {result.get('service', 'czero-api')}",
                    title="Health Check",
                    expand=False
                ))
            except Exception as e:
                console.print(f"[red]✗ API health check failed: {e}[/red]")
                
    asyncio.run(check())


@app.command("create-kb")
def create_knowledge_base(
    directory: str = typer.Argument(..., help="Directory containing documents"),
    name: str = typer.Option("Knowledge Base", "--name", "-n", help="Workspace name"),
    chunk_size: int = typer.Option(1000, "--chunk-size", "-c", help="Chunk size"),
    chunk_overlap: int = typer.Option(200, "--overlap", "-o", help="Chunk overlap"),
    patterns: Optional[str] = typer.Option(None, "--patterns", "-p", help="File patterns (comma-separated)")
):
    """Create a knowledge base from documents."""
    async def create():
        async with KnowledgeBaseWorkflow() as workflow:
            file_patterns = patterns.split(",") if patterns else None
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress.add_task(f"Creating knowledge base from {directory}...", total=None)
                
                try:
                    result = await workflow.create_knowledge_base(
                        name=name,
                        directory_path=directory,
                        file_patterns=file_patterns,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    console.print(f"[green]✓[/green] Knowledge base created successfully")
                    console.print(f"  Workspace ID: {result['workspace_id']}")
                    console.print(f"  Files processed: {result['files_processed']}")
                    console.print(f"  Chunks created: {result['chunks_created']}")
                    
                except Exception as e:
                    console.print(f"[red]✗ Failed to create knowledge base: {e}[/red]")
                    
    asyncio.run(create())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Similarity threshold"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace filter")
):
    """Semantic search across documents."""
    async def run_search():
        async with CZeroEngineClient() as client:
            try:
                results = await client.semantic_search(
                    query=query,
                    limit=limit,
                    similarity_threshold=threshold,
                    workspace_filter=workspace
                )
                
                if results.results:
                    table = Table(title=f"Search Results for: {query}")
                    table.add_column("Score", style="cyan", width=10)
                    table.add_column("Document", style="green")
                    table.add_column("Content", style="white", overflow="fold")
                    
                    for result in results.results:
                        table.add_row(
                            f"{result.similarity:.3f}",
                            result.document_id[:20] + "..." if len(result.document_id) > 20 else result.document_id,
                            result.content[:100] + "..." if len(result.content) > 100 else result.content
                        )
                        
                    console.print(table)
                else:
                    console.print("[yellow]No results found[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]✗ Search failed: {e}[/red]")
                
    asyncio.run(run_search())


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    use_rag: bool = typer.Option(True, "--rag/--no-rag", help="Use RAG for context"),
    chunks: int = typer.Option(5, "--chunks", "-c", help="Number of context chunks"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model ID to use")
):
    """Ask a question using LLM with optional RAG."""
    async def run_ask():
        async with CZeroEngineClient() as client:
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    progress.add_task("Generating response...", total=None)
                    
                    response = await client.chat(
                        message=question,
                        use_rag=use_rag,
                        chunk_limit=chunks if use_rag else None,
                        model_id=model
                    )
                    
                console.print(Panel(
                    response.response,
                    title="Response",
                    expand=False
                ))
                
                if use_rag and response.context_used:
                    console.print("\n[dim]Context sources used:[/dim]")
                    for ctx in response.context_used[:3]:
                        console.print(f"  • [dim]{ctx.chunk_id[:40]}... (similarity: {ctx.similarity:.3f})[/dim]")
                        
            except Exception as e:
                console.print(f"[red]✗ Failed to generate response: {e}[/red]")
                
    asyncio.run(run_ask())


@app.command()
def chat(
    persona: str = typer.Option("gestalt-default", "--persona", "-p", help="Persona ID"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model ID to use")
):
    """Interactive chat with a persona."""
    async def run_chat():
        async with PersonaWorkflow() as workflow:
            # Select persona
            await workflow.select_persona(persona)
            
            console.print("[cyan]Interactive chat started. Type 'exit' to quit.[/cyan]\n")
            
            while True:
                try:
                    message = typer.prompt("You")
                    
                    if message.lower() in ["exit", "quit", "bye"]:
                        console.print("[yellow]Goodbye![/yellow]")
                        break
                        
                    # Get response
                    await workflow.chat(
                        message=message,
                        model_id=model,
                        maintain_history=True
                    )
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Chat interrupted[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    
    asyncio.run(run_chat())


@app.command()
def process(
    directory: str = typer.Argument(..., help="Directory to process"),
    workspace: str = typer.Option("default", "--workspace", "-w", help="Workspace name"),
    batch_size: int = typer.Option(10, "--batch", "-b", help="Batch size"),
    chunk_size: int = typer.Option(1000, "--chunk-size", "-c", help="Chunk size")
):
    """Process documents in a directory."""
    async def run_process():
        async with DocumentProcessingWorkflow() as workflow:
            # Discover files
            files = workflow.discover_files(directory)
            
            if not files:
                console.print("[yellow]No files found to process[/yellow]")
                return
                
            console.print(f"[cyan]Found {len(files)} files to process[/cyan]")
            
            # Process files
            stats = await workflow.process_documents(
                files=files,
                workspace_name=workspace,
                batch_size=batch_size,
                chunk_size=chunk_size
            )
            
            # Show results
            console.print(f"\n[green]Processing complete![/green]")
            console.print(f"  Success rate: {stats.success_rate:.1f}%")
            console.print(f"  Files processed: {stats.processed_files}/{stats.total_files}")
            console.print(f"  Chunks created: {stats.total_chunks}")
            console.print(f"  Time taken: {stats.processing_time:.2f}s")
            
    asyncio.run(run_process())


@app.command()
def personas():
    """List available personas."""
    async def list_personas():
        async with PersonaWorkflow() as workflow:
            await workflow.list_personas()
            
    asyncio.run(list_personas())


@app.command()
def documents():
    """List all documents."""
    async def list_docs():
        async with CZeroEngineClient() as client:
            try:
                result = await client.list_documents()
                
                if result.documents:
                    table = Table(title="Documents")
                    table.add_column("ID", style="cyan", overflow="fold")
                    table.add_column("Title", style="green")
                    table.add_column("Type", style="yellow")
                    table.add_column("Size", style="magenta")
                    table.add_column("Workspace", style="blue")
                    
                    for doc in result.documents[:20]:  # Show first 20
                        size_mb = doc.size / (1024 * 1024)
                        table.add_row(
                            doc.id[:12] + "...",
                            doc.title[:30] + "..." if len(doc.title) > 30 else doc.title,
                            doc.content_type or "unknown",
                            f"{size_mb:.2f} MB",
                            doc.workspace_id[:8] + "..." if doc.workspace_id else ""
                        )
                        
                    console.print(table)
                    
                    if len(result.documents) > 20:
                        console.print(f"\n[dim]Showing 20 of {len(result.documents)} documents[/dim]")
                else:
                    console.print("[yellow]No documents found[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]✗ Failed to list documents: {e}[/red]")
                
    asyncio.run(list_docs())


@app.command()
def embed(
    text: str = typer.Argument(..., help="Text to generate embedding for"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model ID to use"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for embedding")
):
    """Generate embedding for text."""
    async def generate():
        async with CZeroEngineClient() as client:
            try:
                result = await client.generate_embedding(
                    text=text,
                    model_id=model
                )
                
                console.print(f"[green]✓[/green] Embedding generated")
                console.print(f"  Model: {result.model_used}")
                console.print(f"  Dimensions: {len(result.embedding)}")
                
                if output:
                    # Save to file
                    output_data = {
                        "text": text,
                        "model": result.model_used,
                        "embedding": result.embedding
                    }
                    Path(output).write_text(json.dumps(output_data, indent=2))
                    console.print(f"  Saved to: {output}")
                else:
                    # Show first few dimensions
                    console.print(f"  First 10 values: {result.embedding[:10]}")
                    
            except Exception as e:
                console.print(f"[red]✗ Failed to generate embedding: {e}[/red]")
                
    asyncio.run(generate())


@app.command()
def version():
    """Show version information."""
    console.print(Panel(
        "[bold cyan]CZero Engine Python SDK[/bold cyan]\n"
        "Version: 0.1.0\n"
        "Python: 3.11+\n"
        "API Endpoint: http://localhost:1421",
        title="Version Info",
        expand=False
    ))


if __name__ == "__main__":
    app()