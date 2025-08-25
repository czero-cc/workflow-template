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
                    f"Status: {result.status}\n"
                    f"Version: {result.version}\n"
                    f"Service: {result.service}",
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
    patterns: Optional[str] = typer.Option(None, "--patterns", "-p", help="File patterns (comma-separated)")
):
    """Create a knowledge base from documents using hierarchical chunking."""
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
                        file_patterns=file_patterns
                    )
                    
                    console.print(f"[green]✓[/green] Knowledge base created successfully")
                    console.print(f"  Workspace ID: {result['workspace']['id']}")
                    console.print(f"  Files processed: {result['files_processed']}")
                    console.print(f"  Chunks created: {result['chunks_created']}")
                    
                except Exception as e:
                    console.print(f"[red]✗ Failed to create knowledge base: {e}[/red]")
                    
    asyncio.run(create())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results"),
    threshold: float = typer.Option(0.3, "--threshold", "-t", help="Similarity threshold"),
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
        async with CZeroEngineClient(timeout=120.0) as client:
            console.print("[cyan]Interactive chat started. Type 'exit' to quit.[/cyan]")
            console.print(f"[dim]Using persona: {persona}[/dim]\n")
            
            conversation_history = []
            
            while True:
                try:
                    message = typer.prompt("You")
                    
                    if message.lower() in ["exit", "quit", "bye"]:
                        console.print("[yellow]Goodbye![/yellow]")
                        break
                        
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        progress.add_task("Generating response...", total=None)
                        
                        # Get response from persona
                        response = await client.chat_with_persona(
                            persona_id=persona,
                            message=message,
                            model_id=model,
                            conversation_history=conversation_history,
                            max_tokens=300  # Shorter for interactive chat
                        )
                        
                    console.print(f"\n[bold cyan]{persona}:[/bold cyan] {response.response}\n")
                    
                    # Update conversation history
                    conversation_history.append({"role": "user", "content": message})
                    conversation_history.append({"role": "assistant", "content": response.response})
                    
                    # Keep history manageable
                    if len(conversation_history) > 20:
                        conversation_history = conversation_history[-20:]
                        
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
    batch_size: int = typer.Option(10, "--batch", "-b", help="Batch size")
):
    """Process documents in a directory using hierarchical chunking."""
    async def run_process():
        async with CZeroEngineClient() as client:
            # Create or find workspace
            workspaces = await client.list_workspaces()
            workspace_obj = None
            
            for ws in workspaces.workspaces:
                if ws.name == workspace:
                    workspace_obj = ws
                    break
            
            if not workspace_obj:
                console.print(f"[cyan]Creating workspace: {workspace}[/cyan]")
                workspace_obj = await client.create_workspace(
                    name=workspace,
                    path=directory
                )
            
            # Discover files
            directory_path = Path(directory)
            if not directory_path.exists():
                console.print(f"[red]Directory not found: {directory}[/red]")
                return
                
            files = list(directory_path.rglob("*"))
            file_paths = [str(f) for f in files if f.is_file() and f.suffix.lower() in ['.txt', '.md', '.pdf', '.docx', '.py', '.js', '.json', '.yaml', '.yml']]
            
            if not file_paths:
                console.print("[yellow]No processable files found[/yellow]")
                return
                
            console.print(f"[cyan]Found {len(file_paths)} files to process[/cyan]")
            
            # Process files in batches
            total_chunks = 0
            total_processed = 0
            
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i+batch_size]
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    progress.add_task(f"Processing batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}...", total=None)
                    
                    try:
                        result = await client.process_files(
                            workspace_id=workspace_obj.id,
                            files=batch
                        )
                        
                        total_chunks += result.chunks_created
                        total_processed += len(batch)
                        
                    except Exception as e:
                        console.print(f"[red]Error processing batch: {e}[/red]")
            
            # Show results
            console.print(f"\n[green]Processing complete![/green]")
            console.print(f"  Files processed: {total_processed}/{len(file_paths)}")
            console.print(f"  Chunks created: {total_chunks}")
            console.print(f"  Workspace: {workspace_obj.name} ({workspace_obj.id})")
            
    asyncio.run(run_process())


@app.command()
def personas():
    """List available personas."""
    async def list_personas():
        async with CZeroEngineClient() as client:
            try:
                result = await client.list_personas()
                
                if result.personas:
                    table = Table(title="Available Personas")
                    table.add_column("ID", style="cyan")
                    table.add_column("Name", style="green")
                    table.add_column("Tagline", style="yellow")
                    table.add_column("Specialty", style="magenta")
                    table.add_column("Default", style="blue")
                    
                    for persona in result.personas:
                        table.add_row(
                            persona.id,
                            persona.name,
                            persona.tagline or "No tagline",
                            persona.specialty or "General",
                            "✓" if persona.is_default else ""
                        )
                        
                    console.print(table)
                else:
                    console.print("[yellow]No personas found[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]✗ Failed to list personas: {e}[/red]")
            
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