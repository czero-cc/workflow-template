"""Comprehensive API endpoint testing script for CZero Engine.

This script tests all API endpoints to ensure they work correctly.
Run this after starting the CZero Engine app with API server enabled.
"""

import asyncio
import json
import os
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import uuid

console = Console()

class EndpointTester:
    def __init__(self, base_url: str = "http://localhost:1421"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.test_results = []
        self.workspace_id = None
        self.document_id = None
        self.persona_id = None
        self.chunk_id = None
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    def record_result(self, endpoint: str, method: str, success: bool, message: str, details: Any = None):
        """Record test result for reporting."""
        self.test_results.append({
            "endpoint": endpoint,
            "method": method,
            "success": success,
            "message": message,
            "details": details
        })
        
    async def test_endpoint(self, name: str, method: str, path: str, json_data: Optional[Dict] = None) -> Dict:
        """Test a single endpoint and record results."""
        console.print(f"\n[cyan]Testing:[/cyan] {method} {path}")
        
        try:
            if method == "GET":
                response = await self.client.get(f"{self.base_url}{path}")
            elif method == "POST":
                response = await self.client.post(f"{self.base_url}{path}", json=json_data)
            elif method == "DELETE":
                response = await self.client.delete(f"{self.base_url}{path}")
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            response.raise_for_status()
            data = response.json()
            
            self.record_result(path, method, True, f"✅ {name} successful", data)
            console.print(f"  [green]✅ Success[/green] - Status: {response.status_code}")
            
            # Print sample response
            if isinstance(data, dict):
                for key in list(data.keys())[:3]:  # Show first 3 fields
                    value = str(data[key])[:50] + "..." if len(str(data[key])) > 50 else str(data[key])
                    console.print(f"    {key}: {value}")
                    
            return data
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg += f" - {error_data.get('error', 'Unknown error')}"
            except:
                pass
                
            self.record_result(path, method, False, f"❌ {name} failed: {error_msg}")
            console.print(f"  [red]❌ Failed[/red] - {error_msg}")
            return {}
            
        except Exception as e:
            self.record_result(path, method, False, f"❌ {name} error: {str(e)}")
            console.print(f"  [red]❌ Error[/red] - {str(e)}")
            return {}
            
    async def run_all_tests(self):
        """Run all endpoint tests in sequence."""
        console.print(Panel.fit("🚀 CZero Engine API Endpoint Testing", style="bold cyan"))
        
        # 1. Health Check
        console.print("\n[bold yellow]═══ Health & Status ═══[/bold yellow]")
        await self.test_endpoint("Health Check", "GET", "/api/health")
        
        # 2. Workspace Management (create workspace first for other tests)
        console.print("\n[bold yellow]═══ Workspace Management ═══[/bold yellow]")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create workspace
            workspace_data = await self.test_endpoint(
                "Create Workspace", "POST", "/api/workspaces/create",
                {
                    "name": f"Test Workspace {uuid.uuid4().hex[:8]}",
                    "path": temp_dir,
                    "description": "API test workspace"
                }
            )
            if workspace_data:
                self.workspace_id = workspace_data.get("id")
                
            # List workspaces
            await self.test_endpoint("List Workspaces", "GET", "/api/workspaces")
            
            # Create test files for processing
            if self.workspace_id:
                test_file = Path(temp_dir) / "test_document.txt"
                test_file.write_text("This is a test document for CZero Engine API testing. " * 50)
                
                # Process files
                await self.test_endpoint(
                    "Process Files", "POST", "/api/workspaces/process",
                    {
                        "workspace_id": self.workspace_id,
                        "files": [str(test_file)],
                        "config": {
                            "chunk_size": 500,
                            "chunk_overlap": 100
                        }
                    }
                )
                
                # Add dialogue
                await self.test_endpoint(
                    "Add Dialogue", "POST", "/api/workspaces/add-dialogue",
                    {
                        "workspace_id": self.workspace_id,
                        "dialogue_text": "User: Hello AI!\nAI: Hello! How can I help you today?",
                        "character_name": "Test Character"
                    }
                )
        
        # 3. Document Management
        console.print("\n[bold yellow]═══ Document Management ═══[/bold yellow]")
        docs_data = await self.test_endpoint("List Documents", "GET", "/api/documents")
        
        if docs_data and docs_data.get("documents"):
            self.document_id = docs_data["documents"][0]["id"]
            
            # Get single document
            if self.document_id:
                await self.test_endpoint(
                    "Get Document", "GET", f"/api/documents/{self.document_id}"
                )
                
                # Get document full text
                await self.test_endpoint(
                    "Get Document Full Text", "GET", f"/api/documents/{self.document_id}/full-text"
                )
        
        # 4. Embedding Generation
        console.print("\n[bold yellow]═══ Embedding Generation ═══[/bold yellow]")
        embedding_data = await self.test_endpoint(
            "Generate Embedding", "POST", "/api/embeddings/generate",
            {
                "text": "Test text for embedding generation"
            }
        )
        
        # 5. Chat Endpoints
        console.print("\n[bold yellow]═══ Chat/LLM Endpoints ═══[/bold yellow]")
        
        # Basic chat without RAG
        await self.test_endpoint(
            "Chat (No RAG)", "POST", "/api/chat/send",
            {
                "message": "What is 2+2?",
                "use_rag": False,
                "max_tokens": 50
            }
        )
        
        # Chat with RAG (if workspace available)
        if self.workspace_id:
            await self.test_endpoint(
                "Chat with RAG", "POST", "/api/chat/send",
                {
                    "message": "What is in the test document?",
                    "use_rag": True,
                    "workspace_filter": self.workspace_id,
                    "rag_config": {
                        "similarity_threshold": 0.3,
                        "chunk_limit": 5
                    }
                }
            )
        
        # 6. Vector Search
        console.print("\n[bold yellow]═══ Vector Search ═══[/bold yellow]")
        
        # Semantic search
        search_data = await self.test_endpoint(
            "Semantic Search", "POST", "/api/vector/search/semantic",
            {
                "query": "test document",
                "limit": 5,
                "similarity_threshold": 0.3,
                "workspace_filter": self.workspace_id if self.workspace_id else None
            }
        )
        
        # Extract chunk_id for similarity search
        if search_data and search_data.get("results"):
            self.chunk_id = search_data["results"][0]["chunk_id"]
            
            # Similarity search
            await self.test_endpoint(
                "Similarity Search", "POST", "/api/vector/search/similarity",
                {
                    "chunk_id": self.chunk_id,
                    "limit": 3
                }
            )
            
            # Recommendations
            await self.test_endpoint(
                "Get Recommendations", "POST", "/api/vector/recommendations",
                {
                    "positive_chunk_ids": [self.chunk_id],
                    "limit": 5
                }
            )
        
        # 7. Hierarchical Retrieval
        console.print("\n[bold yellow]═══ Hierarchical Retrieval ═══[/bold yellow]")
        if self.workspace_id:
            await self.test_endpoint(
                "Hierarchical Retrieve", "POST", "/api/retrieve",
                {
                    "query": "test document",
                    "workspace_id": self.workspace_id,
                    "limit": 5,
                    "similarity_threshold": 0.3,
                    "include_kg_triples": False,
                    "include_document_info": True
                }
            )
        
        # 8. Persona Management
        console.print("\n[bold yellow]═══ Persona Management ═══[/bold yellow]")
        
        # List personas
        personas_data = await self.test_endpoint("List Personas", "GET", "/api/personas/list")
        
        # Create persona
        persona_data = await self.test_endpoint(
            "Create Persona", "POST", "/api/personas/create",
            {
                "id": f"test-persona-{uuid.uuid4().hex[:8]}",
                "name": "Test Persona",
                "tagline": "A test AI persona",
                "description": "This persona is for testing",
                "specialty": "testing",
                "system_prompt_template": "You are a helpful test assistant.",
                "workspace_id": self.workspace_id
            }
        )
        
        if persona_data and persona_data.get("persona_id"):
            self.persona_id = persona_data["persona_id"]
            
            # Chat with persona
            await self.test_endpoint(
                "Persona Chat", "POST", "/api/personas/chat",
                {
                    "persona_id": self.persona_id,
                    "message": "Hello, test persona!",
                    "max_tokens": 100
                }
            )
            
            # Delete persona
            await self.test_endpoint(
                "Delete Persona", "DELETE", f"/api/personas/{self.persona_id}"
            )
        
        # 9. Cleanup - Delete test workspace
        if self.workspace_id:
            console.print("\n[bold yellow]═══ Cleanup ═══[/bold yellow]")
            await self.test_endpoint(
                "Delete Workspace", "DELETE", f"/api/workspaces/{self.workspace_id}"
            )
        
        # Print summary
        self.print_summary()
        
    def print_summary(self):
        """Print test results summary."""
        console.print("\n" + "="*60)
        console.print(Panel.fit("📊 Test Results Summary", style="bold cyan"))
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Endpoint", style="cyan", width=40)
        table.add_column("Method", style="yellow", width=10)
        table.add_column("Status", width=10)
        table.add_column("Message", style="dim", width=40)
        
        success_count = 0
        total_count = len(self.test_results)
        
        for result in self.test_results:
            status_symbol = "✅" if result["success"] else "❌"
            status_color = "green" if result["success"] else "red"
            
            if result["success"]:
                success_count += 1
                
            table.add_row(
                result["endpoint"],
                result["method"],
                f"[{status_color}]{status_symbol}[/{status_color}]",
                result["message"][:40] + "..." if len(result["message"]) > 40 else result["message"]
            )
        
        console.print(table)
        
        # Summary stats
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        console.print(f"\n[bold]Total Tests:[/bold] {total_count}")
        console.print(f"[bold green]Passed:[/bold green] {success_count}")
        console.print(f"[bold red]Failed:[/bold red] {total_count - success_count}")
        console.print(f"[bold]Success Rate:[/bold] {success_rate:.1f}%")
        
        if success_rate == 100:
            console.print("\n[bold green]🎉 All tests passed![/bold green]")
        elif success_rate >= 80:
            console.print("\n[bold yellow]⚠️ Most tests passed, but some issues found.[/bold yellow]")
        else:
            console.print("\n[bold red]❌ Many tests failed. Please check the API server.[/bold red]")


async def main():
    """Main entry point."""
    try:
        async with EndpointTester() as tester:
            await tester.run_all_tests()
    except Exception as e:
        console.print(f"\n[bold red]Fatal Error:[/bold red] {e}")
        console.print("\n[yellow]Make sure:[/yellow]")
        console.print("1. CZero Engine app is running")
        console.print("2. API server is enabled (port 1421)")
        console.print("3. At least one LLM and embedding model are loaded")


if __name__ == "__main__":
    asyncio.run(main())