"""Test the processing pipeline to ensure it matches UI behavior."""

import asyncio
import httpx
import uuid
import tempfile
from pathlib import Path
import json

async def test_processing_pipeline():
    """Test that API processing matches UI processing behavior."""
    base_url = "http://localhost:1421"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("=" * 60)
        print("Testing Processing Pipeline")
        print("=" * 60)
        
        # 1. Create a test workspace with actual files
        print("\n1. Creating test workspace with files...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_id = None
            
            try:
                # Create test files with content
                test_files = []
                
                # Create a markdown file
                md_file = Path(temp_dir) / "test_document.md"
                md_file.write_text("""# Test Document

## Introduction
This is a test document for hierarchical chunking.

## Section 1: Background
The document processing pipeline should create hierarchical chunks.
This means larger parent chunks and smaller child chunks.

## Section 2: Details
Here are some important details:
- Point 1: Hierarchical chunking improves retrieval
- Point 2: It provides better context
- Point 3: It enables multi-level search

## Conclusion
This test verifies the processing pipeline works correctly.
""")
                test_files.append(str(md_file))
                
                # Create a text file
                txt_file = Path(temp_dir) / "notes.txt"
                txt_file.write_text("""Important Notes

These are some notes to test processing.
The system should chunk this content hierarchically.
Each paragraph might become a chunk.
And larger sections become parent chunks.

This is another paragraph with different content.
It should be processed separately but maintain relationships.
""")
                test_files.append(str(txt_file))
                
                print(f"   Created {len(test_files)} test files in {temp_dir}")
                
                # Create workspace
                response = await client.post(
                    f"{base_url}/api/workspaces/create",
                    json={
                        "name": f"Process Test {uuid.uuid4().hex[:8]}",
                        "path": temp_dir,
                        "description": "Testing processing pipeline"
                    }
                )
                response.raise_for_status()
                workspace_data = response.json()
                workspace_id = workspace_data["id"]
                print(f"   ✅ Created workspace: {workspace_id}")
                
                # 2. Process the files
                print("\n2. Processing workspace files...")
                response = await client.post(
                    f"{base_url}/api/workspaces/process",
                    json={
                        "workspace_id": workspace_id,
                        "files": test_files,
                        "config": None  # Use default config (hierarchical)
                    }
                )
                
                if response.status_code == 200:
                    process_result = response.json()
                    print(f"   ✅ Processing successful!")
                    print(f"      Files processed: {process_result['files_processed']}")
                    print(f"      Files failed: {process_result['files_failed']}")
                    print(f"      Chunks created: {process_result['chunks_created']}")
                    print(f"      Processing time: {process_result['processing_time']:.2f}s")
                    print(f"      Message: {process_result['message']}")
                    
                    # 3. Verify chunks were created by searching
                    print("\n3. Verifying hierarchical chunks...")
                    
                    # Search for content
                    response = await client.post(
                        f"{base_url}/api/vector/search/semantic",
                        json={
                            "query": "hierarchical chunking",
                            "workspace_filter": workspace_id,
                            "limit": 5,
                            "include_hierarchy": True
                        }
                    )
                    
                    if response.status_code == 200:
                        search_results = response.json()
                        if search_results.get("results"):
                            print(f"   ✅ Found {len(search_results['results'])} chunks")
                            
                            # Check first result
                            first_result = search_results["results"][0]
                            print(f"      Chunk ID: {first_result['chunk_id'][:16]}...")
                            print(f"      Content preview: {first_result['content'][:100]}...")
                            print(f"      Similarity: {first_result['similarity']:.3f}")
                            
                            # Check for hierarchical structure
                            if first_result.get('parent_chunk'):
                                print(f"      ✅ Has parent chunk (hierarchical structure confirmed)")
                            elif first_result.get('hierarchy_path'):
                                print(f"      ✅ Has hierarchy path (hierarchical structure confirmed)")
                            else:
                                print(f"      ⚠️ No hierarchical structure detected in search results")
                        else:
                            print(f"   ❌ No chunks found in search")
                    else:
                        print(f"   ❌ Search failed: {response.status_code}")
                    
                    # 4. Test hierarchical retrieval
                    print("\n4. Testing hierarchical retrieval...")
                    response = await client.post(
                        f"{base_url}/api/retrieve",
                        json={
                            "query": "test document",
                            "workspace_id": workspace_id,
                            "limit": 3,
                            "include_document_info": True
                        }
                    )
                    
                    if response.status_code == 200:
                        retrieval_results = response.json()
                        if retrieval_results.get("results"):
                            print(f"   ✅ Hierarchical retrieval found {len(retrieval_results['results'])} results")
                            
                            first = retrieval_results["results"][0]
                            if first.get("small_chunk") and first.get("big_chunk"):
                                print(f"      ✅ Has both small and big chunks")
                                print(f"      Small chunk: {first['small_chunk']['content'][:50]}...")
                                print(f"      Big chunk: {first['big_chunk']['content'][:50]}...")
                            
                            if first.get("document_info"):
                                doc_info = first["document_info"]
                                print(f"      Document: {doc_info['document_name']}")
                                print(f"      Total chunks: {doc_info['total_chunks']}")
                        else:
                            print(f"   ❌ No hierarchical results found")
                    else:
                        print(f"   ❌ Hierarchical retrieval failed: {response.status_code}")
                    
                else:
                    print(f"   ❌ Processing failed: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"      Error: {error_data.get('error', 'Unknown')}")
                        print(f"      Details: {error_data.get('details', 'None')}")
                    except:
                        print(f"      Raw response: {response.text}")
                
                # 5. Cleanup
                if workspace_id:
                    print("\n5. Cleaning up...")
                    response = await client.delete(f"{base_url}/api/workspaces/{workspace_id}")
                    if response.status_code == 200:
                        print(f"   ✅ Workspace deleted")
                    else:
                        print(f"   ⚠️ Failed to delete workspace")
                        
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_processing_pipeline())