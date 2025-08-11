"""Modern LangGraph Integration with CZero Engine (2025 Patterns)

This example demonstrates the latest LangGraph patterns with CZero Engine:
- Using MessagesState and add_messages for state management
- Command-based routing and state updates
- Runtime context injection
- Structured tool calling without native LLM support
- Human-in-the-loop patterns

Requirements:
    pip install langgraph>=0.2.0 langchain-core>=0.3.0
"""

import asyncio
import json
from typing import Annotated, Any, Dict, List, Optional, Literal, Sequence
from typing_extensions import TypedDict
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import (
    AIMessage,
    BaseMessage, 
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AnyMessage
)
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from czero_engine import CZeroEngineClient
from czero_engine.workflows import RAGWorkflow


# ============= Modern State Definition =============
class AgentState(MessagesState):
    """Enhanced state using MessagesState as base with custom fields."""
    # Messages are inherited from MessagesState with add_messages reducer
    # Additional custom fields for our agent
    documents: Annotated[List[str], lambda x, y: x + y]  # Accumulate documents
    current_query: Optional[str]
    search_results: Optional[Dict[str, Any]]
    user_context: Dict[str, Any]
    workflow_stage: Literal["search", "analyze", "respond", "complete"]
    confidence_score: float


class CZeroEngineLLM(BaseChatModel):
    """Modern CZero Engine LLM wrapper for LangGraph."""
    
    client: Optional[CZeroEngineClient] = None
    use_rag: bool = True
    max_tokens: int = 1024
    temperature: float = 0.7
    base_url: str = "http://localhost:1421"
    persona_id: str = "gestalt-default"
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.client:
            self.client = CZeroEngineClient(base_url=self.base_url)
    
    async def __aenter__(self):
        if self.client:
            await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    @property
    def _llm_type(self) -> str:
        return "czero-engine-2025"
    
    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        """Sync wrapper for async generation."""
        import asyncio
        loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()
        return loop.run_until_complete(self._agenerate(messages, **kwargs))
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using CZero Engine."""
        
        # Build conversation context
        conversation = []
        system_prompt = None
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                conversation.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                conversation.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                conversation.append(f"Tool Result: {msg.content}")
        
        prompt = "\n\n".join(conversation)
        
        # Use persona chat for better responses
        if self.persona_id:
            response = await self.client.persona_chat(
                persona_id=self.persona_id,
                message=prompt,
                system_prompt_template=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        else:
            response = await self.client.chat(
                message=prompt,
                use_rag=self.use_rag,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        
        message = AIMessage(content=response.response)
        return ChatResult(generations=[ChatGeneration(message=message)])


# ============= Modern Tool Definitions =============
@tool
async def search_knowledge_base(query: str) -> Dict[str, Any]:
    """Search CZero Engine knowledge base for relevant information.
    
    Args:
        query: Search query
        
    Returns:
        Dictionary with search results and metadata
    """
    async with CZeroEngineClient() as client:
        results = await client.semantic_search(
            query=query,
            limit=5,
            similarity_threshold=0.5,
            include_hierarchy=True
        )
        
        return {
            "query": query,
            "found": len(results.results) > 0,
            "count": len(results.results),
            "results": [
                {
                    "content": r.content[:200],
                    "score": r.similarity,
                    "doc_id": r.document_id
                }
                for r in results.results
            ]
        }


@tool
async def analyze_document(doc_id: str) -> Dict[str, Any]:
    """Analyze a specific document in detail.
    
    Args:
        doc_id: Document ID to analyze
        
    Returns:
        Detailed document analysis
    """
    async with CZeroEngineClient() as client:
        # Get similar chunks to understand document context
        results = await client.find_similar_chunks(
            chunk_id=doc_id,
            limit=3,
            similarity_threshold=0.7
        )
        
        return {
            "doc_id": doc_id,
            "related_chunks": len(results.results),
            "context": " ".join([r.content[:100] for r in results.results])
        }


@tool
async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for given text.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector (first 10 dimensions for display)
    """
    async with CZeroEngineClient() as client:
        result = await client.generate_embedding(text)
        # Return first 10 dimensions for display
        return result.embedding[:10] if result.embedding else []


# ============= Modern Node Functions =============
async def search_node(state: AgentState) -> Command[Literal["analyze", "respond"]]:
    """Search for relevant documents and route based on results."""
    print("🔍 Search Node - Looking for relevant documents...")
    
    # Extract query from last message
    last_message = state["messages"][-1]
    query = last_message.content if isinstance(last_message, HumanMessage) else state.get("current_query", "")
    
    # Perform search
    search_results = await search_knowledge_base.ainvoke({"query": query})
    
    # Update state with Command pattern
    state_update = {
        "current_query": query,
        "search_results": search_results,
        "documents": [r["content"] for r in search_results["results"]],
        "workflow_stage": "analyze" if search_results["found"] else "respond",
        "confidence_score": max([r["score"] for r in search_results["results"]]) if search_results["found"] else 0.0
    }
    
    # Route based on search results
    next_node = "analyze" if search_results["found"] else "respond"
    
    print(f"   Found {search_results['count']} documents, routing to: {next_node}")
    
    return Command(
        update=state_update,
        goto=next_node
    )


async def analyze_node(state: AgentState) -> Command[Literal["respond"]]:
    """Analyze search results and prepare context."""
    print("🧠 Analyze Node - Processing search results...")
    
    # Build analysis context
    context_parts = []
    for i, doc in enumerate(state["documents"][:3], 1):
        context_parts.append(f"Document {i}: {doc}")
    
    # Create analysis message
    analysis_msg = AIMessage(
        content=f"Found {len(state['documents'])} relevant documents with confidence score: {state['confidence_score']:.2f}"
    )
    
    # Update state
    state_update = {
        "messages": [analysis_msg],
        "workflow_stage": "respond",
        "user_context": {
            "has_context": True,
            "document_count": len(state["documents"]),
            "confidence": state["confidence_score"]
        }
    }
    
    print(f"   Analysis complete, confidence: {state['confidence_score']:.2f}")
    
    return Command(
        update=state_update,
        goto="respond"
    )


async def respond_node(state: AgentState) -> Command[Literal["human_review", "complete"]]:
    """Generate response using LLM with context."""
    print("💬 Respond Node - Generating response...")
    
    async with CZeroEngineLLM(use_rag=True) as llm:
        # Prepare context-aware prompt
        messages = []
        
        # Add system message with context
        if state.get("documents"):
            context = "\n\n".join(state["documents"])
            messages.append(SystemMessage(
                content=f"Use this context to answer: {context}"
            ))
        
        # Add conversation history
        messages.extend(state["messages"])
        
        # Generate response
        result = await llm._agenerate(messages)
        response_content = result.generations[0].message.content
        
        # Determine if human review is needed
        needs_review = state.get("confidence_score", 1.0) < 0.7
        
        # Update state
        state_update = {
            "messages": [AIMessage(content=response_content)],
            "workflow_stage": "complete"
        }
        
        next_node = "human_review" if needs_review else "complete"
        
        print(f"   Response generated, routing to: {next_node}")
        
        return Command(
            update=state_update,
            goto=next_node
        )


def human_review_node(state: AgentState) -> Command[Literal["complete"]]:
    """Simulate human review of low-confidence responses."""
    print("👤 Human Review Node - Flagging for review...")
    
    review_msg = HumanMessage(
        content="[Auto-approved after review - confidence threshold met]"
    )
    
    return Command(
        update={
            "messages": [review_msg],
            "workflow_stage": "complete"
        },
        goto="complete"
    )


def complete_node(state: AgentState) -> Dict[str, Any]:
    """Final node to mark workflow as complete."""
    print("✅ Complete Node - Workflow finished")
    
    return {
        "workflow_stage": "complete"
    }


# ============= Modern Graph Construction =============
def create_modern_rag_graph() -> CompiledStateGraph:
    """Create a modern RAG graph using 2025 LangGraph patterns."""
    
    # Initialize graph with custom state
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("complete", complete_node)
    
    # Set entry point
    workflow.add_edge(START, "search")
    
    # Edges are handled by Command returns in nodes
    # But we still need to add terminal edges
    workflow.add_edge("complete", END)
    
    # Add checkpointer for memory
    memory = MemorySaver()
    
    # Compile with configuration
    return workflow.compile(
        checkpointer=memory,
        # interrupt_before=["human_review"]  # Uncomment for real human-in-loop
    )


# ============= Modern Agent with Tool Calling =============
async def create_tool_agent():
    """Create an agent that can use tools despite CZero not having native tool support."""
    
    print("\n🛠️ Tool-Based Agent Example")
    print("-" * 40)
    
    # Define a custom state for tool usage
    class ToolState(MessagesState):
        tool_calls: List[Dict[str, Any]]
        tool_results: List[Dict[str, Any]]
    
    workflow = StateGraph(ToolState)
    
    async def decide_tool_use(state: ToolState) -> Command[Literal["use_tools", "respond"]]:
        """Decide whether to use tools based on the query."""
        last_msg = state["messages"][-1].content.lower()
        
        # Simple heuristic for tool use
        if "search" in last_msg or "find" in last_msg:
            tool_call = {
                "tool": "search_knowledge_base",
                "args": {"query": state["messages"][-1].content}
            }
            return Command(
                update={"tool_calls": [tool_call]},
                goto="use_tools"
            )
        elif "embed" in last_msg:
            tool_call = {
                "tool": "generate_embedding",
                "args": {"text": state["messages"][-1].content}
            }
            return Command(
                update={"tool_calls": [tool_call]},
                goto="use_tools"
            )
        else:
            return Command(goto="respond")
    
    async def use_tools_node(state: ToolState) -> Command[Literal["respond"]]:
        """Execute tool calls."""
        results = []
        
        for tool_call in state.get("tool_calls", []):
            if tool_call["tool"] == "search_knowledge_base":
                result = await search_knowledge_base.ainvoke(tool_call["args"])
            elif tool_call["tool"] == "generate_embedding":
                result = await generate_embedding.ainvoke(tool_call["args"])
            else:
                result = {"error": f"Unknown tool: {tool_call['tool']}"}
            
            results.append(result)
            
            # Add tool result as message
            tool_msg = ToolMessage(
                content=json.dumps(result, indent=2),
                tool_call_id=tool_call.get("id", "tool_call")
            )
            
        return Command(
            update={
                "tool_results": results,
                "messages": [tool_msg]
            },
            goto="respond"
        )
    
    async def respond_with_tools(state: ToolState) -> Dict[str, Any]:
        """Generate response considering tool results."""
        async with CZeroEngineLLM() as llm:
            result = await llm._agenerate(state["messages"])
            return {"messages": [result.generations[0].message]}
    
    # Build the graph
    workflow.add_node("decide", decide_tool_use)
    workflow.add_node("use_tools", use_tools_node)
    workflow.add_node("respond", respond_with_tools)
    
    workflow.add_edge(START, "decide")
    workflow.add_edge("respond", END)
    
    return workflow.compile()


# ============= Main Examples =============
async def main():
    """Run modern LangGraph examples with CZero Engine."""
    
    print("\n🚀 Modern LangGraph + CZero Engine (2025 Patterns)")
    print("=" * 50)
    
    try:
        # Verify CZero Engine
        async with CZeroEngineClient() as client:
            health = await client.health_check()
            print(f"✅ CZero Engine Status: {health.status}\n")
        
        # Example 1: Modern RAG Graph with Commands
        print("1️⃣ Modern RAG Graph with Command-based Routing")
        print("-" * 40)
        
        rag_graph = create_modern_rag_graph()
        
        # Run with thread ID for memory
        config = {"configurable": {"thread_id": "session_001"}}
        
        initial_state = {
            "messages": [
                HumanMessage(content="What are the key features of CZero Engine?")
            ],
            "documents": [],
            "workflow_stage": "search",
            "confidence_score": 0.0,
            "user_context": {}
        }
        
        result = await rag_graph.ainvoke(initial_state, config)
        
        print(f"\nFinal response: {result['messages'][-1].content[:300]}...")
        print(f"Workflow stage: {result['workflow_stage']}")
        print(f"Documents found: {len(result['documents'])}")
        
        # Example 2: Tool-based Agent
        print("\n2️⃣ Tool-Based Agent (Without Native Tool Calling)")
        print("-" * 40)
        
        tool_agent = await create_tool_agent()
        
        queries = [
            "Search for information about semantic search",
            "Generate an embedding for: artificial intelligence"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            result = await tool_agent.ainvoke({
                "messages": [HumanMessage(content=query)],
                "tool_calls": [],
                "tool_results": []
            })
            
            if result.get("tool_results"):
                print(f"Tools used: {len(result['tool_results'])}")
            print(f"Response: {result['messages'][-1].content[:200]}...")
        
        # Example 3: Streaming with State Updates
        print("\n3️⃣ Streaming State Updates")
        print("-" * 40)
        
        # Stream through the graph to see state updates
        stream_state = {
            "messages": [
                HumanMessage(content="Explain how RAG systems work")
            ],
            "documents": [],
            "workflow_stage": "search",
            "confidence_score": 0.0,
            "user_context": {}
        }
        
        print("Streaming through nodes:")
        async for event in rag_graph.astream_events(
            stream_state,
            config,
            version="v2"
        ):
            if event["event"] == "on_chain_start":
                print(f"  → Starting: {event['name']}")
            elif event["event"] == "on_chain_end":
                if "complete" in event["name"].lower():
                    print(f"  ✓ Completed: {event['name']}")
        
        print("\n✅ All modern examples completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        print("\nEnsure:")
        print("1. CZero Engine is running")
        print("2. Latest LangGraph: pip install langgraph>=0.2.0")
        print("3. Documents are indexed in CZero Engine")


if __name__ == "__main__":
    asyncio.run(main())