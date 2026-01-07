"""LangGraph graph construction for Vision RAG agent."""
from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from .state import VisionRAGState, GraphConfig
from .nodes import (
    embed_query_node,
    classify_query_node,
    search_node,
    human_review_node,
    relevance_check_node,
    rerank_node,
    answer_node,
    multi_image_answer_node,
    no_result_node,
    should_check_human,
    after_human_review,
    after_relevance_check,
)

# Global checkpointer for human-in-the-loop persistence
_checkpointer = MemorySaver()


def create_vision_rag_graph(config: GraphConfig, use_checkpointer: bool = False) -> StateGraph:
    """Create the Vision RAG LangGraph.
    
    Graph Flow:
    START → embed_query → classify_query → search → 
        → [if comparative] → multi_image_answer → END
        → [if single] → [human_review?] → relevance_check → 
            → answer (if relevant) 
            → rerank → relevance_check (if not relevant, retry available)
            → no_result (if max retries exceeded)
    
    Args:
        config: Graph configuration
        use_checkpointer: If True, use checkpointer for human-in-the-loop
        
    Returns:
        Compiled StateGraph
    """
    # Create graph with state type
    workflow = StateGraph(VisionRAGState)
    
    # Add nodes with config bound
    workflow.add_node("embed_query", lambda s: embed_query_node(s, config))
    workflow.add_node("classify_query", lambda s: classify_query_node(s, config))
    workflow.add_node("search", lambda s: search_node(s, config))
    workflow.add_node("human_review", lambda s: human_review_node(s, config))
    workflow.add_node("relevance_check", lambda s: relevance_check_node(s, config))
    workflow.add_node("rerank", lambda s: rerank_node(s, config))
    workflow.add_node("answer", lambda s: answer_node(s, config))
    workflow.add_node("multi_image_answer", lambda s: multi_image_answer_node(s, config))
    workflow.add_node("no_result", lambda s: no_result_node(s, config))
    
    # Set entry point
    workflow.set_entry_point("embed_query")
    
    # Add edges
    workflow.add_edge("embed_query", "classify_query")
    workflow.add_edge("classify_query", "search")
    
    # After search: route based on query type
    def after_search_routing(state: VisionRAGState) -> str:
        """Route based on query classification."""
        if state.is_comparative_query:
            return "multi_image_answer"  # Skip single-image flow
        # Continue to single-image flow
        return should_check_human(state, config)
    
    workflow.add_conditional_edges(
        "search",
        after_search_routing,
        {
            "multi_image_answer": "multi_image_answer",
            "human_review": "human_review",
            "relevance_check": "relevance_check",
        }
    )
    
    # After human review: route based on approval
    workflow.add_conditional_edges(
        "human_review",
        lambda s: after_human_review(s, config),
        {
            "relevance_check": "relevance_check",
            "rerank": "rerank",
        }
    )
    
    # After relevance check: answer, rerank, or no_result
    workflow.add_conditional_edges(
        "relevance_check",
        lambda s: after_relevance_check(s, config),
        {
            "answer": "answer",
            "rerank": "rerank",
            "no_result": "no_result",
        }
    )
    
    # After rerank: go back to relevance check
    workflow.add_edge("rerank", "relevance_check")
    
    # Terminal nodes
    workflow.add_edge("answer", END)
    workflow.add_edge("multi_image_answer", END)
    workflow.add_edge("no_result", END)
    
    # Compile with checkpointer and interrupt for human-in-the-loop
    if use_checkpointer and config.human_in_loop:
        return workflow.compile(
            checkpointer=_checkpointer,
            interrupt_before=["human_review"],  # Pause BEFORE human review
        )
    elif use_checkpointer:
        return workflow.compile(checkpointer=_checkpointer)
    else:
        return workflow.compile()


def run_vision_rag(
    query: str,
    config: GraphConfig,
    human_in_loop: bool = False,
    max_iterations: int = 2,
) -> VisionRAGState:
    """Run the Vision RAG graph.
    
    Args:
        query: User's question
        config: Graph configuration
        human_in_loop: Whether to pause for human review
        max_iterations: Max self-correction attempts
        
    Returns:
        Final state with answer
    """
    # Create initial state
    initial_state = VisionRAGState(
        query=query,
        human_in_loop_enabled=human_in_loop,
        max_iterations=max_iterations,
    )
    
    # Update config
    config.human_in_loop = human_in_loop
    
    # Create and run graph
    graph = create_vision_rag_graph(config)
    
    # Run to completion (or until human review pause)
    final_state = graph.invoke(initial_state.model_dump())
    
    return VisionRAGState(**final_state)


def create_graph_visualization() -> str:
    """Generate a Mermaid diagram of the graph.
    
    Returns:
        Mermaid diagram string
    """
    return """
```mermaid
graph TD
    A[START] --> B[embed_query]
    B --> C[search]
    C --> D{human_in_loop?}
    D -->|Yes| E[human_review]
    D -->|No| F[relevance_check]
    E --> G{approved?}
    G -->|Yes| F
    G -->|No| H[rerank]
    F --> I{is_relevant?}
    I -->|Yes| J[answer]
    I -->|No, retry available| H
    I -->|No, max retries| K[no_result]
    H --> F
    J --> L[END]
    K --> L
```
"""
