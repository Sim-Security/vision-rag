"""LangGraph nodes for Vision RAG agent.

Implements the agentic flow with self-correction and human-in-the-loop.
"""
import os
import base64
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .state import VisionRAGState, RetrievedImage, GraphConfig
from .tools import (
    embed_query,
    get_cohere_client,
    get_chromadb_client,
    get_or_create_collection,
    search_chromadb,
    base64_from_image,
)
from .config import OPENROUTER_BASE_URL


def get_llm(config: GraphConfig) -> ChatOpenAI:
    """Get LLM client configured for OpenRouter."""
    return ChatOpenAI(
        model=config.model_id,
        openai_api_key=config.openrouter_api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/vision-rag-agent",
            "X-Title": "Vision RAG Agent",
        },
    )


# -----------------------------------------------------------------------------
# Node: Embed Query
# -----------------------------------------------------------------------------

def embed_query_node(state: VisionRAGState, config: GraphConfig) -> dict[str, Any]:
    """Embed the user's query using Cohere Embed-4."""
    try:
        cohere_client = get_cohere_client(config.cohere_api_key)
        query_embedding = embed_query(state.query, cohere_client)
        
        return {
            "query_embedding": query_embedding,
            "error": None,
        }
    except Exception as e:
        return {
            "query_embedding": None,
            "error": f"Failed to embed query: {str(e)}",
        }


# -----------------------------------------------------------------------------
# Node: Classify Query (detect comparative queries using LLM)
# -----------------------------------------------------------------------------

def classify_query_node(state: VisionRAGState, config: GraphConfig) -> dict[str, Any]:
    """Use LLM to classify if query requires comparing multiple images.
    
    This is more agentic than keyword matching - the LLM understands intent.
    """
    try:
        llm = get_llm(config)
        
        system_prompt = """You are a query classifier. Determine if the user's question requires comparing information across MULTIPLE images/documents to answer correctly.

Examples of COMPARATIVE queries (need multiple images):
- "Which company has the highest revenue?"
- "Compare Tesla and Nike's profit margins"
- "What's the best performing company?"

Examples of SINGLE-image queries:
- "What is Nike's revenue?"
- "How many layers does this network have?"
- "What are the Q4 sales?"

Respond with EXACTLY one word: COMPARATIVE or SINGLE"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {state.query}"),
        ])
        
        is_comparative = "COMPARATIVE" in response.content.upper()
        
        return {
            "is_comparative_query": is_comparative,
        }
        
    except Exception as e:
        # On error, default to single-image mode
        return {
            "is_comparative_query": False,
        }


# -----------------------------------------------------------------------------
# Node: Multi-Image Answer (for comparative queries)
# -----------------------------------------------------------------------------

def multi_image_answer_node(state: VisionRAGState, config: GraphConfig) -> dict[str, Any]:
    """Generate answer by analyzing ALL retrieved images together.
    
    Used for comparative queries that need to see multiple images at once.
    """
    if not state.retrieved_images:
        return {
            "answer": "I couldn't find any relevant images to compare.",
            "error": "No images available",
        }
    
    try:
        llm = get_llm(config)
        
        # Build content with all retrieved images
        content_parts = [
            {"type": "text", "text": f"Question: {state.query}\n\nI'm providing you with {len(state.retrieved_images)} images to analyze and compare:"}
        ]
        
        # Add each image
        for i, img in enumerate(state.retrieved_images):
            if os.path.exists(img.path):
                base64_img = base64_from_image(img.path)
                content_parts.append({"type": "text", "text": f"\n--- Image {i+1}: {os.path.basename(img.path)} ---"})
                content_parts.append({"type": "image_url", "image_url": {"url": base64_img}})
        
        system_prompt = """You are a visual analyst specializing in comparative analysis. You will receive multiple images and must analyze them together to answer the user's question.

Guidelines:
- Compare information across ALL provided images
- Cite specific numbers and data from each image
- Clearly identify which image/company/document each piece of data comes from
- Provide a definitive answer based on your comparison
- If a comparison isn't possible, explain why"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=content_parts),
        ])
        
        return {
            "answer": response.content,
            "is_relevant": True,  # If we got here, we answered
            "error": None,
        }
        
    except Exception as e:
        return {
            "answer": f"Failed to analyze multiple images: {str(e)}",
            "error": str(e),
        }


# -----------------------------------------------------------------------------
# Node: Search ChromaDB
# -----------------------------------------------------------------------------

def search_node(state: VisionRAGState, config: GraphConfig) -> dict[str, Any]:
    """Search ChromaDB for relevant images."""
    if state.query_embedding is None:
        return {
            "retrieved_images": [],
            "error": "No query embedding available",
        }
    
    try:
        client = get_chromadb_client()
        collection = get_or_create_collection(client)
        
        results = search_chromadb(
            collection,
            state.query_embedding,
            top_k=config.top_k,
        )
        
        retrieved = [
            RetrievedImage(
                path=r["path"],
                score=r["score"],
                id=r["id"],
                metadata=r["metadata"],
            )
            for r in results
        ]
        
        return {
            "retrieved_images": retrieved,
            "current_image_idx": 0,
            "error": None,
        }
    except Exception as e:
        return {
            "retrieved_images": [],
            "error": f"Search failed: {str(e)}",
        }


# -----------------------------------------------------------------------------
# Node: Human Review (for human-in-the-loop)
# -----------------------------------------------------------------------------

def human_review_node(state: VisionRAGState, config: GraphConfig) -> dict[str, Any]:
    """Pause for human review of retrieved image.
    
    This node sets a flag indicating we're waiting for human input.
    The Streamlit UI will handle the actual user interaction.
    """
    if not state.retrieved_images:
        return {
            "awaiting_human_input": False,
            "human_approved": None,
        }
    
    # Set flag to pause for human input
    return {
        "awaiting_human_input": True,
        "human_approved": None,  # Will be set by UI
    }


# -----------------------------------------------------------------------------
# Node: Relevance Check
# -----------------------------------------------------------------------------

def relevance_check_node(state: VisionRAGState, config: GraphConfig) -> dict[str, Any]:
    """Check if the current retrieved image is relevant to the query.
    
    Uses vision LLM to determine relevance.
    """
    if not state.retrieved_images or state.current_image_idx >= len(state.retrieved_images):
        return {
            "is_relevant": False,
            "relevance_reasoning": "No images available to check",
        }
    
    current_image = state.retrieved_images[state.current_image_idx]
    image_path = current_image.path
    
    if not os.path.exists(image_path):
        return {
            "is_relevant": False,
            "relevance_reasoning": f"Image file not found: {image_path}",
        }
    
    try:
        llm = get_llm(config)
        
        # Load image as base64
        base64_img = base64_from_image(image_path)
        
        # Create relevance check prompt
        system_prompt = """You are an image relevance checker. Your job is to determine if an image is relevant to answering a user's question.

Respond with EXACTLY this format:
RELEVANT: YES or NO
REASON: Brief explanation (one sentence)

Be strict - the image should contain information that directly helps answer the question."""

        user_content = [
            {"type": "text", "text": f"Question: {state.query}\n\nIs this image relevant to answering the question?"},
            {"type": "image_url", "image_url": {"url": base64_img}},
        ]
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        
        response_text = response.content.upper()
        is_relevant = "RELEVANT: YES" in response_text or "RELEVANT:YES" in response_text
        
        # Extract reasoning
        reasoning = response.content
        if "REASON:" in response.content:
            reasoning = response.content.split("REASON:")[-1].strip()
        
        return {
            "is_relevant": is_relevant,
            "relevance_reasoning": reasoning,
            "iteration": state.iteration + 1,
        }
        
    except Exception as e:
        # On error, assume relevant to avoid blocking
        return {
            "is_relevant": True,
            "relevance_reasoning": f"Relevance check failed: {str(e)}. Proceeding anyway.",
            "iteration": state.iteration + 1,
        }


# -----------------------------------------------------------------------------
# Node: Rerank (try next image)
# -----------------------------------------------------------------------------

def rerank_node(state: VisionRAGState, config: GraphConfig) -> dict[str, Any]:
    """Move to the next retrieved image for checking."""
    next_idx = state.current_image_idx + 1
    
    return {
        "current_image_idx": next_idx,
        "is_relevant": False,
        "human_approved": None,  # Reset for new image
    }


# -----------------------------------------------------------------------------
# Node: Generate Answer
# -----------------------------------------------------------------------------

def answer_node(state: VisionRAGState, config: GraphConfig) -> dict[str, Any]:
    """Generate final answer using vision LLM."""
    if not state.retrieved_images or state.current_image_idx >= len(state.retrieved_images):
        return {
            "answer": "I couldn't find a relevant image to answer your question.",
            "error": "No relevant image available",
        }
    
    current_image = state.retrieved_images[state.current_image_idx]
    image_path = current_image.path
    
    if not os.path.exists(image_path):
        return {
            "answer": f"The image file was not found: {image_path}",
            "error": "Image file missing",
        }
    
    try:
        llm = get_llm(config)
        
        # Load image as base64
        base64_img = base64_from_image(image_path)
        
        # Create answer prompt
        system_prompt = """You are a helpful visual analyst. Answer the user's question based on the provided image.

Guidelines:
- Be thorough and provide specific details from the image
- If the image contains charts/graphs, cite specific numbers
- Provide context for your answer
- Be factual - only state what you can see in the image"""

        user_content = [
            {"type": "text", "text": f"Question: {state.query}"},
            {"type": "image_url", "image_url": {"url": base64_img}},
        ]
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        
        return {
            "answer": response.content,
            "error": None,
        }
        
    except Exception as e:
        return {
            "answer": f"Failed to generate answer: {str(e)}",
            "error": str(e),
        }


# -----------------------------------------------------------------------------
# Node: No Result
# -----------------------------------------------------------------------------

def no_result_node(state: VisionRAGState, config: GraphConfig) -> dict[str, Any]:
    """Handle case where no relevant image was found after all retries."""
    return {
        "answer": "I searched through the available images but couldn't find one that's relevant to your question. Try uploading more images or rephrasing your question.",
        "is_relevant": False,
    }


# -----------------------------------------------------------------------------
# Conditional Edge Functions
# -----------------------------------------------------------------------------

def should_check_human(state: VisionRAGState, config: GraphConfig) -> str:
    """Decide whether to route to human review or relevance check."""
    if config.human_in_loop and state.retrieved_images:
        return "human_review"
    return "relevance_check"


def after_human_review(state: VisionRAGState, config: GraphConfig) -> str:
    """Route after human review.
    
    Note: Human-in-the-loop in Streamlit is handled by the UI showing the image
    before we get here. This node just processes the decision.
    """
    if state.human_approved is False:
        return "rerank"  # Human rejected - try next image
    # Default: proceed with relevance check (human approved or skipped)
    return "relevance_check"


def after_relevance_check(state: VisionRAGState, config: GraphConfig) -> str:
    """Route after relevance check."""
    if state.is_relevant:
        return "answer"
    
    # Check if we have more images to try
    if state.current_image_idx + 1 < len(state.retrieved_images):
        # Check iteration limit
        if state.iteration < state.max_iterations:
            return "rerank"
    
    # No more images or max iterations reached
    return "no_result"
