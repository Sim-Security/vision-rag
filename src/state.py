"""Pydantic state models for LangGraph Vision RAG."""
from typing import Optional
from pydantic import BaseModel, Field


class RetrievedImage(BaseModel):
    """Represents a retrieved image from ChromaDB."""
    path: str
    score: float
    id: str
    metadata: dict = Field(default_factory=dict)


class VisionRAGState(BaseModel):
    """State for the Vision RAG agent graph."""
    
    # Input
    query: str = ""
    
    # Embedding state
    query_embedding: Optional[list[float]] = None
    
    # Retrieval state
    retrieved_images: list[RetrievedImage] = Field(default_factory=list)
    current_image_idx: int = 0
    
    # Relevance checking
    is_relevant: bool = False
    relevance_reasoning: str = ""
    
    # Human-in-the-loop
    human_in_loop_enabled: bool = False
    human_approved: Optional[bool] = None  # None = pending, True/False = decided
    awaiting_human_input: bool = False
    
    # Self-correction
    iteration: int = 0
    max_iterations: int = 2
    
    # Multi-image comparison
    is_comparative_query: bool = False  # True if query requires multi-image comparison
    
    # Output
    answer: str = ""
    error: Optional[str] = None
    
    # Tracing
    langsmith_url: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class GraphConfig(BaseModel):
    """Configuration passed to the graph at runtime."""
    model_id: str = "google/gemini-2.5-flash-preview"
    human_in_loop: bool = False
    top_k: int = 3
    openrouter_api_key: str = ""
    cohere_api_key: str = ""
