"""Configuration for LangGraph Vision RAG."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "vision-rag-agent")

# OpenRouter Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Available models via OpenRouter (cheap/fast models prioritized)
AVAILABLE_MODELS = {
    "grok-4.1-fast": {
        "id": "x-ai/grok-4.1-fast",
        "name": "Grok 4.1 Fast",
        "supports_vision": True,
        "cost": "$",
    },
    "gemini-2.5-flash": {
        "id": "google/gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "supports_vision": True,
        "cost": "$",
    },
    "gemini-2.0-flash": {
        "id": "google/gemini-2.0-flash-001",
        "name": "Gemini 2.0 Flash",
        "supports_vision": True,
        "cost": "$",
    },
    "gpt-4o-mini": {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o Mini",
        "supports_vision": True,
        "cost": "$$",
    },
    "claude-3.5-haiku": {
        "id": "anthropic/claude-3.5-haiku",
        "name": "Claude 3.5 Haiku",
        "supports_vision": True,
        "cost": "$$",
    },
}

DEFAULT_MODEL = "grok-4.1-fast"

# Cohere Configuration
COHERE_EMBED_MODEL = "embed-v4.0"

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHROMA_COLLECTION_NAME = "vision_rag_images"

# Image Processing
MAX_IMAGE_PIXELS = 1568 * 1568  # Max resolution for images

# RAG Configuration
DEFAULT_TOP_K = 3  # Number of images to retrieve
MAX_CORRECTION_ITERATIONS = 2  # Max self-correction attempts
