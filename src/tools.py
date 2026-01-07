"""Tools for Vision RAG - embedding, vector search, PDF processing.

Ported from original vision_rag.py with ChromaDB integration.
"""
import os
import io
import base64
from typing import Optional
from pathlib import Path

import numpy as np
import cohere
from PIL import Image
import chromadb
from chromadb.config import Settings
import fitz  # PyMuPDF

from .config import (
    COHERE_EMBED_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    MAX_IMAGE_PIXELS,
)


# -----------------------------------------------------------------------------
# Image Processing (from original)
# -----------------------------------------------------------------------------

def resize_image(pil_image: Image.Image) -> None:
    """Resizes the image in-place if it exceeds max_pixels."""
    org_width, org_height = pil_image.size
    if org_width * org_height > MAX_IMAGE_PIXELS:
        scale_factor = (MAX_IMAGE_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))


def base64_from_image(img_path: str) -> str:
    """Converts an image file to a base64 encoded string."""
    pil_image = Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"
    resize_image(pil_image)
    
    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64," + base64.b64encode(img_buffer.read()).decode("utf-8")
    
    return img_data


def pil_to_base64(pil_image: Image.Image) -> str:
    """Converts a PIL image to a base64 encoded string."""
    img_format = pil_image.format if pil_image.format else "PNG"
    resize_image(pil_image)
    
    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64," + base64.b64encode(img_buffer.read()).decode("utf-8")
    
    return img_data


# -----------------------------------------------------------------------------
# Cohere Embedding
# -----------------------------------------------------------------------------

def get_cohere_client(api_key: str) -> cohere.ClientV2:
    """Get a Cohere client instance."""
    return cohere.ClientV2(api_key=api_key)


def embed_images(
    image_paths: list[str],
    cohere_client: cohere.ClientV2,
    progress_callback: Optional[callable] = None,
) -> list[list[float]]:
    """Embed multiple images using Cohere Embed-4.
    
    Args:
        image_paths: List of paths to images
        cohere_client: Initialized Cohere client
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        List of embedding vectors
    """
    embeddings = []
    
    for i, path in enumerate(image_paths):
        try:
            base64_img = base64_from_image(path)
            response = cohere_client.embed(
                model=COHERE_EMBED_MODEL,
                input_type="search_document",
                embedding_types=["float"],
                images=[base64_img],
            )
            
            if response.embeddings and response.embeddings.float:
                embeddings.append(response.embeddings.float[0])
            else:
                # Return zero vector as placeholder for failed embeddings
                embeddings.append([0.0] * 1024)  # Cohere Embed-4 dimension
                
        except Exception as e:
            print(f"Error embedding {path}: {e}")
            embeddings.append([0.0] * 1024)
        
        if progress_callback:
            progress_callback(i + 1, len(image_paths))
    
    return embeddings


def embed_query(query: str, cohere_client: cohere.ClientV2) -> list[float]:
    """Embed a text query using Cohere Embed-4.
    
    Args:
        query: Text query to embed
        cohere_client: Initialized Cohere client
        
    Returns:
        Embedding vector
    """
    response = cohere_client.embed(
        model=COHERE_EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
        texts=[query],
    )
    
    if response.embeddings and response.embeddings.float:
        return response.embeddings.float[0]
    
    raise ValueError("Failed to get query embedding from Cohere")


# -----------------------------------------------------------------------------
# ChromaDB Vector Store
# -----------------------------------------------------------------------------

def get_chromadb_client(persist_dir: Optional[str] = None) -> chromadb.Client:
    """Get a ChromaDB client with persistence."""
    persist_path = persist_dir or CHROMA_PERSIST_DIR
    os.makedirs(persist_path, exist_ok=True)
    
    return chromadb.PersistentClient(
        path=persist_path,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.Client,
    collection_name: Optional[str] = None,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    name = collection_name or CHROMA_COLLECTION_NAME
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )


def add_images_to_chromadb(
    collection: chromadb.Collection,
    image_paths: list[str],
    embeddings: list[list[float]],
    metadata: Optional[list[dict]] = None,
) -> list[str]:
    """Add images and their embeddings to ChromaDB.
    
    Args:
        collection: ChromaDB collection
        image_paths: List of image paths
        embeddings: List of embedding vectors
        metadata: Optional list of metadata dicts
        
    Returns:
        List of generated IDs
    """
    # Generate unique IDs based on paths
    ids = [f"img_{hash(path) & 0xffffffff}" for path in image_paths]
    
    # Prepare metadata
    if metadata is None:
        metadata = [{"path": path, "filename": os.path.basename(path)} for path in image_paths]
    else:
        # Ensure path is in metadata
        for i, m in enumerate(metadata):
            m["path"] = image_paths[i]
            m["filename"] = os.path.basename(image_paths[i])
    
    # Upsert to handle duplicates gracefully
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadata,
        documents=image_paths,  # Store path as document for easy retrieval
    )
    
    return ids


def search_chromadb(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Search ChromaDB for similar images.
    
    Args:
        collection: ChromaDB collection
        query_embedding: Query embedding vector
        top_k: Number of results to return
        
    Returns:
        List of dicts with {id, path, score, metadata}
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    retrieved = []
    if results["ids"] and results["ids"][0]:
        for i, id_ in enumerate(results["ids"][0]):
            retrieved.append({
                "id": id_,
                "path": results["documents"][0][i] if results["documents"] else "",
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })
    
    return retrieved


def get_collection_count(collection: chromadb.Collection) -> int:
    """Get the number of items in a collection."""
    return collection.count()


def clear_collection(collection: chromadb.Collection) -> None:
    """Clear all items from a collection."""
    # Get all IDs and delete them
    all_items = collection.get()
    if all_items["ids"]:
        collection.delete(ids=all_items["ids"])


# -----------------------------------------------------------------------------
# PDF Processing (from original)
# -----------------------------------------------------------------------------

def process_pdf(
    pdf_file,
    output_folder: str = "pdf_pages",
    progress_callback: Optional[callable] = None,
) -> list[str]:
    """Extract pages from a PDF as images.
    
    Args:
        pdf_file: File-like object (e.g., Streamlit UploadedFile)
        output_folder: Directory to save page images
        progress_callback: Optional callback(current, total) for progress
        
    Returns:
        List of paths to saved page images
    """
    page_paths = []
    
    # Get filename for folder naming
    if hasattr(pdf_file, 'name'):
        pdf_filename = pdf_file.name
    else:
        pdf_filename = "uploaded_pdf"
    
    # Create output directory
    pdf_output_dir = os.path.join(output_folder, os.path.splitext(pdf_filename)[0])
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    try:
        # Read PDF content
        if hasattr(pdf_file, 'read'):
            pdf_content = pdf_file.read()
            # Reset file pointer if possible
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
        else:
            with open(pdf_file, 'rb') as f:
                pdf_content = f.read()
        
        # Open PDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        total_pages = len(doc)
        
        for i, page in enumerate(doc.pages()):
            page_num = i + 1
            page_img_path = os.path.join(pdf_output_dir, f"page_{page_num}.png")
            
            # Render page to image
            pix = page.get_pixmap(dpi=150)
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil_image.save(page_img_path, "PNG")
            
            page_paths.append(page_img_path)
            
            if progress_callback:
                progress_callback(page_num, total_pages)
        
        doc.close()
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise
    
    return page_paths


# -----------------------------------------------------------------------------
# Sample Images (from original)
# -----------------------------------------------------------------------------

SAMPLE_IMAGES = {
    "tesla.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbef936e6-3efa-43b3-88d7-7ec620cdb33b_2744x1539.png",
    "netflix.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F23bd84c9-5b62-4526-b467-3088e27e4193_2744x1539.png",
    "nike.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5cd33ba-ae1a-42a8-a254-d85e690d9870_2741x1541.png",
    "google.png": "https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F395dd3b9-b38e-4d1f-91bc-d37b642ee920_2741x1541.png",
    "accenture.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08b2227c-7dc8-49f7-b3c5-13cab5443ba6_2741x1541.png",
    "tencent.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ec8448c-c4d1-4aab-a8e9-2ddebe0c95fd_2741x1541.png",
}


def download_sample_images(
    output_folder: str = "img",
    progress_callback: Optional[callable] = None,
) -> list[str]:
    """Download sample images for demo.
    
    Args:
        output_folder: Directory to save images
        progress_callback: Optional callback(current, total) for progress
        
    Returns:
        List of paths to downloaded images
    """
    import requests
    
    os.makedirs(output_folder, exist_ok=True)
    paths = []
    total = len(SAMPLE_IMAGES)
    
    for i, (name, url) in enumerate(SAMPLE_IMAGES.items()):
        img_path = os.path.join(output_folder, name)
        paths.append(img_path)
        
        if not os.path.exists(img_path):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                paths.pop()  # Remove failed path
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return paths
