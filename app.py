"""Streamlit UI for LangGraph Vision RAG Agent.

Features:
- API key configuration
- Model selection (OpenRouter)
- PDF and image upload
- Human-in-the-loop review
- Self-correction visualization
- LangSmith trace links
"""
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    CHROMA_COLLECTION_NAME,
)
from src.state import VisionRAGState, GraphConfig, RetrievedImage
from src.tools import (
    get_cohere_client,
    get_chromadb_client,
    get_or_create_collection,
    embed_images,
    add_images_to_chromadb,
    get_collection_count,
    clear_collection,
    download_sample_images,
    process_pdf,
)
from src.graph import create_vision_rag_graph, create_graph_visualization

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Vision RAG Agent",
    page_icon="🔍",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .relevance-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 500;
    }
    .relevant { background-color: #d4edda; color: #155724; }
    .not-relevant { background-color: #f8d7da; color: #721c24; }
    .iteration-badge {
        background-color: #e2e3e5;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "rag_state" not in st.session_state:
    st.session_state.rag_state = None
if "graph_running" not in st.session_state:
    st.session_state.graph_running = False
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False


# --- Header ---
st.title("🔍 Vision RAG Agent")
st.markdown("""
**LangGraph-powered** visual RAG with self-correction and human-in-the-loop.  
Upload images or PDFs, ask questions, and watch the agent reason about relevance.
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # Load defaults from environment
    env_openrouter = os.getenv("OPENROUTER_API_KEY", "")
    env_cohere = os.getenv("COHERE_API_KEY", "")
    env_langsmith = os.getenv("LANGCHAIN_API_KEY", "")
    
    # Show status if keys loaded from .env
    if env_openrouter or env_cohere:
        st.success("✅ Keys loaded from .env")
    
    # OpenRouter API Key
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        value=env_openrouter,
        type="password",
        key="openrouter_key",
        help="Get your key at openrouter.ai/keys",
    )
    
    # Cohere API Key
    cohere_key = st.text_input(
        "Cohere API Key",
        value=env_cohere,
        type="password",
        key="cohere_key",
        help="Get your key at dashboard.cohere.com/api-keys",
    )
    
    # LangSmith API Key (optional)
    langsmith_key = st.text_input(
        "LangSmith API Key (optional)",
        value=env_langsmith,
        type="password",
        key="langsmith_key",
        help="For tracing - get at smith.langchain.com",
    )
    
    # Set environment variables for LangSmith
    if langsmith_key:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "vision-rag-agent"
    
    st.markdown("---")
    
    # Model Selection
    st.header("🤖 Model Selection")
    
    model_options = {f"{m['name']} ({m['cost']})": k for k, m in AVAILABLE_MODELS.items()}
    selected_model_display = st.selectbox(
        "Vision Model",
        options=list(model_options.keys()),
        index=0,  # Default to first (Gemini 2.5 Flash)
    )
    selected_model_key = model_options[selected_model_display]
    selected_model_id = AVAILABLE_MODELS[selected_model_key]["id"]
    
    st.markdown("---")
    
    # Agent Settings
    st.header("⚙️ Agent Settings")
    
    human_in_loop = st.toggle(
        "Human-in-the-Loop",
        value=False,
        help="Pause to approve/reject retrieved images before answering",
    )
    
    max_retries = st.slider(
        "Max Self-Correction Retries",
        min_value=1,
        max_value=5,
        value=2,
        help="How many times to try finding a relevant image",
    )
    
    top_k = st.slider(
        "Images to Retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of candidate images to retrieve for each query",
    )
    
    st.markdown("---")
    
    # Collection Management
    st.header("🗄️ Vector Store")
    
    try:
        if cohere_key:
            client = get_chromadb_client()
            collection = get_or_create_collection(client)
            count = get_collection_count(collection)
            st.metric("Images Indexed", count)
            
            if st.button("🗑️ Clear Index", type="secondary"):
                clear_collection(collection)
                st.success("Index cleared!")
                st.rerun()
    except Exception as e:
        st.warning(f"ChromaDB: {e}")
    
    st.markdown("---")
    
    # Architecture Diagram
    with st.expander("📊 Agent Architecture"):
        st.markdown(create_graph_visualization())


# --- Validation ---
api_ready = bool(openrouter_key and cohere_key)

if not api_ready:
    st.warning("⚠️ Please enter your OpenRouter and Cohere API keys in the sidebar to continue.")
    st.stop()


# --- Main Content ---
col1, col2 = st.columns([1, 1])

# --- Left Column: Data Loading ---
with col1:
    st.header("📁 Load Data")
    
    tab1, tab2 = st.tabs(["📤 Upload Files", "📊 Sample Images"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload images or PDFs",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        
        if uploaded_files and st.button("🔄 Process & Index", key="process_btn"):
            with st.status("Processing files...", expanded=True) as status:
                cohere_client = get_cohere_client(cohere_key)
                chroma_client = get_chromadb_client()
                collection = get_or_create_collection(chroma_client)
                
                all_paths = []
                
                # Create directories
                os.makedirs("uploaded_img", exist_ok=True)
                os.makedirs("pdf_pages", exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    st.write(f"Processing: {uploaded_file.name}")
                    
                    if uploaded_file.type == "application/pdf":
                        # Process PDF
                        progress = st.progress(0)
                        paths = process_pdf(
                            uploaded_file,
                            progress_callback=lambda c, t: progress.progress(c / t),
                        )
                        all_paths.extend(paths)
                        progress.empty()
                    else:
                        # Save image
                        img_path = os.path.join("uploaded_img", uploaded_file.name)
                        with open(img_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        all_paths.append(img_path)
                
                if all_paths:
                    st.write(f"Embedding {len(all_paths)} images...")
                    progress = st.progress(0)
                    embeddings = embed_images(
                        all_paths,
                        cohere_client,
                        progress_callback=lambda c, t: progress.progress(c / t),
                    )
                    progress.empty()
                    
                    st.write("Adding to ChromaDB...")
                    add_images_to_chromadb(collection, all_paths, embeddings)
                    
                    status.update(label=f"✅ Indexed {len(all_paths)} images!", state="complete")
                else:
                    status.update(label="❌ No files processed", state="error")
    
    with tab2:
        st.markdown("Load sample financial charts for demo:")
        
        if st.button("📥 Download & Index Samples", key="sample_btn"):
            with st.status("Loading sample images...", expanded=True) as status:
                cohere_client = get_cohere_client(cohere_key)
                chroma_client = get_chromadb_client()
                collection = get_or_create_collection(chroma_client)
                
                st.write("Downloading images...")
                progress = st.progress(0)
                paths = download_sample_images(
                    progress_callback=lambda c, t: progress.progress(c / t),
                )
                progress.empty()
                
                if paths:
                    st.write(f"Embedding {len(paths)} images...")
                    progress = st.progress(0)
                    embeddings = embed_images(
                        paths,
                        cohere_client,
                        progress_callback=lambda c, t: progress.progress(c / t),
                    )
                    progress.empty()
                    
                    st.write("Adding to ChromaDB...")
                    add_images_to_chromadb(collection, paths, embeddings)
                    
                    status.update(label=f"✅ Indexed {len(paths)} sample images!", state="complete")
                else:
                    status.update(label="❌ Failed to download samples", state="error")


# --- Right Column: Query & Results ---
with col2:
    st.header("❓ Ask a Question")
    
    # Check if we have indexed images
    try:
        chroma_client = get_chromadb_client()
        collection = get_or_create_collection(chroma_client)
        image_count = get_collection_count(collection)
    except:
        image_count = 0
    
    if image_count == 0:
        st.info("👈 Load some images first, then ask questions about them.")
    else:
        # Query Input
        query = st.text_input(
            "Your question:",
            placeholder="e.g., What is Nike's net profit?",
            key="query_input",
        )
        
        # Initialize session state for human-in-the-loop
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = None
        if "pending_review" not in st.session_state:
            st.session_state.pending_review = False
        if "graph_instance" not in st.session_state:
            st.session_state.graph_instance = None
        if "current_config" not in st.session_state:
            st.session_state.current_config = None
        
        run_button = st.button("🚀 Run Vision RAG", type="primary", disabled=not query)
        
        if run_button and query:
            import uuid
            
            # Create new thread for this query
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.pending_review = False
            st.session_state.rag_state = None
            
            # Create config
            config = GraphConfig(
                model_id=selected_model_id,
                human_in_loop=human_in_loop,
                top_k=top_k,
                openrouter_api_key=openrouter_key,
                cohere_api_key=cohere_key,
            )
            st.session_state.current_config = config
            
            # Create initial state
            initial_state = VisionRAGState(
                query=query,
                human_in_loop_enabled=human_in_loop,
                max_iterations=max_retries,
            )
            
            # Create graph (with checkpointer for human-in-the-loop)
            graph = create_vision_rag_graph(config, use_checkpointer=human_in_loop)
            st.session_state.graph_instance = graph
            
            with st.status("🔍 Running Vision RAG Agent...", expanded=True) as status:
                st.write("📝 Embedding query and searching...")
                
                try:
                    # Run the graph with thread config
                    thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    # Stream through the graph - will stop at interrupt
                    events = list(graph.stream(initial_state.model_dump(), config=thread_config))
                    
                    # Get current state after streaming
                    current_state = graph.get_state(thread_config)
                    
                    # Check if we're at an interrupt (human_review)
                    if current_state.next and "human_review" in current_state.next:
                        # We're paused before human_review
                        st.session_state.pending_review = True
                        st.session_state.rag_state = VisionRAGState(**current_state.values)
                        status.update(label="⏸️ Awaiting Human Review", state="running")
                        st.rerun()
                    else:
                        # Graph completed normally
                        final_state = VisionRAGState(**current_state.values)
                        st.session_state.rag_state = final_state
                        st.session_state.pending_review = False
                        
                        if final_state.iteration > 1:
                            st.write(f"🔄 Self-correction iterations: {final_state.iteration}")
                        
                        status.update(label="✅ Complete!", state="complete")
                        
                except Exception as e:
                    st.error(f"Error running graph: {e}")
                    status.update(label="❌ Error", state="error")
        
        # Handle pending human review
        if st.session_state.pending_review and st.session_state.rag_state:
            state = st.session_state.rag_state
            
            st.markdown("---")
            st.subheader("🔍 Human Review Required")
            
            if state.retrieved_images and state.current_image_idx < len(state.retrieved_images):
                current_img = state.retrieved_images[state.current_image_idx]
                
                if os.path.exists(current_img.path):
                    st.image(
                        current_img.path,
                        caption=f"Retrieved: {os.path.basename(current_img.path)} (Score: {current_img.score:.2%})",
                        use_container_width=True,
                    )
                
                col_approve, col_reject = st.columns(2)
                
                with col_approve:
                    if st.button("✅ Approve", type="primary", use_container_width=True):
                        # Resume graph with approval
                        graph = st.session_state.graph_instance
                        thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        
                        # Update state with approval and resume
                        graph.update_state(thread_config, {"human_approved": True})
                        
                        with st.status("🔄 Continuing...", expanded=True):
                            events = list(graph.stream(None, config=thread_config))
                            current_state = graph.get_state(thread_config)
                            
                            if current_state.next and "human_review" in current_state.next:
                                st.session_state.rag_state = VisionRAGState(**current_state.values)
                            else:
                                final_state = VisionRAGState(**current_state.values)
                                st.session_state.rag_state = final_state
                                st.session_state.pending_review = False
                        
                        st.rerun()
                
                with col_reject:
                    if st.button("❌ Reject (Try Next)", use_container_width=True):
                        # Resume graph with rejection
                        graph = st.session_state.graph_instance
                        thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        
                        # Update state with rejection and resume
                        graph.update_state(thread_config, {"human_approved": False})
                        
                        with st.status("🔄 Trying next image...", expanded=True):
                            events = list(graph.stream(None, config=thread_config))
                            current_state = graph.get_state(thread_config)
                            
                            if current_state.next and "human_review" in current_state.next:
                                st.session_state.rag_state = VisionRAGState(**current_state.values)
                            else:
                                final_state = VisionRAGState(**current_state.values)
                                st.session_state.rag_state = final_state
                                st.session_state.pending_review = False
                        
                        st.rerun()
        
        # Show Results (when not pending review)
        if st.session_state.rag_state and not st.session_state.pending_review:
            state = st.session_state.rag_state
            
            st.markdown("---")
            st.subheader("📊 Results")
            
            # Show retrieved image
            if state.retrieved_images and state.current_image_idx < len(state.retrieved_images):
                current_img = state.retrieved_images[state.current_image_idx]
                
                col_img, col_meta = st.columns([2, 1])
                
                with col_img:
                    if os.path.exists(current_img.path):
                        st.image(
                            current_img.path,
                            caption=f"Retrieved: {os.path.basename(current_img.path)}",
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Image not found: {current_img.path}")
                
                with col_meta:
                    st.markdown("**Retrieval Info**")
                    st.write(f"Similarity: {current_img.score:.2%}")
                    
                    if state.is_relevant:
                        st.success("✅ Relevant")
                    else:
                        st.error("❌ Not Relevant")
                    
                    st.write(f"Iterations: {state.iteration}/{state.max_iterations}")
                    
                    if state.relevance_reasoning:
                        with st.expander("Reasoning"):
                            st.write(state.relevance_reasoning)
            
            # Show Answer
            if state.answer:
                st.markdown("### 💬 Answer")
                st.markdown(state.answer)
            
            # Show Error if any
            if state.error:
                st.error(f"Error: {state.error}")
            
            # LangSmith link
            if langsmith_key:
                st.markdown("---")
                st.markdown("🔗 [View trace in LangSmith](https://smith.langchain.com)")


# --- Footer ---
st.markdown("---")
st.caption(
    "Built with [LangGraph](https://github.com/langchain-ai/langgraph) • "
    "[Cohere Embed-4](https://cohere.com) • "
    "[OpenRouter](https://openrouter.ai) • "
    "[ChromaDB](https://www.trychroma.com)"
)
