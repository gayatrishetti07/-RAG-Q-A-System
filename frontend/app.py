"""
frontend/app.py
---------------
Streamlit-based chat UI for the RAG Q&A system.

Features:
- PDF file upload (single or multiple)
- Chat interface with message history
- Source document display (expandable)
- System status indicator
- Conversation reset button
- Responsive layout with sidebar controls

Run with:
    streamlit run frontend/app.py
"""

import os
import time
import requests
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main chat container */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .main-header h1 {
        color: #e0e0ff;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #9090bb;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    /* Status badges */
    .status-ready {
        background: #1a4a2e;
        color: #4ade80;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #4ade80;
        display: inline-block;
    }

    .status-not-ready {
        background: #4a1a1a;
        color: #f87171;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #f87171;
        display: inline-block;
    }

    /* Source cards */
    .source-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed rgba(99, 102, 241, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background: rgba(99, 102, 241, 0.03);
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_status() -> dict:
    """Fetch system status from the backend API."""
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        return {"index_exists": False, "pipeline_ready": False,
                "llm_provider": "unknown", "embedding_provider": "unknown",
                "_connection_error": True}
    except Exception:
        return {"index_exists": False, "pipeline_ready": False,
                "llm_provider": "unknown", "embedding_provider": "unknown"}


def upload_pdfs(files) -> dict:
    """Upload PDF files to the backend."""
    file_tuples = [
        ("files", (f.name, f.getvalue(), "application/pdf"))
        for f in files
    ]
    response = requests.post(
        f"{BACKEND_URL}/upload",
        files=file_tuples,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def ask_question(question: str) -> dict:
    """Send a question to the RAG backend."""
    response = requests.post(
        f"{BACKEND_URL}/ask",
        json={"question": question},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def reset_conversation() -> bool:
    """Reset conversation memory on the backend."""
    try:
        response = requests.post(f"{BACKEND_URL}/reset", timeout=10)
        response.raise_for_status()
        return True
    except Exception:
        return False


def format_source(source: dict, index: int) -> str:
    """Format a source document for display."""
    file_name = source.get("file_name", "Unknown")
    page = source.get("page", "?")
    content = source.get("content", "")
    return f"📄 **{file_name}** — Page {page}\n\n_{content}_"


# ── Session State ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 RAG Q&A System")
    st.markdown("---")

    # ── System Status ──────────────────────────────────────────────────────────
    st.markdown("### 🔌 System Status")
    status = get_status()

    if status.get("_connection_error"):
        st.error("⚠️ Cannot connect to backend. Is it running?\n\n```\nuvicorn backend.main:app --reload\n```")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if status.get("pipeline_ready"):
                st.markdown('<div class="status-ready">✅ Ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-not-ready">⭕ Not Ready</div>', unsafe_allow_html=True)
        with col2:
            if status.get("index_exists"):
                st.caption("📦 Index: Found")
            else:
                st.caption("📦 Index: Empty")

        st.caption(f"🤖 LLM: `{status.get('llm_provider', 'N/A')}`")
        st.caption(f"🔢 Embeddings: `{status.get('embedding_provider', 'N/A')}`")

    st.markdown("---")

    # ── PDF Upload ─────────────────────────────────────────────────────────────
    st.markdown("### 📄 Upload Documents")
    st.caption("Upload one or more PDF files to index.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDFs to build the knowledge base.",
        label_visibility="collapsed",
    )

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        st.caption(f"Selected: {', '.join(file_names)}")

        if st.button("📥 Index Documents", use_container_width=True, type="primary"):
            with st.spinner("🔄 Uploading and indexing... This may take a moment."):
                try:
                    result = upload_pdfs(uploaded_files)
                    st.session_state.indexed_files.extend(result.get("file_names", []))
                    st.success(
                        f"✅ Indexed **{result['files_processed']}** PDF(s) → "
                        f"**{result['chunks_indexed']}** chunks"
                    )
                    # Refresh status
                    time.sleep(0.5)
                    st.rerun()
                except requests.HTTPError as e:
                    st.error(f"Upload failed: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

    # ── Indexed Files ──────────────────────────────────────────────────────────
    if st.session_state.indexed_files:
        st.markdown("---")
        st.markdown("### 📚 Indexed Files")
        for fname in set(st.session_state.indexed_files):
            st.caption(f"✅ {fname}")

    st.markdown("---")

    # ── Conversation Controls ──────────────────────────────────────────────────
    st.markdown("### 💬 Conversation")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        if reset_conversation():
            st.session_state.messages = []
            st.success("Chat history cleared.")
            st.rerun()
        else:
            st.warning("Could not clear backend memory.")

    # ── Settings ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Info")
    st.caption(f"Backend: `{BACKEND_URL}`")
    st.caption("Powered by LangChain + FAISS")
    st.caption("[📖 API Docs](http://localhost:8000/docs)")


# ── Main Area ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧠 RAG Q&A System</h1>
    <p>Ask questions about your documents — powered by Retrieval-Augmented Generation</p>
</div>
""", unsafe_allow_html=True)

# ── Getting Started Banner ─────────────────────────────────────────────────────
if not status.get("index_exists") and not status.get("_connection_error"):
    st.info(
        "👈 **Getting Started:** Upload PDF files in the sidebar, click **Index Documents**, "
        "then come back here to ask questions!",
        icon="🚀"
    )

# ── Chat History ──────────────────────────────────────────────────────────────

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

        # Show sources if available (assistant messages only)
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} Source(s) Used", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(format_source(source, i))
                    if i < len(message["sources"]):
                        st.divider()

# ── Chat Input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=not status.get("pipeline_ready") and status.get("index_exists"),
):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": [],
    })

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # ── Generate Response ──────────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                if not status.get("index_exists"):
                    answer = "⚠️ No documents indexed yet. Please upload PDFs via the sidebar first."
                    sources = []
                else:
                    result = ask_question(prompt)
                    answer = result.get("answer", "No answer generated.")
                    sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander(f"📎 {len(sources)} Source(s) Used", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(format_source(source, i))
                            if i < len(sources):
                                st.divider()

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.HTTPError as e:
                error_detail = e.response.json().get("detail", str(e))
                error_msg = f"❌ API Error: {error_detail}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })

            except requests.ConnectionError:
                error_msg = "❌ Cannot connect to backend. Make sure the FastAPI server is running."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })

            except Exception as e:
                error_msg = f"❌ Unexpected error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption(
        "Built with ❤️ using LangChain · FAISS · OpenAI · FastAPI · Streamlit"
    )
