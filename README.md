# 🧠 RAG Q&A System
### Production-Grade Retrieval-Augmented Generation for Document Intelligence

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green?logo=chainlink)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?logo=streamlit)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What This Project Does

This is a **production-ready Retrieval-Augmented Generation (RAG) system** that lets you upload any PDF document and ask natural language questions about it. The system retrieves the most relevant sections and uses an LLM to generate accurate, context-grounded answers — with full conversation memory.

> **"Think of it as ChatGPT, but trained on YOUR documents."**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG Q&A SYSTEM                               │
│                                                                     │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │   Streamlit  │    │              FastAPI Backend             │   │
│  │   Frontend   │◄──►│                                         │   │
│  │              │    │  POST /upload   POST /ask   GET /history│   │
│  └──────────────┘    └─────────────┬───────────────────────────┘   │
│                                    │                                │
│         ┌──────────────────────────▼───────────────────────┐       │
│         │                  RAG Pipeline                     │       │
│         │                                                   │       │
│  ┌──────▼──────┐   ┌──────────────┐   ┌─────────────────┐  │       │
│  │  PDF Loader │──►│ Text Chunker │──►│ Embedding Model │  │       │
│  │ (PyPDF /    │   │ (Recursive   │   │ (OpenAI ada-002 │  │       │
│  │  PyMuPDF)   │   │  Character)  │   │  / HuggingFace) │  │       │
│  └─────────────┘   └──────────────┘   └────────┬────────┘  │       │
│                                                 │           │       │
│                    ┌────────────────────────────▼─────┐     │       │
│                    │         FAISS Vector Store        │     │       │
│                    │   (Persisted to disk as index)    │     │       │
│                    └────────────────────┬─────────────┘     │       │
│                                         │                   │       │
│  User Query ──► Embed Query ────────────▼──────────────┐    │       │
│                                  Top-K Retrieval        │    │       │
│                                  (Similarity Search)    │    │       │
│                                         │               │    │       │
│  ┌──────────────────────────────────────▼─────────────┐ │    │       │
│  │               Prompt Template                       │ │    │       │
│  │  System + Context Chunks + Chat History + Question  │◄┘    │       │
│  └──────────────────────────┬──────────────────────────┘      │       │
│                              │                                 │       │
│  ┌───────────────────────────▼──────────────────────────┐      │       │
│  │                    LLM (GPT-3.5 / GPT-4)              │      │       │
│  │         Generates grounded, cited answer              │      │       │
│  └───────────────────────────┬──────────────────────────┘      │       │
│                              │                                  │       │
│                     ┌────────▼────────┐                         │       │
│                     │  Conversation   │                         │       │
│                     │     Memory      │                         │       │
│                     └─────────────────┘                         │       │
│                                                                  │       │
└──────────────────────────────────────────────────────────────────┘       │
```

### RAG Pipeline — Step by Step

```
PDF(s)
  │
  ▼
[1. LOAD]        PyPDFLoader → List of Documents (one per page)
  │
  ▼
[2. CHUNK]       RecursiveCharacterTextSplitter → Chunks (1000 chars, 200 overlap)
  │
  ▼
[3. EMBED]       OpenAI text-embedding-ada-002 → 1536-dim vectors per chunk
  │
  ▼
[4. INDEX]       FAISS.from_documents() → Persisted vector index on disk
  │
  ▼ (At query time)
  │
[5. RETRIEVE]    Query → embed → FAISS similarity search → Top-4 chunks
  │
  ▼
[6. AUGMENT]     Prompt = System + Context + Chat History + User Question
  │
  ▼
[7. GENERATE]    LLM(prompt) → Grounded answer with citations
  │
  ▼
[8. RESPOND]     Answer + Sources → FastAPI → Streamlit → User
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **Multi-PDF Support** | Upload and query across multiple PDFs simultaneously |
| 🔍 **Semantic Search** | FAISS approximate nearest-neighbor search over dense embeddings |
| 🧠 **Conversational Memory** | Multi-turn context with `ConversationBufferMemory` |
| 🤖 **Dual LLM Support** | OpenAI GPT-3.5/4 or HuggingFace Flan-T5 (free, offline) |
| 🔢 **Dual Embedding Support** | OpenAI ada-002 or sentence-transformers (offline) |
| ⚡ **FastAPI Backend** | RESTful API with Pydantic validation and OpenAPI docs |
| 🎨 **Streamlit UI** | Chat interface with source citation display |
| 🐳 **Docker Ready** | Fully containerized with docker-compose |
| 📝 **Structured Logging** | Loguru with file rotation and colored output |
| 🔒 **Environment Config** | All secrets via `.env` — nothing hardcoded |

---

## 📁 Project Structure

```
rag-qna-system/
│
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — routes, middleware, lifecycle
│   ├── rag_pipeline.py      # ConversationalRetrievalChain + LLM factory
│   └── models.py            # Pydantic request/response schemas
│
├── frontend/
│   └── app.py               # Streamlit chat UI
│
├── utils/
│   ├── __init__.py
│   ├── loader.py            # PDF loading (PyPDF + PyMuPDF fallback)
│   ├── splitter.py          # Chunking strategies (recursive / token)
│   ├── embeddings.py        # Embedding model factory (OpenAI / HF)
│   └── vector_store.py      # FAISS build, load, merge, retriever
│
├── data/                    # Drop your PDFs here
│   └── .gitkeep
│
├── faiss_index/             # Auto-created after first indexing
│
├── logs/
│   └── app.log              # Rotating log file
│
├── test_pipeline.py         # End-to-end integration test script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example             # Template — copy to .env
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (or use HuggingFace fallback — free)

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/rag-qna-system.git
cd rag-qna-system

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
nano .env
```

### 3. Run the Backend (FastAPI)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/docs** for the interactive API explorer.

### 4. Run the Frontend (Streamlit)

In a second terminal:

```bash
streamlit run frontend/app.py
```

Visit **http://localhost:8501**

### 5. Test the Pipeline (Optional)

```bash
# Place any PDF in data/ folder, then:
python test_pipeline.py --pdf data/your_file.pdf --query "What is this about?"
```

---

## 🐳 Docker Deployment

### Run with Docker Compose (Recommended)

```bash
# Build and start both services
docker-compose up --build

# Backend:  http://localhost:8000
# Frontend: http://localhost:8501

# Stop
docker-compose down
```

### Build Images Individually

```bash
# Backend only
docker build -t rag-backend .
docker run -p 8000:8000 --env-file .env rag-backend

# Frontend only
docker build -t rag-frontend .
docker run -p 8501:8501 --env-file .env \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  rag-frontend streamlit run frontend/app.py --server.port=8501
```

---

## ☁️ Cloud Deployment

### Option A: Render (Easiest — Free Tier Available)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. **Backend service:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Add all `.env` variables in Render's dashboard
5. **Frontend service:**
   - Start Command: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`
   - Set `BACKEND_URL` to your backend Render URL

### Option B: AWS EC2

```bash
# SSH into your EC2 instance (Ubuntu)
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update && sudo apt install -y python3-pip python3-venv git
git clone https://github.com/yourusername/rag-qna-system.git
cd rag-qna-system

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
cp .env.example .env && nano .env

# Run with process manager (keeps alive after SSH disconnect)
pip install supervisor
# Or use tmux/screen:
tmux new -s backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Ctrl+B D to detach

tmux new -s frontend
streamlit run frontend/app.py --server.port 8501
```

### Option C: Using Docker on EC2

```bash
# Install Docker on EC2
sudo apt install -y docker.io docker-compose
sudo systemctl start docker

# Run
git clone your-repo && cd rag-qna-system
cp .env.example .env && nano .env
sudo docker-compose up -d

# Open ports 8000 and 8501 in EC2 Security Group
```

---

## 🔧 Configuration Options

All configuration is via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `LLM_PROVIDER` | `openai` | `openai` or `huggingface` |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Any OpenAI chat model |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `huggingface` |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Chunks retrieved per query |
| `FAISS_INDEX_PATH` | `faiss_index` | Where to persist the index |
| `BACKEND_HOST` | `0.0.0.0` | FastAPI host |
| `BACKEND_PORT` | `8000` | FastAPI port |

---

## 🆓 Using Free (No API Key) Mode

Set these in `.env` to run 100% locally for free:

```env
LLM_PROVIDER=huggingface
HUGGINGFACE_MODEL=google/flan-t5-large

EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

> Note: Quality is lower than OpenAI, but works completely offline.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/status` | System status (index, pipeline, providers) |
| `POST` | `/upload` | Upload and index PDF files |
| `POST` | `/ask` | Ask a question (RAG query) |
| `GET` | `/history` | Get conversation history |
| `POST` | `/reset` | Clear conversation memory |
| `DELETE` | `/index` | Delete FAISS index |

Full interactive docs at: **http://localhost:8000/docs**

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings of this report?"}'
```

Response:
```json
{
  "question": "What are the key findings of this report?",
  "answer": "The report identifies three key findings: ...",
  "sources": [
    {
      "content": "The analysis reveals that...",
      "file_name": "annual_report.pdf",
      "page": "12"
    }
  ]
}
```

---

## 🧪 Example Queries to Test

After uploading a PDF, try these types of questions:

```
# Factual
"What is the main topic of this document?"
"Who are the key people mentioned?"
"What dates are referenced in the document?"

# Summary
"Summarize the key points of chapter 2."
"What are the main conclusions?"

# Analytical
"What evidence supports the author's argument?"
"Compare the approaches described in sections 1 and 2."

# Follow-up (tests memory)
"Tell me more about that last point."
"Can you elaborate on what you just explained?"
"Give me a concrete example of this."
```

---

## 🧩 How RAG Works — Key Concepts

### Why RAG over Fine-tuning?

| | RAG | Fine-tuning |
|---|---|---|
| **Update knowledge** | Upload new PDF ✅ | Retrain model ❌ |
| **Cost** | API calls only ✅ | GPU training hours ❌ |
| **Transparency** | Shows sources ✅ | Black box ❌ |
| **Hallucination** | Grounded in docs ✅ | Can hallucinate ❌ |
| **Speed** | Ready in minutes ✅ | Hours/days ❌ |

### The Chunking Strategy

```
Chunk size = 1000 chars (~250 tokens)
Overlap    = 200 chars  (prevents context loss at boundaries)

Document: [────────────────────────────────────────────]
Chunk 1:  [──────────1000──────────]
Chunk 2:            [──200──+──────1000──────]
Chunk 3:                           [──200──+──────1000──────]
```

### Similarity Search Explained

```
User Query: "What are the revenue figures?"
Query Embedding: [0.23, -0.45, 0.78, ...]  (1536 dimensions)

FAISS computes cosine similarity against all chunk embeddings:
  Chunk 42: similarity = 0.94 ← Top match!  "Q3 revenue was $4.2M..."
  Chunk 17: similarity = 0.89               "Annual revenue growth..."
  Chunk 88: similarity = 0.82               "Revenue breakdown by..."
  Chunk 31: similarity = 0.79               "Sales forecast shows..."
```

---

## 📊 Performance Notes

- **Indexing speed:** ~100 pages/minute (OpenAI embeddings)
- **Query latency:** 1–3 seconds (retrieval + GPT-3.5)
- **FAISS index size:** ~6KB per chunk (1536-dim float32)
- **Memory usage:** ~500MB RAM for a 500-page document

---

## 🔮 Potential Enhancements

- [ ] **Hybrid Search:** BM25 + dense retrieval (keyword + semantic)
- [ ] **Re-ranking:** Cross-encoder model to rerank retrieved chunks
- [ ] **Multi-modal:** Support for images, tables within PDFs
- [ ] **Streaming:** Server-sent events for real-time answer streaming
- [ ] **Authentication:** JWT-based user auth and per-user indexes
- [ ] **Evaluation:** RAGAS metrics for retrieval and answer quality
- [ ] **Graph RAG:** Knowledge graph integration for relationship queries
- [ ] **Web UI Upgrade:** Next.js frontend with better UX

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | LangChain 0.2 | RAG chain, memory, prompt templates |
| **LLM** | OpenAI GPT-3.5/4 | Answer generation |
| **LLM (fallback)** | HuggingFace Flan-T5 | Free local inference |
| **Embeddings** | OpenAI ada-002 | Semantic vector generation |
| **Embeddings (fallback)** | sentence-transformers | Free local embeddings |
| **Vector DB** | FAISS | Fast similarity search |
| **PDF Parsing** | PyPDF + PyMuPDF | Robust text extraction |
| **Backend** | FastAPI + Uvicorn | REST API, async, OpenAPI |
| **Frontend** | Streamlit | Chat UI |
| **Logging** | Loguru | Structured, rotating logs |
| **Containerization** | Docker + Compose | Reproducible deployment |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/graph-rag`)
3. Commit changes (`git commit -m 'Add graph RAG support'`)
4. Push and open a PR

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Your Name**
- GitHub: https://github.com/gayatrishetti07
- LinkedIn: www.linkedin.com/in/gayatri-shetti-0208b4255

---

*Built as a portfolio project demonstrating end-to-end GenAI engineering — from PDF parsing to production deployment.*
