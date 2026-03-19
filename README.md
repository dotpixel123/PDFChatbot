# RAG System

A Retrieval-Augmented Generation (RAG) system for document-based question answering.

## Quick Start

### Prerequisites
- Python 3.12+
- `uv` package manager
- Google API key (set in `.env`)

### Installation

```bash
# Set up environment
export GOOGLE_API_KEY=your_api_key_here
```

### Running the System

```bash
uv run main_api.py
```

Then open your browser and go to:
- **Web Frontend**: http://localhost:8000
- **Swagger API Docs**: http://localhost:8000/docs

## Web Frontend

The system includes a modern, professional web interface for testing:

1. **Upload Documents** - Drag & drop or click to upload PDFs
2. **View Documents** - See all uploaded documents with metadata
3. **Ask Questions** - Query your documents in natural language
4. **View Results** - Get answers with source documents highlighted
5. **Manage Documents** - Delete documents as needed
6. **System Status** - Monitor server health and database status

### Frontend Features

- 📤 **Easy Upload** - Drag-and-drop PDF upload with progress tracking
- ❓ **Smart Queries** - Natural language question answering
- 📚 **Source Citations** - See which documents your answers came from
- 🎯 **Status Monitoring** - Real-time system health and document count
- 🎨 **Modern UI** - Professional, responsive design
- ⚡ **Instant Feedback** - Real-time status updates

## Architecture

### Core Components

- **Ingestion** (`ingestion_pipeline/`)
  - Document loading and chunking
  - Vector database management
  
- **Retrieval** (`retriever/`)
  - Multi-query expansion
  - Hybrid search (vector + BM25)
  - Cross-encoder reranking
  
- **Generation** (`generation/`)
  - LLM answer generation
  - RAG pipeline coordination

- **Backend** (`backend/`)
  - FastAPI routes and schemas
  - Document upload management

- **Frontend** (`frontend/`)
  - Single-page HTML5 application
  - Real-time WebSocket-like updates
  - Responsive design with embedded CSS and JavaScript

### Configuration

Edit `config.py` to adjust:
- Model settings
- Chunk sizes
- Retrieval parameters
- Storage locations

## API Endpoints

### Chat & Queries
- `POST /chat` - Query the knowledge base
- `GET /health` - System health check

### Document Management
- `POST /upload` - Upload a PDF
- `GET /documents` - List all documents
- `DELETE /documents/{name}` - Delete a document
- `POST /rebuild-index` - Rebuild the vector index

### Web Interface
- `GET /` - Serve the web frontend
- `GET /docs` - Interactive API documentation

## Workflow

1. **Start Server** → `uv run main_api.py`
2. **Open Frontend** → http://localhost:8000
3. **Upload PDFs** → Use the upload area (drag & drop supported)
4. **Ask Questions** → Type your question and get instant answers
5. **Manage Documents** → Delete documents as needed

## Development

Code organization:
- Configuration centralized in `config.py`
- Shared LLM instance via `get_llm()`
- Type hints throughout
- Modular, testable functions
- Frontend is fully client-side (no Node.js needed)

## Technical Stack

- **Backend**: FastAPI
- **LLM**: Google Generative AI (Gemini)
- **Embeddings**: Hugging Face (`BAAI/bge-small-en-v1.5`)
- **Reranking**: Cross-encoders (`BAAI/bge-reranker-base`)
- **Vector DB**: Chroma
- **Search**: Hybrid (vector + BM25)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## Notes

- The system starts with an empty knowledge base
- Upload at least one document before asking questions
- The frontend automatically detects when the database is empty
- All timestamps are displayed in local time
- The API automatically handles document chunking and embedding

