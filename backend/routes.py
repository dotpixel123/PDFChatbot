"""
FastAPI routes for RAG system
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import os
import tempfile
from typing import AsyncGenerator

from backend.schemas import (
    ChatRequest, HealthResponse, UploadResponse, DocumentListResponse,
    DeleteResponse, RebuildIndexResponse, DocumentMetadata
)
from backend.document_manager import DocumentManager
from generation.pipeline import rag_pipeline
from generation.generator import llm

router = APIRouter()
doc_manager = DocumentManager(upload_dir="uploads")


# ==================== Chat & Streaming ====================

async def answer_generator(query: str, conversation_history: list = None) -> AsyncGenerator[str, None]:
    """
    Generate streaming answer from RAG pipeline.
    Yields tokens as they are generated.
    """
    try:
        # Run RAG pipeline to get answer
        answer, chunks = rag_pipeline(query)
        
        # Stream the answer in chunks (simulate token streaming)
        # Split into sentences for more natural streaming
        sentences = answer.split('. ')
        for i, sentence in enumerate(sentences):
            chunk = sentence + ('. ' if i < len(sentences) - 1 else '')
            yield chunk
            
    except Exception as e:
        yield f"ERROR: {str(e)}"


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Streams answer via Server-Sent Events.
    
    Args:
        request: ChatRequest with query and optional conversation_history
    
    Returns:
        StreamingResponse with answer chunks
    """
    try:
        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        return StreamingResponse(
            answer_generator(request.query, request.conversation_history),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Document Management ====================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to the knowledge base.
    
    Args:
        file: PDF file to upload
    
    Returns:
        UploadResponse with document name and chunks created
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Upload through document manager
            doc_name, chunks_created = doc_manager.upload_document(
                tmp_path,
                len(content)
            )
            
            return UploadResponse(
                status="success",
                document_name=doc_name,
                chunks_created=chunks_created
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents.
    
    Returns:
        DocumentListResponse with list of documents
    """
    try:
        documents = doc_manager.list_documents()
        doc_metadata = [
            DocumentMetadata(
                name=doc["name"],
                size=doc["size"],
                uploaded_at=doc["uploaded_at"]
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            documents=doc_metadata,
            total_documents=len(documents)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_name}", response_model=DeleteResponse)
async def delete_document(document_name: str):
    """
    Delete a document from the knowledge base.
    
    Args:
        document_name: Name of the document to delete
    
    Returns:
        DeleteResponse with number of chunks removed
    """
    try:
        chunks_removed = doc_manager.delete_document(document_name)
        
        return DeleteResponse(
            status="success",
            document_name=document_name,
            chunks_removed=chunks_removed
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Index Management ====================

@router.post("/rebuild-index", response_model=RebuildIndexResponse)
async def rebuild_index():
    """
    Rebuild vector index from all uploaded documents.
    
    Returns:
        RebuildIndexResponse with total chunks and documents
    """
    try:
        # This would need to be implemented in document_manager
        # For now, return stats
        total_chunks = doc_manager.total_chunks()
        total_docs = doc_manager.total_documents()
        
        return RebuildIndexResponse(
            status="success",
            total_chunks=total_chunks,
            total_documents=total_docs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Health & Status ====================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint. Verifies backend is running.
    
    Returns:
        HealthResponse with status and model info
    """
    try:
        return HealthResponse(
            status="ok",
            model="gemini-2.5-flash",
            vector_db="chroma_db"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
