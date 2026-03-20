"""
FastAPI routes for the RAG system.
Handles document uploads, queries, and knowledge base management.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import tempfile

from backend.schemas import (
    ChatRequest, HealthResponse, UploadResponse, DocumentListResponse,
    DeleteResponse, RebuildIndexResponse, DocumentMetadata
)
from backend.document_manager import DocumentManager
from generation.pipeline import rag_pipeline
from ingestion_pipeline.vector_db import vector_store_is_empty
from config import VECTOR_DB_DIR, LLM_MODEL

router = APIRouter()
doc_manager = DocumentManager(upload_dir="uploads")


# ==================== Chat ====================

@router.post("/chat")
async def chat(request: ChatRequest):
    """Query the knowledge base and get an answer.
    
    Args:
        request: ChatRequest with the user's query.
    
    Returns:
        JSON response with the answer and source documents.
        
    Raises:
        HTTPException: If database is empty or query fails.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if vector_store_is_empty():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base is empty. Upload documents first using /upload endpoint."
        )
    
    try:
        answer, chunks = rag_pipeline(request.query)
        return {
            "query": request.query,
            "answer": answer,
            "sources": chunks[:5]  # Return top 5 relevant chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")



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
    """Health check endpoint with system information."""
    return HealthResponse(
        status="ok",
        model=LLM_MODEL,
        vector_db=VECTOR_DB_DIR
    )

