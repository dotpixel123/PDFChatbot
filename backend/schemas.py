"""
Pydantic schemas for FastAPI request/response models
"""

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for /chat endpoint"""
    query: str
    conversation_history: Optional[List[dict]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is meditation?",
                "conversation_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help?"}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    model: str
    vector_db: str


class UploadResponse(BaseModel):
    """Response model for /upload endpoint"""
    status: str
    document_name: str
    chunks_created: int


class DocumentMetadata(BaseModel):
    """Metadata for a single document"""
    name: str
    size: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    """Response model for /documents endpoint"""
    documents: List[DocumentMetadata]
    total_documents: int


class DeleteResponse(BaseModel):
    """Response model for /documents/{name} DELETE endpoint"""
    status: str
    document_name: str
    chunks_removed: int


class RebuildIndexResponse(BaseModel):
    """Response model for /rebuild-index endpoint"""
    status: str
    total_chunks: int
    total_documents: int


class ErrorResponse(BaseModel):
    """Generic error response"""
    error: str
    details: Optional[str] = None
