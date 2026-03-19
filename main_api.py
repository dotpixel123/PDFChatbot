"""
Main FastAPI application for RAG system backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Backend API for Retrieval-Augmented Generation system",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origin in production: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, tags=["rag"])


@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "RAG System API",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
