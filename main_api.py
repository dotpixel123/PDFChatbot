"""
Main FastAPI application for RAG system backend with web frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, tags=["rag"])

frontend_path = Path(__file__).parent / "frontend" / "index.html"


@app.get("/")
async def root():
    """Serve the web frontend."""
    if frontend_path.exists():
        return FileResponse(frontend_path, media_type="text/html")
    return {
        "message": "RAG System API",
        "docs": "/docs",
        "frontend": "Not available",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
