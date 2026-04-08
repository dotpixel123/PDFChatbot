"""
Document management for PDF uploads and vector database operations.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

from ingestion_pipeline.pdf_ingest import load_and_chunk_documents
from ingestion_pipeline.vector_db import get_vector_store
from retriever.hybrid_retriever import mark_bm25_dirty
from config import UPLOAD_DIR, REGISTRY_FILE


class DocumentManager:
    """Manages PDF document uploads and metadata."""

    def __init__(self, upload_dir: str = UPLOAD_DIR, registry_file: str = REGISTRY_FILE):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.registry_file = registry_file
        self._load_registry()

    def _load_registry(self):
        """Load or create document registry"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save_registry(self):
        """Save document registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def upload_document(self, file_path: str, file_size: int) -> tuple[str, int]:
        """Upload a PDF document to the vector store.

        Args:
            file_path: Path to the uploaded file
            file_size: Size of the file in bytes

        Returns:
            tuple of (document_name, chunks_created)

        Raises:
            ValueError: If file is not a PDF or processing fails
        """
        file_name = os.path.basename(file_path)

        if not file_name.lower().endswith('.pdf'):
            raise ValueError(f"File {file_name} is not a PDF")

        try:
            # Store file in uploads directory using Path for consistent cross-platform handling
            dest_path = Path(self.upload_dir) / file_name
            
            if os.path.exists(file_path):
                shutil.copy(file_path, str(dest_path))

            # Load and chunk the document - pass as posix path for consistency
            chunks = load_and_chunk_documents(str(dest_path))
            num_chunks = len(chunks)

            if num_chunks == 0:
                raise ValueError(f"No content extracted from {file_name}")

            # Add to vector store
            vector_store = get_vector_store()
            vector_store.add_documents(chunks)
            vector_store.persist()
            mark_bm25_dirty()

            # Update registry
            self.registry[file_name] = {
                "uploaded_at": datetime.now().isoformat(),
                "size": file_size,
                "chunks": num_chunks,
                "path": dest_path.as_posix()
            }
            self._save_registry()

            return file_name, num_chunks

        except Exception as e:
            # Clean up on failure
            try:
                if dest_path.exists():
                    dest_path.unlink()
            except:
                pass
            raise ValueError(f"Failed to process {file_name}: {str(e)}")

    def delete_document(self, document_name: str) -> int:
        """
        Delete a document from the vector store.

        Args:
            document_name: Name of the document to delete

        Returns:
            Number of chunks removed

        Raises:
            FileNotFoundError: If document not found in registry
        """
        if document_name not in self.registry:
            raise FileNotFoundError(f"Document {document_name} not found")

        doc_info = self.registry[document_name]
        chunks_removed = doc_info.get("chunks", 0)

        try:
            # Remove from vector store by collection (requires refactor of vector_db.py)
            # For now, we'll need to rebuild the index
            # This is a placeholder - actual implementation depends on vector_db interface
            
            # Remove file
            if "path" in doc_info and os.path.exists(doc_info["path"]):
                os.remove(doc_info["path"])

            # Remove from registry
            del self.registry[document_name]
            self._save_registry()

            return chunks_removed

        except Exception as e:
            raise ValueError(f"Failed to delete {document_name}: {str(e)}")

    def list_documents(self) -> list[dict]:
        """
        List all uploaded documents.

        Returns:
            List of document metadata dicts
        """
        documents = []
        for name, info in self.registry.items():
            documents.append({
                "name": name,
                "size": info.get("size", 0),
                "uploaded_at": info.get("uploaded_at", ""),
                "chunks": info.get("chunks", 0)
            })
        return documents

    def get_document_info(self, document_name: str) -> dict:
        """Get information about a specific document"""
        if document_name not in self.registry:
            raise FileNotFoundError(f"Document {document_name} not found")
        return self.registry[document_name]

    def total_documents(self) -> int:
        """Return total number of uploaded documents"""
        return len(self.registry)

    def total_chunks(self) -> int:
        """Return total number of chunks across all documents"""
        return sum(doc.get("chunks", 0) for doc in self.registry.values())
