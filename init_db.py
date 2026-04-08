"""
Initialize the database and required directories on first run.
"""

import os
from pathlib import Path
import json

def initialize_system():
    """Create all necessary directories and files for the system to run."""
    
    # Define paths
    chroma_db = Path("chroma_db")
    uploads_dir = Path("uploads")
    registry_file = "document_registry.json"
    
    # Create directories
    chroma_db.mkdir(exist_ok=True)
    uploads_dir.mkdir(exist_ok=True)
    
    # Create empty registry if it doesn't exist
    if not os.path.exists(registry_file):
        with open(registry_file, 'w') as f:
            json.dump({}, f, indent=2)
        print(f"✓ Created {registry_file}")
    
    print("✓ Created chroma_db/ directory")
    print("✓ Created uploads/ directory")
    print("\n✅ System initialized successfully!")

if __name__ == "__main__":
    initialize_system()
