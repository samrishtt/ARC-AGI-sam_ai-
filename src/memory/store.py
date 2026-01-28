try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from typing import List, Dict, Any
import uuid
import os

class MockMemoryModule:
    """Fallback memory when ChromaDB is unavailable."""
    def __init__(self, persistence_path: str = "db"):
        self.storage = []
        print("Warning: Using In-Memory Mock Storage (ChromaDB not found)")

    def store(self, content: str, metadata: Dict[str, Any] = None):
        self.storage.append({"content": content, "metadata": metadata})

    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        # Simple keyword search fallback
        results = [
            item["content"] for item in self.storage 
            if any(word in item["content"] for word in query.split())
        ]
        return results[:n_results]

    def clear(self):
        self.storage = []

if HAS_CHROMA:
    class MemoryModule:
        def __init__(self, persistence_path: str = "db"):
            self.client = chromadb.PersistentClient(path=persistence_path)
            self.collection = self.client.get_or_create_collection(name="arc_insights")

        def store(self, content: str, metadata: Dict[str, Any] = None):
            self.collection.add(
                documents=[content],
                metadatas=[metadata] if metadata else None,
                ids=[str(uuid.uuid4())]
            )

        def retrieve(self, query: str, n_results: int = 3) -> List[str]:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                return results['documents'][0] if results['documents'] else []
            except:
                return []

        def clear(self):
            try:
                self.client.delete_collection("arc_insights")
                self.collection = self.client.create_collection(name="arc_insights")
            except:
                pass
else:
    MemoryModule = MockMemoryModule
