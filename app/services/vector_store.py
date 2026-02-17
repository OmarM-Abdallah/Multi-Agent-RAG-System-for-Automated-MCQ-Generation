"""
Vector store service using ChromaDB for document storage and retrieval.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector database operations using ChromaDB."""
    
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=settings.vector_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize sentence-transformers for free local embeddings
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"description": "PDF document chunks with embeddings"}
        )
        
        logger.info(f"Initialized vector store with collection: {settings.collection_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using sentence-transformers (free, local).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Generate embedding locally (no API call!)
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def add_documents(
        self, 
        chunks: List[Dict[str, Any]], 
        filename: str
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of text chunks with metadata
            filename: Source filename
            
        Returns:
            Number of chunks added
        """
        try:
            # Clear existing documents for this file
            self.delete_by_filename(filename)
            
            documents = []
            embeddings = []
            ids = []
            metadatas = []
            
            for chunk in chunks:
                chunk_text = chunk["text"]
                chunk_id = chunk["chunk_id"]
                
                # Generate embedding
                embedding = self.generate_embedding(chunk_text)
                
                # Prepare data
                doc_id = f"{filename}_{chunk_id}"
                documents.append(chunk_text)
                embeddings.append(embedding)
                ids.append(doc_id)
                metadatas.append({
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "char_count": chunk["char_count"]
                })
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return len(chunks)
        
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filename: Optional filter by filename
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        try:
            if top_k is None:
                top_k = settings.retrieval_top_k
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Build where clause for filtering
            where = {"filename": filename} if filename else None
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def delete_by_filename(self, filename: str) -> None:
        """
        Delete all chunks for a specific filename.
        
        Args:
            filename: Filename to delete
        """
        try:
            # Get all IDs for this filename
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for {filename}")
        
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            # Don't raise - this is cleanup, not critical
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": settings.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_chunks": 0, "collection_name": settings.collection_name}
    
    def reset_collection(self) -> None:
        """Reset the entire collection (useful for testing)."""
        try:
            self.client.delete_collection(name=settings.collection_name)
            self.collection = self.client.create_collection(
                name=settings.collection_name,
                metadata={"description": "PDF document chunks with embeddings"}
            )
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
