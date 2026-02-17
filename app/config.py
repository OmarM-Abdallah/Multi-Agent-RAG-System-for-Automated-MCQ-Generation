"""
Configuration settings for the RAG Question Generator application.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    groq_api_key: str
    
    # LLM Provider
    llm_provider: str = "groq"  # groq or openai
    
    # Application Settings
    app_name: str = "RAG Question Generator"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Vector Database Settings
    vector_db_path: str = "./vectordb"
    collection_name: str = "documents"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # RAG Settings
    retrieval_top_k: int = 5
    
    # Embedding Settings (using sentence-transformers for free embeddings)
    embedding_model: str = "all-MiniLM-L6-v2"  # Free local embeddings
    
    # LLM Settings for Groq
    llm_model: str = "llama-3.1-70b-versatile"  # Groq's best free model
    llm_temperature: float = 0.7
    max_tokens: int = 2000
    
    # Question Generation Settings
    num_questions_per_request: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
