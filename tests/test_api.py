"""
Basic tests for the RAG Question Generator API.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_stats_endpoint():
    """Test the stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "collection_name" in data


def test_generate_questions_without_ingestion():
    """Test question generation without ingested documents."""
    response = client.post(
        "/generate/questions",
        json={
            "query": "test query",
            "num_questions": 3
        }
    )
    # Should fail because no documents ingested
    assert response.status_code == 400
    assert "No documents in vector store" in response.json()["detail"]


def test_ingest_invalid_file_type():
    """Test ingesting non-PDF file."""
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"test content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_question_generation_validation():
    """Test request validation for question generation."""
    # Test with invalid num_questions
    response = client.post(
        "/generate/questions",
        json={
            "query": "test",
            "num_questions": 0
        }
    )
    assert response.status_code == 422  # Validation error
    
    # Test with num_questions too high
    response = client.post(
        "/generate/questions",
        json={
            "query": "test",
            "num_questions": 25
        }
    )
    assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
