"""
Pydantic models for request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class IngestResponse(BaseModel):
    """Response model for PDF ingestion."""
    message: str
    filename: str
    total_chunks: int
    table_of_contents: List[Dict[str, Any]]


class QuestionOption(BaseModel):
    """Model for a single MCQ option."""
    option_id: str = Field(..., description="Option identifier (A, B, C, D)")
    text: str = Field(..., description="Option text")


class MCQQuestion(BaseModel):
    """Model for a multiple choice question."""
    question_id: int = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="The question text")
    options: List[QuestionOption] = Field(..., description="List of answer options")
    correct_answer: str = Field(..., description="Correct option ID (A, B, C, D)")
    explanation: str = Field(..., description="Explanation of the correct answer")
    topic: str = Field(..., description="Topic or concept covered")
    difficulty: str = Field(..., description="Difficulty level: easy, medium, hard")
    evaluation_score: float = Field(..., ge=0, le=10, description="Quality score from evaluator (0-10)")
    evaluation_feedback: str = Field(..., description="Feedback from evaluator agent")


class QuestionGenerationRequest(BaseModel):
    """Request model for question generation."""
    query: str = Field(..., description="Concept or topic to generate questions about")
    num_questions: Optional[int] = Field(default=5, ge=1, le=20, description="Number of questions to generate")


class QuestionGenerationResponse(BaseModel):
    """Response model for question generation."""
    query: str
    num_questions_generated: int
    questions: List[MCQQuestion]
    retrieval_context: List[str] = Field(..., description="Retrieved document chunks used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    app_name: str
    version: str
