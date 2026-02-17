"""
Main FastAPI application for RAG-based Question Generation.
"""
import logging
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
import uvicorn

from app.config import settings
from app.models.schemas import (
    IngestResponse,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    HealthResponse,
    MCQQuestion
)
from app.services.pdf_processor import PDFProcessor
from app.services.vector_store import VectorStore
from app.services.rag_pipeline import RAGPipeline
from app.agents.generator import QuestionGeneratorAgent
from app.agents.evaluator import QuestionEvaluatorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-agent system for generating MCQ questions from PDF documents using RAG"
)

# Initialize services and agents (lazy loading)
pdf_processor: Optional[PDFProcessor] = None
vector_store: Optional[VectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None
generator_agent: Optional[QuestionGeneratorAgent] = None
evaluator_agent: Optional[QuestionEvaluatorAgent] = None


def get_services():
    """Initialize services if not already initialized."""
    global pdf_processor, vector_store, rag_pipeline, generator_agent, evaluator_agent
    
    if pdf_processor is None:
        logger.info("Initializing services...")
        pdf_processor = PDFProcessor()
        vector_store = VectorStore()
        rag_pipeline = RAGPipeline(vector_store)
        generator_agent = QuestionGeneratorAgent()
        evaluator_agent = QuestionEvaluatorAgent()
        logger.info("Services initialized successfully")
    
    return pdf_processor, vector_store, rag_pipeline, generator_agent, evaluator_agent


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs(settings.vector_db_path, exist_ok=True)
    
    # Initialize services
    get_services()
    
    logger.info("Application startup complete")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    
    This endpoint:
    1. Accepts a PDF file upload
    2. Extracts text and structure
    3. Chunks the content
    4. Generates embeddings
    5. Stores in vector database
    6. Returns Table of Contents
    
    Args:
        file: PDF file to ingest
        
    Returns:
        IngestResponse with ToC and processing stats
    """
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Get services
        processor, store, _, _, _ = get_services()
        
        # Save uploaded file temporarily
        file_path = Path("./data") / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved file to {file_path}")
        
        # Extract table of contents
        logger.info("Extracting Table of Contents...")
        toc = processor.extract_table_of_contents(str(file_path))
        
        # Extract and process text
        logger.info("Extracting text from PDF...")
        text = processor.extract_text_from_pdf(str(file_path))
        
        if not text or len(text.strip()) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF appears to be empty or contains insufficient text"
            )
        
        # Chunk the text
        logger.info("Chunking text...")
        chunks = processor.chunk_text(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        # Add to vector store
        logger.info("Adding chunks to vector store...")
        num_chunks = store.add_documents(chunks, file.filename)
        
        # Clean up temporary file
        file_path.unlink()
        
        logger.info(f"Successfully ingested {file.filename}: {num_chunks} chunks")
        
        return IngestResponse(
            message="PDF ingested successfully",
            filename=file.filename,
            total_chunks=num_chunks,
            table_of_contents=toc
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing PDF: {str(e)}"
        )


@app.post("/generate/questions", response_model=QuestionGenerationResponse)
async def generate_questions(request: QuestionGenerationRequest):
    """
    Generate MCQ questions based on a query/concept.
    
    This endpoint orchestrates the multi-agent workflow:
    1. Retrieves relevant context using RAG
    2. Generator agent creates questions
    3. Evaluator agent reviews and scores questions
    4. Returns approved high-quality questions
    
    Args:
        request: Query and parameters for question generation
        
    Returns:
        QuestionGenerationResponse with generated questions
    """
    try:
        logger.info(f"Generating questions for query: {request.query}")
        
        # Get services
        _, store, pipeline, generator, evaluator = get_services()
        
        # Check if vector store has documents
        stats = store.get_collection_stats()
        if stats["total_chunks"] == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents in vector store. Please ingest a PDF first using /ingest endpoint."
            )
        
        # Step 1: Retrieve relevant context
        logger.info("Step 1: Retrieving relevant context...")
        retrieval_result = pipeline.retrieve_context(
            query=request.query,
            top_k=settings.retrieval_top_k
        )
        
        if not retrieval_result["chunks"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No relevant content found for query: {request.query}"
            )
        
        formatted_context = retrieval_result["formatted_context"]
        chunks = retrieval_result["chunks"]
        
        logger.info(f"Retrieved {len(chunks)} relevant chunks")
        
        # Step 2: Generate questions using Generator Agent
        logger.info("Step 2: Generating questions with Generator Agent...")
        generation_prompt = pipeline.prepare_generation_prompt(
            query=request.query,
            context=formatted_context,
            num_questions=request.num_questions
        )
        
        generated_questions = generator.generate_questions(
            prompt=generation_prompt,
            num_questions=request.num_questions
        )
        
        if not generated_questions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate questions. Please try again."
            )
        
        logger.info(f"Generated {len(generated_questions)} questions")
        
        # Step 3: Evaluate questions using Evaluator Agent
        logger.info("Step 3: Evaluating questions with Evaluator Agent...")
        evaluation_prompt = pipeline.prepare_evaluation_prompt(
            questions=generated_questions,
            context=formatted_context
        )
        
        evaluations = evaluator.evaluate_questions(
            questions=generated_questions,
            prompt=evaluation_prompt
        )
        
        # Step 4: Filter approved questions
        logger.info("Step 4: Filtering approved questions...")
        approved_questions = evaluator.filter_approved_questions(
            questions=generated_questions,
            evaluations=evaluations
        )
        
        if not approved_questions:
            # If no questions approved, return best ones anyway
            logger.warning("No questions met approval threshold, returning all with scores")
            approved_questions = []
            for i, (q, e) in enumerate(zip(generated_questions, evaluations)):
                approved_questions.append({
                    "question_id": i,
                    **q,
                    "evaluation_score": e.get("score", 0),
                    "evaluation_feedback": e.get("feedback", "")
                })
        
        # Convert to response model
        mcq_questions = []
        for q in approved_questions:
            mcq_questions.append(MCQQuestion(**q))
        
        logger.info(f"Returning {len(mcq_questions)} approved questions")
        
        return QuestionGenerationResponse(
            query=request.query,
            num_questions_generated=len(mcq_questions),
            questions=mcq_questions,
            retrieval_context=chunks
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating questions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating questions: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """Get statistics about the vector store."""
    try:
        _, store, _, _, _ = get_services()
        stats = store.get_collection_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
