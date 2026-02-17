"""
RAG (Retrieval Augmented Generation) pipeline for question generation.
"""
from typing import List, Dict, Any
import logging
from app.services.vector_store import VectorStore
from app.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Coordinates retrieval and context preparation for LLM."""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: Vector store instance for retrieval
        """
        self.vector_store = vector_store
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query or concept
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with retrieved chunks and formatted context
        """
        try:
            # Retrieve relevant chunks
            results = self.vector_store.search(query, top_k=top_k)
            
            if not results:
                logger.warning(f"No results found for query: {query}")
                return {
                    "chunks": [],
                    "formatted_context": "",
                    "sources": []
                }
            
            # Format context for LLM
            formatted_context = self._format_context(results)
            
            # Extract source information
            sources = [
                {
                    "chunk_id": r["metadata"]["chunk_id"],
                    "filename": r["metadata"]["filename"],
                    "similarity": r["similarity_score"]
                }
                for r in results
            ]
            
            logger.info(f"Retrieved {len(results)} chunks for query")
            
            return {
                "chunks": [r["text"] for r in results],
                "formatted_context": formatted_context,
                "sources": sources,
                "num_chunks": len(results)
            }
        
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            raise
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            results: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            text = result["text"]
            similarity = result["similarity_score"]
            
            context_parts.append(
                f"[Context {i}] (Relevance: {similarity:.2f})\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def prepare_generation_prompt(
        self, 
        query: str, 
        context: str,
        num_questions: int
    ) -> str:
        """
        Prepare the prompt for question generation.
        
        Args:
            query: User query/concept
            context: Retrieved context
            num_questions: Number of questions to generate
            
        Returns:
            Formatted prompt for the generator agent
        """
        prompt = f"""You are an expert educational content creator specializing in generating high-quality Multiple Choice Questions (MCQs).

Based on the following context retrieved from educational materials, generate {num_questions} multiple choice questions about: "{query}"

CONTEXT:
{context}

REQUIREMENTS:
1. Generate exactly {num_questions} questions
2. Each question must:
   - Have a clear, unambiguous question text
   - Have exactly 4 options (A, B, C, D)
   - Have only ONE correct answer
   - Include a detailed explanation of why the answer is correct
   - Be directly based on the provided context
   - Cover important concepts from the material
3. Vary the difficulty: mix of easy, medium, and hard questions
4. Focus on understanding, not just memorization

OUTPUT FORMAT (JSON):
{{
  "questions": [
    {{
      "question_text": "...",
      "options": [
        {{"option_id": "A", "text": "..."}},
        {{"option_id": "B", "text": "..."}},
        {{"option_id": "C", "text": "..."}},
        {{"option_id": "D", "text": "..."}}
      ],
      "correct_answer": "A",
      "explanation": "...",
      "topic": "...",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Generate the questions now:"""
        
        return prompt
    
    def prepare_evaluation_prompt(
        self, 
        questions: List[Dict[str, Any]], 
        context: str
    ) -> str:
        """
        Prepare prompt for question evaluation.
        
        Args:
            questions: Generated questions to evaluate
            context: Original context used for generation
            
        Returns:
            Formatted prompt for the evaluator agent
        """
        import json
        
        questions_json = json.dumps(questions, indent=2)
        
        prompt = f"""You are an expert educational content evaluator. Your task is to critically evaluate the quality of multiple choice questions.

ORIGINAL CONTEXT:
{context}

GENERATED QUESTIONS:
{questions_json}

EVALUATION CRITERIA:
For each question, evaluate based on:
1. **Relevance** (0-2): Is it based on the context? Does it match the topic?
2. **Clarity** (0-2): Is the question clear and unambiguous?
3. **Quality of Options** (0-2): Are distractors plausible? Is there only one correct answer?
4. **Explanation Quality** (0-2): Is the explanation clear and accurate?
5. **Educational Value** (0-2): Does it test understanding vs. memorization?

SCORING: Sum the above for a total score out of 10 for each question.

OUTPUT FORMAT (JSON):
{{
  "evaluations": [
    {{
      "question_id": 0,
      "score": 8.5,
      "feedback": "Detailed feedback on strengths and weaknesses",
      "relevance": 2.0,
      "clarity": 1.5,
      "quality_of_options": 2.0,
      "explanation_quality": 1.5,
      "educational_value": 1.5,
      "approved": true,
      "suggested_improvements": "Optional suggestions"
    }}
  ]
}}

Only approve questions with score >= 7.0. Provide constructive feedback for all questions.

Evaluate now:"""
        
        return prompt
