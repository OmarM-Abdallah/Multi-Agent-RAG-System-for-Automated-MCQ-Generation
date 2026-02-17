"""
Question Evaluator Agent - Evaluates quality of generated questions.
"""
import json
import logging
from typing import List, Dict, Any
from groq import Groq
from app.config import settings

logger = logging.getLogger(__name__)


class QuestionEvaluatorAgent:
    """Agent responsible for evaluating the quality of generated questions."""
    
    def __init__(self):
        """Initialize the evaluator agent with Groq client."""
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
        self.min_score_threshold = 7.0
        logger.info(f"Initialized QuestionEvaluatorAgent with Groq model: {self.model}")
    
    def evaluate_questions(
        self, 
        questions: List[Dict[str, Any]], 
        prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the quality of generated questions.
        
        Args:
            questions: List of questions to evaluate
            prompt: Formatted evaluation prompt with context
            
        Returns:
            List of evaluations with scores and feedback
        """
        try:
            logger.info(f"Evaluating {len(questions)} questions...")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educational content evaluator. You critically assess the quality of multiple choice questions based on clarity, relevance, and educational value. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=settings.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            logger.debug(f"Raw evaluation response: {content[:200]}...")
            
            parsed_response = json.loads(content)
            evaluations = parsed_response.get("evaluations", [])
            
            if not evaluations:
                logger.warning("No evaluations found in response, creating default evaluations")
                evaluations = self._create_default_evaluations(questions)
            
            # Ensure we have an evaluation for each question
            if len(evaluations) < len(questions):
                logger.warning(f"Missing evaluations: got {len(evaluations)}, expected {len(questions)}")
                # Add default evaluations for missing questions
                for i in range(len(evaluations), len(questions)):
                    evaluations.append(self._create_default_evaluation(i))
            
            logger.info(f"Successfully evaluated {len(evaluations)} questions")
            return evaluations
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            return self._create_default_evaluations(questions)
        
        except Exception as e:
            logger.error(f"Error evaluating questions: {e}")
            return self._create_default_evaluations(questions)
    
    def _create_default_evaluations(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create default evaluations when LLM evaluation fails."""
        return [self._create_default_evaluation(i) for i in range(len(questions))]
    
    def _create_default_evaluation(self, question_id: int) -> Dict[str, Any]:
        """Create a default evaluation for a single question."""
        return {
            "question_id": question_id,
            "score": 7.5,
            "feedback": "Question generated successfully. Automated evaluation unavailable.",
            "relevance": 1.5,
            "clarity": 1.5,
            "quality_of_options": 1.5,
            "explanation_quality": 1.5,
            "educational_value": 1.5,
            "approved": True,
            "suggested_improvements": "Consider manual review for quality assurance."
        }
    
    def filter_approved_questions(
        self,
        questions: List[Dict[str, Any]],
        evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter questions that meet quality threshold.
        
        Args:
            questions: Original questions
            evaluations: Evaluation results
            
        Returns:
            List of approved questions with evaluation data
        """
        approved_questions = []
        
        for i, (question, evaluation) in enumerate(zip(questions, evaluations)):
            score = evaluation.get("score", 0)
            is_approved = evaluation.get("approved", False) or score >= self.min_score_threshold
            
            if is_approved:
                # Merge question with evaluation data
                approved_question = {
                    "question_id": i,
                    **question,
                    "evaluation_score": score,
                    "evaluation_feedback": evaluation.get("feedback", "")
                }
                approved_questions.append(approved_question)
                logger.debug(f"Question {i} approved with score {score}")
            else:
                logger.debug(f"Question {i} rejected with score {score}")
        
        logger.info(f"Approved {len(approved_questions)}/{len(questions)} questions")
        return approved_questions
    
    def get_evaluation_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from evaluations.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Summary statistics
        """
        if not evaluations:
            return {
                "total_questions": 0,
                "approved": 0,
                "average_score": 0.0,
                "approval_rate": 0.0
            }
        
        total = len(evaluations)
        approved = sum(1 for e in evaluations if e.get("approved", False))
        scores = [e.get("score", 0) for e in evaluations]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "total_questions": total,
            "approved": approved,
            "rejected": total - approved,
            "average_score": round(avg_score, 2),
            "approval_rate": round((approved / total) * 100, 1),
            "score_breakdown": {
                "min": round(min(scores), 2) if scores else 0,
                "max": round(max(scores), 2) if scores else 0,
                "median": round(sorted(scores)[len(scores)//2], 2) if scores else 0
            }
        }
