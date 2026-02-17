"""
Question Generator Agent - Creates MCQ questions from context.
"""
import json
import logging
from typing import List, Dict, Any
from groq import Groq
from app.config import settings

logger = logging.getLogger(__name__)


class QuestionGeneratorAgent:
    """Agent responsible for generating MCQ questions from retrieved context."""
    
    def __init__(self):
        """Initialize the generator agent with Groq client."""
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        logger.info(f"Initialized QuestionGeneratorAgent with Groq model: {self.model}")
    
    def generate_questions(
        self, 
        prompt: str,
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """
        Generate MCQ questions using the LLM.
        
        Args:
            prompt: Formatted prompt with context and instructions
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        try:
            logger.info(f"Generating {num_questions} questions...")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educational content creator. You generate high-quality multiple choice questions based on provided context. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=settings.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            logger.debug(f"Raw LLM response: {content[:200]}...")
            
            # Parse JSON
            parsed_response = json.loads(content)
            
            # Extract questions
            questions = parsed_response.get("questions", [])
            
            if not questions:
                logger.warning("No questions found in LLM response")
                return []
            
            # Validate and format questions
            validated_questions = []
            for i, q in enumerate(questions):
                try:
                    validated_q = self._validate_question(q, i)
                    validated_questions.append(validated_q)
                except Exception as e:
                    logger.warning(f"Invalid question format at index {i}: {e}")
                    continue
            
            logger.info(f"Successfully generated {len(validated_questions)} questions")
            return validated_questions
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {content}")
            raise ValueError("LLM returned invalid JSON")
        
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            raise
    
    def _validate_question(self, question: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Validate and format a single question.
        
        Args:
            question: Raw question data
            index: Question index
            
        Returns:
            Validated and formatted question
        """
        # Required fields
        required_fields = ["question_text", "options", "correct_answer", "explanation"]
        for field in required_fields:
            if field not in question:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate options
        options = question.get("options", [])
        if len(options) != 4:
            raise ValueError(f"Expected 4 options, got {len(options)}")
        
        # Ensure options have correct structure
        formatted_options = []
        expected_ids = ["A", "B", "C", "D"]
        
        for i, opt in enumerate(options):
            if isinstance(opt, dict):
                formatted_options.append({
                    "option_id": opt.get("option_id", expected_ids[i]),
                    "text": opt.get("text", "")
                })
            else:
                # If option is just a string, format it
                formatted_options.append({
                    "option_id": expected_ids[i],
                    "text": str(opt)
                })
        
        # Validate correct answer
        correct_answer = question.get("correct_answer", "").upper()
        if correct_answer not in ["A", "B", "C", "D"]:
            logger.warning(f"Invalid correct_answer: {correct_answer}, defaulting to A")
            correct_answer = "A"
        
        # Build validated question
        validated = {
            "question_text": question["question_text"],
            "options": formatted_options,
            "correct_answer": correct_answer,
            "explanation": question["explanation"],
            "topic": question.get("topic", "General"),
            "difficulty": question.get("difficulty", "medium").lower()
        }
        
        # Ensure difficulty is valid
        if validated["difficulty"] not in ["easy", "medium", "hard"]:
            validated["difficulty"] = "medium"
        
        return validated
    
    def regenerate_question(
        self, 
        question: Dict[str, Any],
        feedback: str
    ) -> Dict[str, Any]:
        """
        Regenerate a single question based on feedback.
        
        Args:
            question: Original question
            feedback: Evaluator feedback
            
        Returns:
            Improved question
        """
        try:
            prompt = f"""Given the following MCQ question and feedback, generate an improved version.

ORIGINAL QUESTION:
{json.dumps(question, indent=2)}

FEEDBACK:
{feedback}

Generate an improved version that addresses the feedback. Return in the same JSON format with fields:
- question_text
- options (array of 4 options with option_id and text)
- correct_answer
- explanation
- topic
- difficulty

Improved question:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at improving educational questions based on feedback. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            improved = json.loads(content)
            
            return self._validate_question(improved, 0)
        
        except Exception as e:
            logger.error(f"Error regenerating question: {e}")
            return question  # Return original if regeneration fails
