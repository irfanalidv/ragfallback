"""Answer confidence scoring."""

from typing import List, Dict, Any, Optional
import logging

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage


class ConfidenceScorer:
    """
    Score answer confidence using LLM or embedding similarity.
    """

    DEFAULT_PROMPT_TEMPLATE = """Rate the confidence that this answer correctly addresses the question.

Question: {question}

Answer: {answer}

Documents Used: {num_documents} documents

Rate confidence on a scale of 0.0 to 1.0, where:
- 1.0 = Answer is definitely correct and fully addresses the question
- 0.7 = Answer is likely correct but may have minor gaps
- 0.5 = Answer is partially correct but incomplete
- 0.3 = Answer is uncertain or only partially relevant
- 0.0 = Answer is incorrect or not found

Return ONLY a number between 0.0 and 1.0, no explanation."""

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        method: str = "llm",
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize ConfidenceScorer.

        Args:
            llm: Language model for LLM-based scoring
            method: Scoring method ("llm" or "embedding_similarity")
            prompt_template: Custom prompt template
        """
        self.llm = llm
        self.method = method
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.logger = logging.getLogger(__name__)

    def score(
        self,
        question: str,
        answer: str,
        documents: List,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Score answer confidence.

        Args:
            question: Original question
            answer: Generated answer
            documents: Retrieved documents
            context: Optional context

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if self.method == "llm" and self.llm:
            return self._score_with_llm(question, answer, documents)
        elif self.method == "embedding_similarity":
            return self._score_with_embedding(question, answer, documents)
        else:
            # Default: simple heuristic
            return self._score_heuristic(question, answer, documents)

    def _score_with_llm(self, question: str, answer: str, documents: List) -> float:
        """Score using LLM."""
        if not self.llm:
            return self._score_heuristic(question, answer, documents)

        prompt = self.prompt_template.format(
            question=question, answer=answer, num_documents=len(documents)
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Extract number from response
            import re

            numbers = re.findall(r"\d+\.?\d*", response_text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]

            return 0.5  # Default if parsing fails
        except Exception as e:
            self.logger.error(f"Error scoring with LLM: {e}")
            return self._score_heuristic(question, answer, documents)

    def _score_with_embedding(
        self, question: str, answer: str, documents: List
    ) -> float:
        """
        Score using embedding similarity.

        Note: Currently falls back to heuristic scoring.
        Future enhancement: Calculate cosine similarity between question/answer
        embeddings and document embeddings for more accurate confidence scoring.
        """
        # Future enhancement: Implement embedding-based scoring
        # This would involve:
        # 1. Generate embeddings for question, answer, and documents
        # 2. Calculate cosine similarity between answer and documents
        # 3. Weight by document relevance to question
        return self._score_heuristic(question, answer, documents)

    def _score_heuristic(self, question: str, answer: str, documents: List) -> float:
        """Simple heuristic-based scoring."""
        # Check for failure indicators
        failure_indicators = ["not found", "n/a", "x", "unknown", "unable to"]
        answer_lower = answer.lower()

        if any(indicator in answer_lower for indicator in failure_indicators):
            return 0.0

        # Base score on document count and answer length
        doc_score = min(len(documents) / 5.0, 1.0)  # More docs = higher score
        length_score = min(len(answer) / 50.0, 1.0)  # Longer answer = higher score

        # Combine scores
        confidence = doc_score * 0.6 + length_score * 0.4
        return max(0.0, min(1.0, confidence))
