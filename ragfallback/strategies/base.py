"""Base class for fallback strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel


class FallbackStrategy(ABC):
    """Base class for all fallback strategies."""
    
    @abstractmethod
    def generate_queries(
        self,
        original_query: str,
        context: Dict[str, Any],
        attempt: int,
        llm: BaseLanguageModel
    ) -> List[str]:
        """
        Generate query variations for fallback.
        
        Args:
            original_query: The original query string
            context: Context dictionary (e.g., {"company": "Acme"})
            attempt: Current attempt number (1-indexed)
            llm: Language model for query generation
            
        Returns:
            List of query strings to try
        """
        pass
    
    def get_name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__

