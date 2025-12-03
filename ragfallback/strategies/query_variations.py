"""Query Variations Strategy - LLM-based query rewriting."""

import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage

from ragfallback.strategies.base import FallbackStrategy


class QueryVariationsStrategy(FallbackStrategy):
    """
    Generate query variations using LLM.
    
    This strategy uses an LLM to generate alternative formulations
    of the original query, increasing the chances of finding relevant documents.
    """
    
    DEFAULT_PROMPT_TEMPLATE = """Generate {num_variations} alternative ways to ask this question that might find the answer in documentation.

Original question: "{query}"
{context}

Requirements:
- Use different terminology or phrasing
- Be more specific or more general as needed
- Focus on key concepts from the original question

Return ONLY a JSON array of strings: ["variation 1", "variation 2", ...]
Do not include any explanation or markdown formatting."""
    
    def __init__(
        self,
        num_variations: int = 2,
        include_original: bool = True,
        variation_prompt_template: Optional[str] = None
    ):
        """
        Initialize QueryVariationsStrategy.
        
        Args:
            num_variations: Number of variations to generate
            include_original: Include original query in results
            variation_prompt_template: Custom prompt template
        """
        self.num_variations = num_variations
        self.include_original = include_original
        self.prompt_template = variation_prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.logger = logging.getLogger(__name__)
    
    def generate_queries(
        self,
        original_query: str,
        context: Dict[str, Any],
        attempt: int,
        llm: BaseLanguageModel
    ) -> List[str]:
        """
        Generate query variations.
        
        Args:
            original_query: Original query string
            context: Context dictionary
            attempt: Current attempt number
            llm: Language model for generation
            
        Returns:
            List of query strings
        """
        queries = []
        
        # Include original query first
        if self.include_original and attempt == 1:
            queries.append(original_query)
        
        # Generate variations
        if self.num_variations > 0:
            variations = self._generate_variations(
                original_query=original_query,
                context=context,
                num_variations=self.num_variations,
                llm=llm
            )
            queries.extend(variations)
        
        self.logger.info(f"Generated {len(queries)} query variations")
        return queries
    
    def _generate_variations(
        self,
        original_query: str,
        context: Dict[str, Any],
        num_variations: int,
        llm: BaseLanguageModel
    ) -> List[str]:
        """Generate query variations using LLM."""
        # Build context string
        context_str = ""
        if context:
            context_str = f"\n\nContext: {json.dumps(context, indent=2)}"
        
        # Build prompt
        prompt = self.prompt_template.format(
            query=original_query,
            num_variations=num_variations,
            context=context_str
        )
        
        try:
            # Generate variations
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON array
            variations = json.loads(response_text)
            if isinstance(variations, list):
                return variations[:num_variations]
            else:
                self.logger.warning(f"Expected list, got {type(variations)}")
                return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON from LLM response: {e}")
            # Try to extract array from markdown or text
            return self._extract_variations_from_text(response_text)
        except Exception as e:
            self.logger.error(f"Error generating query variations: {e}")
            return []
    
    def _extract_variations_from_text(self, text: str) -> List[str]:
        """Extract variations from text if JSON parsing fails."""
        import re
        # Try to find array-like patterns
        array_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
        if array_match:
            try:
                variations = json.loads(f"[{array_match.group(1)}]")
                return variations if isinstance(variations, list) else []
            except:
                pass
        
        # Fallback: split by lines and clean
        lines = [line.strip().strip('"').strip("'") for line in text.split('\n')]
        return [line for line in lines if line and len(line) > 10][:self.num_variations]

