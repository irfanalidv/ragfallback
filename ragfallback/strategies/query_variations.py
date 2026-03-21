"""Query rewriting strategy — broadens retrieval by generating phrasing alternatives."""

import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage

from ragfallback.strategies.base import FallbackStrategy


class QueryVariationsStrategy(FallbackStrategy):
    """Rewrites the original query into N variations to broaden retrieval recall.

    When the user's phrasing doesn't match the document vocabulary, a single
    vector search misses. This strategy generates semantically equivalent rewrites
    so the retriever has more surface area to hit.
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
        """Return queries to try for this attempt.

        On attempt 1, prepends the original query when include_original is set.
        On later attempts, returns LLM-generated rewrites only.
        """
        queries = []

        if self.include_original and attempt == 1:
            queries.append(original_query)

        if self.num_variations > 0:
            variations = self._generate_variations(
                original_query=original_query,
                context=context,
                num_variations=self.num_variations,
                llm=llm
            )
            queries.extend(variations)
        
        self.logger.debug("generated %d query variations", len(queries))
        return queries
    
    def _generate_variations(
        self,
        original_query: str,
        context: Dict[str, Any],
        num_variations: int,
        llm: BaseLanguageModel
    ) -> List[str]:
        """Call the LLM and parse its JSON array of rewritten queries."""
        context_str = ""
        if context:
            context_str = f"\n\nContext: {json.dumps(context, indent=2)}"

        prompt = self.prompt_template.format(
            query=original_query,
            num_variations=num_variations,
            context=context_str
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)

            variations = json.loads(response_text)
            if isinstance(variations, list):
                return variations[:num_variations]
            else:
                self.logger.warning("expected list from LLM, got %s", type(variations).__name__)
                return []
        except json.JSONDecodeError as e:
            self.logger.warning("LLM response was not valid JSON, falling back to text parse: %s", e)
            return self._extract_variations_from_text(response_text)
        except Exception as e:
            self.logger.warning("query variation generation failed: %s", e)
            return []

    def _extract_variations_from_text(self, text: str) -> List[str]:
        """Extract variations from text if JSON parsing fails."""
        import re
        array_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
        if array_match:
            try:
                variations = json.loads(f"[{array_match.group(1)}]")
                return variations if isinstance(variations, list) else []
            except (json.JSONDecodeError, ValueError):
                pass

        lines = [line.strip().strip('"').strip("'") for line in text.split('\n')]
        return [line for line in lines if line and len(line) > 10][:self.num_variations]

