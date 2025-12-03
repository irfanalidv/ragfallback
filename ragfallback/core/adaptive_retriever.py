"""Adaptive RAG Retriever with fallback strategies."""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import time
import json
import re

from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage, HumanMessage

from ragfallback.strategies.base import FallbackStrategy
from ragfallback.strategies.query_variations import QueryVariationsStrategy
from ragfallback.tracking.cost_tracker import CostTracker
from ragfallback.tracking.metrics import MetricsCollector
from ragfallback.utils.confidence_scorer import ConfidenceScorer


@dataclass
class QueryResult:
    """Result of a RAG query with metadata."""
    
    answer: str
    source: str
    confidence: float
    attempts: int
    cost: float
    intermediate_steps: Optional[List[Dict]] = None
    
    def __repr__(self):
        return (
            f"QueryResult(answer='{self.answer[:50]}...', "
            f"confidence={self.confidence:.2%}, attempts={self.attempts}, "
            f"cost=${self.cost:.4f})"
        )


class AdaptiveRAGRetriever:
    """
    Adaptive RAG Retriever with intelligent fallback strategies.
    
    This retriever attempts to answer queries using multiple strategies,
    falling back to alternative approaches when initial attempts fail
    or yield low-confidence results.
    """
    
    DEFAULT_ANSWER_PROMPT = """You are a helpful assistant that answers questions based on provided documents.

Answer the question based on the documents provided. If the answer is not in the documents, respond with "Not found".

Return your answer in JSON format: {"answer": "...", "source": "..."}"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseLanguageModel,
        embedding_model: Embeddings,
        fallback_strategy: str = "query_variations",
        fallback_strategies: Optional[List[FallbackStrategy]] = None,
        max_attempts: int = 3,
        min_confidence: float = 0.7,
        cost_tracker: Optional[CostTracker] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        enable_logging: bool = True,
        answer_prompt_template: Optional[str] = None
    ):
        """
        Initialize AdaptiveRAGRetriever.
        
        Args:
            vector_store: Vector store instance (Chroma, Qdrant, etc.)
            llm: Language model for query generation and answer synthesis
            embedding_model: Embedding model for semantic search
            fallback_strategy: Name of default fallback strategy
            fallback_strategies: List of custom fallback strategies
            max_attempts: Maximum number of query attempts
            min_confidence: Minimum confidence threshold for success
            cost_tracker: Optional cost tracking instance
            metrics_collector: Optional metrics collection instance
            enable_logging: Enable detailed logging
            answer_prompt_template: Custom prompt template for answer generation
        """
        self.vector_store = vector_store
        self.llm = llm
        self.embedding_model = embedding_model
        self.max_attempts = max_attempts
        self.min_confidence = min_confidence
        self.cost_tracker = cost_tracker or CostTracker()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.answer_prompt_template = answer_prompt_template or self.DEFAULT_ANSWER_PROMPT
        self.logger = logging.getLogger(__name__) if enable_logging else None
        
        # Setup fallback strategies
        if fallback_strategies:
            self.strategies = fallback_strategies
        else:
            if fallback_strategy == "query_variations":
                self.strategies = [QueryVariationsStrategy()]
            else:
                raise ValueError(f"Unknown fallback strategy: {fallback_strategy}")
    
    def query_with_fallback(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        return_intermediate_steps: bool = False,
        enforce_budget: bool = False
    ) -> QueryResult:
        """
        Query with automatic fallback strategies.
        
        Args:
            question: The question to answer
            context: Optional context dictionary (e.g., {"company": "Acme"})
            return_intermediate_steps: Return all intermediate attempts
            enforce_budget: Stop if budget exceeded
            
        Returns:
            QueryResult with answer, source, confidence, and metadata
        """
        context = context or {}
        intermediate_steps = []
        total_cost = 0.0
        start_time = time.time()
        
        # Try each strategy in order
        for strategy_idx, strategy in enumerate(self.strategies):
            if strategy_idx >= self.max_attempts:
                break
            
            # Check budget
            if enforce_budget and self.cost_tracker.budget_exceeded():
                if self.logger:
                    self.logger.warning("Budget exceeded, stopping fallback attempts")
                break
            
            # Generate queries using strategy
            queries = strategy.generate_queries(
                original_query=question,
                context=context,
                attempt=strategy_idx + 1,
                llm=self.llm
            )
            
            # Try each query variation
            for query_idx, query in enumerate(queries):
                attempt_num = strategy_idx * len(queries) + query_idx + 1
                
                if attempt_num > self.max_attempts:
                    break
                
                if self.logger:
                    self.logger.info(f"Attempt {attempt_num}/{self.max_attempts}: {query[:100]}...")
                
                # Retrieve documents
                docs = self._retrieve_documents(query, context)
                
                if not docs:
                    if self.logger:
                        self.logger.warning(f"No documents found for query: {query}")
                    intermediate_steps.append({
                        "attempt": attempt_num,
                        "query": query,
                        "documents": 0,
                        "confidence": 0.0,
                        "cost": 0.0
                    })
                    continue
                
                # Generate answer
                answer, source, confidence, cost = self._generate_answer(
                    question=question,
                    query=query,
                    documents=docs,
                    context=context
                )
                
                total_cost += cost
                latency_ms = (time.time() - start_time) * 1000
                
                # Track attempt
                step_data = {
                    "attempt": attempt_num,
                    "query": query,
                    "documents": len(docs),
                    "answer": answer,
                    "source": source,
                    "confidence": confidence,
                    "cost": cost
                }
                intermediate_steps.append(step_data)
                
                # Check if we have a good answer
                if confidence >= self.min_confidence and answer.lower() not in ["x", "not found", "n/a", "unknown"]:
                    if self.logger:
                        self.logger.info(f"✅ Success on attempt {attempt_num} (confidence: {confidence:.2%})")
                    
                    # Update metrics
                    self.metrics_collector.record_success(
                        attempts=attempt_num,
                        confidence=confidence,
                        cost=total_cost,
                        latency_ms=latency_ms,
                        strategy_used=strategy.get_name()
                    )
                    
                    return QueryResult(
                        answer=answer,
                        source=source,
                        confidence=confidence,
                        attempts=attempt_num,
                        cost=total_cost,
                        intermediate_steps=intermediate_steps if return_intermediate_steps else None
                    )
        
        # All attempts failed
        latency_ms = (time.time() - start_time) * 1000
        
        if self.logger:
            self.logger.warning(f"⚠️ All {len(intermediate_steps)} attempts failed")
        
        # Return best attempt or default
        if intermediate_steps:
            best_attempt = max(intermediate_steps, key=lambda x: x.get("confidence", 0.0))
            best_answer = best_attempt.get("answer", "No answer found")
            best_source = best_attempt.get("source", "")
            best_confidence = best_attempt.get("confidence", 0.0)
        else:
            best_answer = "No answer found"
            best_source = ""
            best_confidence = 0.0
        
        self.metrics_collector.record_failure(
            attempts=len(intermediate_steps),
            cost=total_cost,
            latency_ms=latency_ms,
            strategy_used=self.strategies[0].get_name() if self.strategies else "unknown"
        )
        
        return QueryResult(
            answer=best_answer,
            source=best_source,
            confidence=best_confidence,
            attempts=len(intermediate_steps) or 1,
            cost=total_cost,
            intermediate_steps=intermediate_steps if return_intermediate_steps else None
        )
    
    def _retrieve_documents(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List:
        """Retrieve documents from vector store."""
        try:
            # Apply context filters if supported
            search_kwargs = self._build_search_kwargs(context)
            
            retriever = self.vector_store.as_retriever(
                search_kwargs=search_kwargs
            )
            
            docs = retriever.get_relevant_documents(query)
            return docs
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _generate_answer(
        self,
        question: str,
        query: str,
        documents: List,
        context: Dict[str, Any]
    ) -> Tuple[str, str, float, float]:
        """
        Generate answer from documents.
        
        Returns:
            Tuple of (answer, source, confidence, cost)
        """
        # Format documents
        docs_text = self._format_documents(documents)
        
        # Create prompt
        prompt = self._build_answer_prompt(question, docs_text, context)
        
        # Generate answer with cost tracking
        with self.cost_tracker.track(operation="answer_generation"):
            messages = [
                SystemMessage(content=self.answer_prompt_template),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract token usage if available
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    self.cost_tracker.record_tokens(
                        input_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('completion_tokens', 0),
                        model=getattr(self.llm, 'model_name', 'gpt-4')
                    )
        
        # Extract answer and source
        answer, source = self._parse_answer(answer_text)
        
        # Calculate confidence
        scorer = ConfidenceScorer(llm=self.llm)
        confidence = scorer.score(
            question=question,
            answer=answer,
            documents=documents,
            context=context
        )
        
        # Get cost
        cost = self.cost_tracker.get_last_cost()
        
        return answer, source, confidence, cost
    
    def _format_documents(self, documents: List) -> str:
        """Format documents for prompt."""
        formatted = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            formatted.append(f"Document {i} (from {source}):\n{content}\n")
        return "\n".join(formatted)
    
    def _build_answer_prompt(
        self,
        question: str,
        docs_text: str,
        context: Dict[str, Any]
    ) -> str:
        """Build answer generation prompt."""
        context_str = ""
        if context:
            context_str = f"\n\nContext: {json.dumps(context, indent=2)}\n"
        
        return f"""Based on the following documents, answer the question.

Question: {question}
{context_str}
Documents:
{docs_text}

Provide a clear, concise answer. If the answer is not in the documents, respond with "Not found".
Return your answer in JSON format: {{"answer": "...", "source": "..."}}"""
    
    def _parse_answer(self, answer_text: str) -> Tuple[str, str]:
        """Parse answer from LLM response."""
        # Try to extract JSON
        json_match = re.search(r'\{[^}]+\}', answer_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return parsed.get("answer", answer_text), parsed.get("source", "")
            except json.JSONDecodeError:
                pass
        
        # Fallback: return as-is
        return answer_text, ""
    
    def _build_search_kwargs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build search kwargs with context filters."""
        kwargs = {"k": 5}  # Default top-k
        
        # Add context-based filters if vector store supports it
        if hasattr(self.vector_store, 'filter'):
            filters = {}
            for key, value in context.items():
                if key in ["company_key", "unique_id", "filter_id"]:
                    filters[key] = value
            if filters:
                kwargs["filter"] = filters
        
        return kwargs

