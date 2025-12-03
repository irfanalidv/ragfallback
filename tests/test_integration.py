"""Integration tests for end-to-end RAG workflows."""

import pytest
from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.strategies.query_variations import QueryVariationsStrategy


def test_full_retrieval_workflow(real_llm, real_embeddings, real_vector_store):
    """Test complete RAG retrieval workflow with real components."""
    cost_tracker = CostTracker()
    metrics = MetricsCollector()
    
    retriever = AdaptiveRAGRetriever(
        vector_store=real_vector_store,
        llm=real_llm,
        embedding_model=real_embeddings,
        fallback_strategy="query_variations",
        cost_tracker=cost_tracker,
        metrics_collector=metrics,
        max_attempts=2,
        min_confidence=0.5
    )
    
    # Query with fallback
    result = retriever.query_with_fallback(
        question="What is Acme Corp's revenue?",
        context={"company": "Acme Corp"},
        return_intermediate_steps=True
    )
    
    # Verify result structure
    assert result.answer is not None
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert 0.0 <= result.confidence <= 1.0
    assert result.attempts >= 1
    assert result.cost >= 0.0
    
    # Verify intermediate steps
    if result.intermediate_steps:
        assert len(result.intermediate_steps) == result.attempts
        for step in result.intermediate_steps:
            assert "attempt" in step
            assert "query" in step
            assert "confidence" in step
    
    # Verify metrics were recorded
    stats = metrics.get_stats()
    assert stats["total_queries"] >= 1
    
    # Verify cost tracking
    report = cost_tracker.get_report()
    assert report["total_cost"] >= 0.0


def test_query_variations_strategy_integration(real_llm, real_vector_store, real_embeddings):
    """Test query variations strategy with real LLM and vector store."""
    strategy = QueryVariationsStrategy(num_variations=2, include_original=True)
    
    queries = strategy.generate_queries(
        original_query="What is the revenue?",
        context={"company": "Acme Corp"},
        attempt=1,
        llm=real_llm
    )
    
    # Verify queries were generated
    assert len(queries) >= 1
    assert all(isinstance(q, str) for q in queries)
    
    # Test retrieval with each query
    retriever = real_vector_store.as_retriever(search_kwargs={"k": 2})
    
    for query in queries[:2]:  # Test first 2 queries
        docs = retriever.get_relevant_documents(query)
        assert len(docs) >= 0  # May or may not find documents
        if docs:
            assert all(hasattr(doc, 'page_content') for doc in docs)


def test_confidence_scoring_integration(real_llm, real_vector_store, real_embeddings):
    """Test confidence scoring with real components."""
    from ragfallback.utils.confidence_scorer import ConfidenceScorer
    
    scorer = ConfidenceScorer(method="heuristic")
    
    # Get documents from vector store
    retriever = real_vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents("What is the revenue?")
    
    if docs:
        # Generate answer using LLM
        from langchain_core.messages import HumanMessage
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Based on this context, answer the question: What is Acme Corp's revenue?\n\nContext:\n{context}\n\nAnswer:"
        
        response = real_llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Score confidence
        confidence = scorer.score(
            question="What is Acme Corp's revenue?",
            answer=answer,
            documents=docs
        )
        
        assert 0.0 <= confidence <= 1.0

