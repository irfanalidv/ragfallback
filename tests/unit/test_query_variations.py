"""Tests for QueryVariationsStrategy using real LLM."""

import pytest
from ragfallback.strategies.query_variations import QueryVariationsStrategy


def test_query_variations_strategy_initialization():
    """Test QueryVariationsStrategy initialization."""
    strategy = QueryVariationsStrategy(num_variations=2)
    assert strategy.num_variations == 2
    assert strategy.include_original is True


def test_query_variations_generate_queries_with_original(real_llm):
    """Test generating queries with original included using real LLM."""
    strategy = QueryVariationsStrategy(num_variations=2, include_original=True)
    
    queries = strategy.generate_queries(
        original_query="What is the revenue?",
        context={"company": "Acme Corp"},
        attempt=1,
        llm=real_llm
    )
    
    # Should have original + variations
    assert len(queries) >= 1
    assert queries[0] == "What is the revenue?"  # Original should be first
    # All queries should be strings
    assert all(isinstance(q, str) for q in queries)
    assert all(len(q) > 0 for q in queries)


def test_query_variations_generate_queries_without_original(real_llm):
    """Test generating queries without original using real LLM."""
    strategy = QueryVariationsStrategy(num_variations=2, include_original=False)
    
    queries = strategy.generate_queries(
        original_query="What is the revenue?",
        context={"company": "Acme Corp"},
        attempt=1,
        llm=real_llm
    )
    
    # Should have variations only
    assert len(queries) >= 1
    assert "What is the revenue?" not in queries  # Original should not be included
    # All queries should be strings
    assert all(isinstance(q, str) for q in queries)


def test_query_variations_with_context(real_llm):
    """Test query variations with context using real LLM."""
    strategy = QueryVariationsStrategy(num_variations=2, include_original=True)
    
    queries = strategy.generate_queries(
        original_query="What is the company size?",
        context={"company": "Acme Corp", "industry": "Technology"},
        attempt=1,
        llm=real_llm
    )
    
    assert len(queries) >= 1
    assert all(isinstance(q, str) for q in queries)


def test_query_variations_handles_errors_gracefully(real_llm):
    """Test that query variations handles errors gracefully."""
    strategy = QueryVariationsStrategy(num_variations=2, include_original=True)
    
    # Even if LLM fails, should return at least original query
    queries = strategy.generate_queries(
        original_query="test query",
        context={},
        attempt=1,
        llm=real_llm
    )
    
    # Should always return at least one query
    assert len(queries) >= 1
    assert all(isinstance(q, str) for q in queries)

