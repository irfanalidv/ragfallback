"""Tests for ConfidenceScorer."""

import pytest
from langchain_core.documents import Document
from ragfallback.utils.confidence_scorer import ConfidenceScorer


def test_confidence_scorer_heuristic():
    """Test heuristic-based confidence scoring with real documents."""
    scorer = ConfidenceScorer(method="heuristic")
    
    # Good answer with multiple documents
    documents = [
        Document(page_content="Acme Corp revenue is $10M annually.", metadata={"source": "report.pdf"}),
        Document(page_content="The company was founded in 2020.", metadata={"source": "info.pdf"}),
        Document(page_content="Acme Corp is a technology company.", metadata={"source": "about.pdf"})
    ]
    
    confidence = scorer.score(
        question="What is the revenue?",
        answer="The revenue is $10M annually",
        documents=documents
    )
    assert 0.0 <= confidence <= 1.0
    
    # Failure indicator
    confidence_fail = scorer.score(
        question="What is the revenue?",
        answer="Not found",
        documents=[documents[0]]
    )
    assert confidence_fail == 0.0


def test_confidence_scorer_with_llm(real_llm):
    """Test LLM-based confidence scoring with real LLM."""
    scorer = ConfidenceScorer(llm=real_llm, method="llm")
    
    documents = [
        Document(page_content="Acme Corp revenue is $10M annually.", metadata={"source": "report.pdf"}),
        Document(page_content="The company was founded in 2020.", metadata={"source": "info.pdf"})
    ]
    
    confidence = scorer.score(
        question="What is the revenue?",
        answer="The revenue is $10M",
        documents=documents
    )
    
    assert 0.0 <= confidence <= 1.0


def test_confidence_scorer_with_different_answers(real_llm):
    """Test confidence scoring with different answer qualities."""
    scorer = ConfidenceScorer(method="heuristic")
    
    documents = [
        Document(page_content="Acme Corp revenue is $10M annually.", metadata={"source": "report.pdf"}),
        Document(page_content="The company was founded in 2020.", metadata={"source": "info.pdf"})
    ]
    
    # High confidence answer
    high_conf = scorer.score(
        question="What is the revenue?",
        answer="The revenue is $10M annually according to the annual report",
        documents=documents
    )
    
    # Low confidence answer
    low_conf = scorer.score(
        question="What is the revenue?",
        answer="I don't know",
        documents=documents
    )
    
    assert high_conf >= low_conf
    assert 0.0 <= high_conf <= 1.0
    assert 0.0 <= low_conf <= 1.0

