"""Pytest configuration and fixtures."""

import pytest
from langchain_core.documents import Document


@pytest.fixture(scope="session")
def real_llm():
    """
    Real LLM for testing - uses HuggingFace Inference API (free tier, no API key needed).
    Falls back to Ollama if HuggingFace is unavailable.
    """
    try:
        # Try HuggingFace Inference API first (no API key needed for public models)
        from ragfallback.utils.llm_factory import create_huggingface_llm
        return create_huggingface_llm(
            model_id="google/flan-t5-base",  # Small, fast model for testing
            use_inference_api=True,
            temperature=0
        )
    except Exception:
        # Fallback to Ollama if available
        try:
            from ragfallback.utils.llm_factory import create_open_source_llm
            return create_open_source_llm(
                model="llama3",
                provider="ollama",
                temperature=0
            )
        except Exception:
            pytest.skip("No LLM available for testing. Install Ollama or use HuggingFace Inference API.")


@pytest.fixture(scope="session")
def real_embeddings():
    """Real embeddings for testing - uses HuggingFace (local, no API key needed)."""
    try:
        from ragfallback.utils.embedding_factory import create_open_source_embeddings
        return create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    except Exception:
        pytest.skip("HuggingFace embeddings not available. Install with: pip install sentence-transformers")


@pytest.fixture
def real_vector_store(real_embeddings):
    """Real vector store for testing - uses FAISS (local, no API key needed)."""
    from langchain_core.documents import Document
    from ragfallback.utils.vector_store_factory import create_faiss_vector_store
    
    documents = [
        Document(
            page_content="Acme Corp revenue is $10M annually. The company was founded in 2020.",
            metadata={"source": "annual_report.pdf"}
        ),
        Document(
            page_content="Acme Corp is a technology company specializing in AI solutions.",
            metadata={"source": "company_info.pdf"}
        ),
        Document(
            page_content="Acme Corp has 50 employees and is headquartered in San Francisco.",
            metadata={"source": "company_info.pdf"}
        ),
    ]
    
    return create_faiss_vector_store(
        documents=documents,
        embeddings=real_embeddings
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            page_content="Acme Corp revenue is $10M annually. The company was founded in 2020.",
            metadata={"source": "annual_report.pdf"}
        ),
        Document(
            page_content="Acme Corp is a technology company specializing in AI solutions.",
            metadata={"source": "company_info.pdf"}
        ),
        Document(
            page_content="Acme Corp has 50 employees and is headquartered in San Francisco.",
            metadata={"source": "company_info.pdf"}
        ),
    ]

