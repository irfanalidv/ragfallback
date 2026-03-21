"""Pytest configuration and fixtures."""

import pytest
from langchain_core.documents import Document


@pytest.fixture(scope="session")
def real_llm():
    """
    Prefer Mistral when MISTRAL_API_KEY is set (.env supported via python-dotenv).
    Otherwise HuggingFace Inference API, then Ollama.
    """
    try:
        from ragfallback.utils.llm_factory import create_mistral_llm, load_env

        load_env()
        return create_mistral_llm(load_dotenv=False, temperature=0)
    except Exception:
        pass
    try:
        from ragfallback.utils.llm_factory import create_huggingface_llm

        return create_huggingface_llm(
            model_id="google/flan-t5-base",
            use_inference_api=True,
            temperature=0,
        )
    except Exception:
        try:
            from ragfallback.utils.llm_factory import create_open_source_llm

            return create_open_source_llm(
                model="llama3",
                provider="ollama",
                temperature=0,
            )
        except Exception:
            pytest.skip(
                "No LLM: set MISTRAL_API_KEY, or install HF/Ollama fallbacks."
            )


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

