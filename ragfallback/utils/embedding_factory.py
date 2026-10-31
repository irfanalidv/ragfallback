"""Factory functions for creating embeddings (open-source and paid)."""

from langchain_core.embeddings import Embeddings


def create_open_source_embeddings(model_name: str = "all-MiniLM-L6-v2") -> Embeddings:
    """
    Create open-source embeddings using HuggingFace (runs locally, no API key needed).

    Args:
        model_name: HuggingFace model name
                   Options: "all-MiniLM-L6-v2", "all-mpnet-base-v2",
                   "sentence-transformers/all-MiniLM-L6-v2"

    Returns:
        Embeddings instance

    Example:
        >>> from ragfallback.utils.embedding_factory import create_open_source_embeddings
        >>> embeddings = create_open_source_embeddings()
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as e:
            raise ImportError(
                "HuggingFace embeddings not installed. "
                "Install with: pip install ragfallback[huggingface]"
            ) from e
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def create_ollama_embeddings(
    model: str = "nomic-embed-text", base_url: str | None = None
) -> Embeddings:
    """
    Create embeddings using Ollama (open-source, runs locally).

    Args:
        model: Ollama embedding model name
        base_url: Ollama base URL (default: "http://localhost:11434")

    Returns:
        Embeddings instance
    """
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError as e:
        raise ImportError(
            "Ollama embeddings not installed. Install with: pip install langchain-community"
        ) from e
    return OllamaEmbeddings(model=model, base_url=base_url or "http://localhost:11434")


def create_openai_embeddings(
    model: str = "text-embedding-3-small", api_key: str | None = None
) -> Embeddings:
    """
    Create OpenAI embeddings (paid, requires API key).

    Args:
        model: OpenAI embedding model name
        api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)

    Returns:
        Embeddings instance
    """
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        raise ImportError(
            "OpenAI embeddings not installed. Install with: pip install langchain-openai"
        ) from e
    return OpenAIEmbeddings(model=model, api_key=api_key)


def create_anthropic_embeddings(api_key: str | None = None) -> Embeddings:
    """
    Create Anthropic embeddings (paid, requires API key).

    Args:
        api_key: Anthropic API key (optional, uses ANTHROPIC_API_KEY env var if not provided)

    Raises:
        NotImplementedError: always. Anthropic does not offer an embeddings API; use
            :func:`create_openai_embeddings` or :func:`create_open_source_embeddings`.
    """
    raise NotImplementedError(
        "Anthropic doesn't provide separate embeddings. "
        "Use OpenAI embeddings or open-source alternatives."
    )
