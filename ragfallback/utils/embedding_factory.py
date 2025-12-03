"""Factory functions for creating embeddings (open-source and paid)."""

from typing import Optional
from langchain_core.embeddings import Embeddings


def create_open_source_embeddings(
    model_name: str = "all-MiniLM-L6-v2"
) -> Embeddings:
    """
    Create open-source embeddings using HuggingFace (runs locally, no API key needed).
    
    Args:
        model_name: HuggingFace model name
                   Options: "all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"
    
    Returns:
        Embeddings instance
        
    Example:
        >>> from ragfallback.utils.embedding_factory import create_open_source_embeddings
        >>> embeddings = create_open_source_embeddings()
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if GPU available
        )
    except ImportError:
        raise ImportError(
            "HuggingFace embeddings not installed. "
            "Install with: pip install sentence-transformers"
        )


def create_ollama_embeddings(
    model: str = "nomic-embed-text",
    base_url: Optional[str] = None
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
        
        base_url = base_url or "http://localhost:11434"
        return OllamaEmbeddings(
            model=model,
            base_url=base_url
        )
    except ImportError:
        raise ImportError(
            "Ollama embeddings not installed. "
            "Install with: pip install langchain-community"
        )


def create_openai_embeddings(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None
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
        
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key
        )
    except ImportError:
        raise ImportError(
            "OpenAI embeddings not installed. "
            "Install with: pip install langchain-openai"
        )


def create_anthropic_embeddings(
    api_key: Optional[str] = None
) -> Embeddings:
    """
    Create Anthropic embeddings (paid, requires API key).
    
    Args:
        api_key: Anthropic API key (optional, uses ANTHROPIC_API_KEY env var if not provided)
    
    Returns:
        Embeddings instance
    """
    try:
        from langchain_anthropic import ChatAnthropic
        
        # Note: Anthropic doesn't have separate embeddings, use OpenAI or open-source
        raise NotImplementedError(
            "Anthropic doesn't provide separate embeddings. "
            "Use OpenAI embeddings or open-source alternatives."
        )
    except ImportError:
        raise ImportError(
            "Anthropic not installed. Install with: pip install langchain-anthropic"
        )

