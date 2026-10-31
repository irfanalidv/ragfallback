"""Factory functions for creating vector stores (open-source and paid)."""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


def create_faiss_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    persist_directory: str | None = None,
    allow_dangerous_deserialization: bool = False,
) -> VectorStore:
    """
    Create a FAISS vector store (open-source, runs locally, no API key needed).

    Args:
        documents: List of Document objects
        embeddings: Embeddings instance
        persist_directory: Optional directory to persist the index
        allow_dangerous_deserialization: Reload an existing index from
            ``persist_directory`` instead of rebuilding it. FAISS stores its docstore
            with pickle, so only enable this for directories you wrote yourself.

    Returns:
        VectorStore instance

    Example:
        >>> from ragfallback.utils.vector_store_factory import create_faiss_vector_store
        >>> from ragfallback.utils.embedding_factory import create_open_source_embeddings
        >>> embeddings = create_open_source_embeddings()
        >>> vector_store = create_faiss_vector_store(documents, embeddings)
    """
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as e:
        raise ImportError("FAISS not installed. Install with: pip install faiss-cpu") from e

    if persist_directory and (Path(persist_directory) / "index.faiss").exists():
        if allow_dangerous_deserialization:
            return FAISS.load_local(
                persist_directory, embeddings, allow_dangerous_deserialization=True
            )
        logger.warning(
            "FAISS index found at %s but allow_dangerous_deserialization=False; "
            "rebuilding and overwriting it.",
            persist_directory,
        )

    vector_store = FAISS.from_documents(documents, embeddings)
    if persist_directory:
        vector_store.save_local(persist_directory)
    return vector_store


def create_chroma_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    persist_directory: str | None = None,
    collection_name: str = "ragfallback",
) -> VectorStore:
    """
    Create a ChromaDB vector store (open-source, runs locally, no API key needed).

    Args:
        documents: List of Document objects
        embeddings: Embeddings instance
        persist_directory: Optional directory to persist the database
        collection_name: Name of the collection

    Returns:
        VectorStore instance
    """
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError as e:
        raise ImportError("ChromaDB not installed. Install with: pip install chromadb") from e
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def create_qdrant_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    collection_name: str = "ragfallback",
    url: str | None = None,
    api_key: str | None = None,
    prefer_grpc: bool = False,
) -> VectorStore:
    """
    Create a Qdrant vector store (open-source, can run locally or cloud).

    Args:
        documents: List of Document objects
        embeddings: Embeddings instance
        collection_name: Name of the collection
        url: Qdrant server URL (default: "http://localhost:6333" for local)
        api_key: Qdrant API key (only needed for cloud)
        prefer_grpc: Use gRPC instead of REST API

    Returns:
        VectorStore instance
    """
    try:
        from langchain_community.vectorstores import Qdrant
    except ImportError as e:
        raise ImportError("Qdrant not installed. Install with: pip install qdrant-client") from e
    return Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        url=url or "http://localhost:6333",
        api_key=api_key,
        prefer_grpc=prefer_grpc,
        collection_name=collection_name,
    )


def create_pinecone_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    index_name: str = "ragfallback",
    api_key: str | None = None,
    environment: str | None = None,
) -> VectorStore:
    """
    Create a Pinecone vector store (paid cloud service, requires API key).

    Args:
        documents: List of Document objects
        embeddings: Embeddings instance
        index_name: Name of the Pinecone index
        api_key: Pinecone API key (uses PINECONE_API_KEY env var if not provided)
        environment: Pinecone environment (uses PINECONE_ENVIRONMENT env var if not provided)

    Returns:
        VectorStore instance
    """
    try:
        import pinecone
        from langchain_community.vectorstores import Pinecone
    except ImportError as e:
        raise ImportError(
            "Pinecone not installed. Install with: pip install pinecone-client"
        ) from e
    if api_key:
        pinecone.init(api_key=api_key, environment=environment)
    return Pinecone.from_documents(documents=documents, embedding=embeddings, index_name=index_name)
