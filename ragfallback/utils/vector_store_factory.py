"""Factory functions for creating vector stores (open-source and paid)."""

from typing import List, Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def create_faiss_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    persist_directory: Optional[str] = None
) -> VectorStore:
    """
    Create a FAISS vector store (open-source, runs locally, no API key needed).
    
    Args:
        documents: List of Document objects
        embeddings: Embeddings instance
        persist_directory: Optional directory to persist the index
    
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
        
        if persist_directory:
            # Try to load existing index
            try:
                return FAISS.load_local(persist_directory, embeddings)
            except:
                pass
        
        vector_store = FAISS.from_documents(documents, embeddings)
        
        if persist_directory:
            vector_store.save_local(persist_directory)
        
        return vector_store
    except ImportError:
        raise ImportError(
            "FAISS not installed. Install with: pip install faiss-cpu"
        )


def create_chroma_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    persist_directory: Optional[str] = None,
    collection_name: str = "ragfallback"
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
        
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    except ImportError:
        raise ImportError(
            "ChromaDB not installed. Install with: pip install chromadb"
        )


def create_qdrant_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    collection_name: str = "ragfallback",
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    prefer_grpc: bool = False
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
        
        url = url or "http://localhost:6333"
        
        return Qdrant.from_documents(
            documents=documents,
            embedding=embeddings,
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            collection_name=collection_name
        )
    except ImportError:
        raise ImportError(
            "Qdrant not installed. Install with: pip install qdrant-client"
        )


def create_pinecone_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    index_name: str = "ragfallback",
    api_key: Optional[str] = None,
    environment: Optional[str] = None
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
        from langchain_community.vectorstores import Pinecone
        import pinecone
        
        if api_key:
            pinecone.init(api_key=api_key, environment=environment)
        
        return Pinecone.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )
    except ImportError:
        raise ImportError(
            "Pinecone not installed. Install with: pip install pinecone-client"
        )

