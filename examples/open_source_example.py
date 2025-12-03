"""
Open-Source Example - No API keys required!

This example uses:
- Ollama for LLM (runs locally)
- HuggingFace embeddings (runs locally)
- FAISS vector store (runs locally)

Setup:
1. Install Ollama: https://ollama.ai
2. Pull a model: ollama pull llama3
3. Install dependencies: pip install sentence-transformers faiss-cpu
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_open_source_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document


def main():
    """Open-source example - no API keys needed!"""
    print("="*60)
    print("ragfallback - Open-Source Example")
    print("="*60)
    print("\nThis example uses:")
    print("  - Ollama LLM (local, free)")
    print("  - HuggingFace embeddings (local, free)")
    print("  - FAISS vector store (local, free)")
    print("\nNo API keys required! 🎉\n")
    
    # Python documentation content
    documents = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python's extensive standard library and large ecosystem of third-party packages make it suitable for a wide range of applications.",
            metadata={"source": "python_intro.pdf", "topic": "introduction"}
        ),
        Document(
            page_content="Python lists are mutable sequences that can hold items of different types. Lists are created using square brackets: numbers = [1, 2, 3]. Common operations include append() to add items, remove() to delete items, and len() to get the length. List indexing starts at 0.",
            metadata={"source": "python_lists.pdf", "topic": "data_structures"}
        ),
        Document(
            page_content="Python dictionaries store key-value pairs and are created with curly braces: person = {'name': 'Alice', 'age': 30}. Access values using keys: person['name'] returns 'Alice'. Use get() method for safe access: person.get('email', 'N/A') returns 'N/A' if key doesn't exist.",
            metadata={"source": "python_dicts.pdf", "topic": "data_structures"}
        ),
        Document(
            page_content="Python functions are defined with the def keyword. Functions can have parameters with default values: def greet(name='World'): return f'Hello, {name}'. Functions can return multiple values as tuples. Functions are first-class objects and can be assigned to variables or passed as arguments.",
            metadata={"source": "python_functions.pdf", "topic": "functions"}
        ),
    ]
    
    # Create open-source embeddings (no API key needed!)
    print("Creating open-source embeddings...")
    embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    
    # Create FAISS vector store (local, free)
    print("Creating FAISS vector store...")
    vector_store = create_faiss_vector_store(
        documents=documents,
        embeddings=embeddings,
        persist_directory="./faiss_index"  # Optional: persist to disk
    )
    
    # Create open-source LLM using Ollama (no API key needed!)
    print("Creating Ollama LLM...")
    print("Note: Make sure Ollama is running: ollama serve")
    print("      And you have pulled a model: ollama pull llama3")
    llm = create_open_source_llm(
        model="llama3",  # or "llama2", "mistral", etc.
        base_url="http://localhost:11434",  # Default Ollama URL
        temperature=0
    )
    
    # Setup cost tracking (costs will be 0 for open-source!)
    cost_tracker = CostTracker()
    
    # Setup metrics
    metrics = MetricsCollector()
    
    # Create adaptive retriever
    print("\nCreating adaptive retriever...")
    retriever = AdaptiveRAGRetriever(
        vector_store=vector_store,
        llm=llm,
        embedding_model=embeddings,
        fallback_strategy="query_variations",
        cost_tracker=cost_tracker,
        metrics_collector=metrics,
        max_attempts=3,
        min_confidence=0.7
    )
    
    # Query with fallback
    print("\n" + "="*60)
    print("Querying with fallback...")
    print("="*60)
    
    result = retriever.query_with_fallback(
        question="How do I create a list in Python?",
        return_intermediate_steps=True
    )
    
    # Print results
    print(f"\n✅ Answer: {result.answer}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print(f"📄 Source: {result.source}")
    print(f"🔄 Attempts: {result.attempts}")
    print(f"💰 Cost: ${result.cost:.4f} (Free with open-source!)")
    
    # Print intermediate steps
    if result.intermediate_steps:
        print("\n" + "-"*60)
        print("Intermediate Steps:")
        print("-"*60)
        for step in result.intermediate_steps:
            print(f"\nAttempt {step['attempt']}:")
            print(f"  Query: {step['query'][:60]}...")
            print(f"  Documents: {step['documents']}")
            print(f"  Confidence: {step['confidence']:.2%}")
    
    # Print metrics
    print("\n" + "="*60)
    print("Overall Metrics:")
    print("="*60)
    stats = metrics.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Average Attempts: {stats['avg_attempts']:.2f}")


if __name__ == "__main__":
    main()

