"""
Basic usage example for ragfallback.

This example shows how to use ragfallback with OpenAI (paid).
For open-source examples, see:
- examples/open_source_example.py (Ollama + HuggingFace + FAISS)
- examples/paid_llm_example.py (OpenAI/Anthropic + open-source vector store)
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_openai_llm,
    create_openai_embeddings,
    create_chroma_vector_store
)
from langchain.docstore.document import Document


def main():
    """Basic usage example."""
    # Python documentation content
    documents = [
        Document(
            page_content="Python is a high-level programming language known for simplicity and readability. It supports multiple paradigms including procedural, object-oriented, and functional programming. Python's extensive standard library makes it suitable for many applications.",
            metadata={"source": "python_intro.pdf", "topic": "introduction"}
        ),
        Document(
            page_content="Python lists are mutable sequences created with square brackets: numbers = [1, 2, 3]. Common operations include append() to add items, remove() to delete items, and len() to get length. List indexing starts at 0.",
            metadata={"source": "python_lists.pdf", "topic": "data_structures"}
        ),
        Document(
            page_content="Python dictionaries store key-value pairs: person = {'name': 'Alice', 'age': 30}. Access values using keys: person['name'] returns 'Alice'. Use get() method for safe access: person.get('email', 'N/A') returns 'N/A' if key doesn't exist.",
            metadata={"source": "python_dicts.pdf", "topic": "data_structures"}
        ),
    ]
    
    # Create embeddings (OpenAI - paid, requires API key)
    # For open-source alternative, use: create_open_source_embeddings()
    print("Creating embeddings...")
    embeddings = create_openai_embeddings(model="text-embedding-3-small")
    
    # Initialize vector store
    print("Initializing vector store...")
    # Using ChromaDB (open-source, local)
    # For FAISS alternative, use: create_faiss_vector_store()
    vector_store = create_chroma_vector_store(
        documents=documents,
        embeddings=embeddings
    )
    
    # Initialize LLM (OpenAI - paid, requires API key)
    # For open-source alternative, use: create_open_source_llm()
    print("Initializing LLM...")
    llm = create_openai_llm(model="gpt-4o-mini", temperature=0)
    
    # Setup cost tracking
    cost_tracker = CostTracker(budget=10.0)  # $10 budget
    
    # Setup metrics
    metrics = MetricsCollector()
    
    # Create adaptive retriever
    print("Creating adaptive retriever...")
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
    print("\n" + "="*50)
    print("Querying with fallback...")
    print("="*50)
    
    result = retriever.query_with_fallback(
        question="How do I create a list in Python?",
        return_intermediate_steps=True
    )
    
    # Print results
    print(f"\n✅ Answer: {result.answer}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print(f"📄 Source: {result.source}")
    print(f"🔄 Attempts: {result.attempts}")
    print(f"💰 Cost: ${result.cost:.4f}")
    
    # Print intermediate steps
    if result.intermediate_steps:
        print("\n" + "-"*50)
        print("Intermediate Steps:")
        print("-"*50)
        for step in result.intermediate_steps:
            print(f"\nAttempt {step['attempt']}:")
            print(f"  Query: {step['query'][:60]}...")
            print(f"  Documents: {step['documents']}")
            print(f"  Confidence: {step['confidence']:.2%}")
            print(f"  Cost: ${step['cost']:.4f}")
    
    # Print metrics
    print("\n" + "="*50)
    print("Overall Metrics:")
    print("="*50)
    stats = metrics.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Average Attempts: {stats['avg_attempts']:.2f}")
    
    # Print cost report
    print("\n" + "="*50)
    print("Cost Report:")
    print("="*50)
    report = cost_tracker.get_report()
    print(f"Total Cost: ${report['total_cost']:.4f}")
    if report['budget_remaining'] is not None:
        print(f"Budget Remaining: ${report['budget_remaining']:.4f}")
        print(f"Budget Usage: {report['budget_usage_percent']:.1f}%")


if __name__ == "__main__":
    main()

