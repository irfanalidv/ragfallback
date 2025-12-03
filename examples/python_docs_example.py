"""
Python Documentation Example

This example demonstrates ragfallback with Python documentation content.
Uses HuggingFace Inference API (free tier) - no API keys needed!
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document


def main():
    """Python documentation example."""
    print("="*70)
    print("ragfallback - Python Documentation Example")
    print("="*70)
    print("\nThis example uses Python documentation content.")
    print("Uses HuggingFace Inference API (free tier) - no API keys needed!\n")
    
    # Python documentation content
    documents = [
        Document(
            page_content="Python is a high-level, interpreted programming language with dynamic semantics. Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse.",
            metadata={"source": "python_intro.pdf", "topic": "introduction"}
        ),
        Document(
            page_content="Python lists are mutable sequences, typically used to store collections of homogeneous items. Lists are created using square brackets: my_list = [1, 2, 3]. You can add items using append(), insert(), or extend() methods. List comprehension provides a concise way to create lists.",
            metadata={"source": "python_lists.pdf", "topic": "data_structures"}
        ),
        Document(
            page_content="Python dictionaries are unordered collections of key-value pairs. Dictionaries are created using curly braces: my_dict = {'name': 'Python', 'version': '3.11'}. You can access values using keys: my_dict['name']. The get() method returns None if key doesn't exist, preventing KeyError.",
            metadata={"source": "python_dicts.pdf", "topic": "data_structures"}
        ),
        Document(
            page_content="Python functions are defined using the def keyword. Functions can accept arguments, return values, and have default parameters. Example: def greet(name='World'): return f'Hello, {name}'. Functions are first-class objects in Python, meaning they can be passed as arguments to other functions.",
            metadata={"source": "python_functions.pdf", "topic": "functions"}
        ),
        Document(
            page_content="Python classes are created using the class keyword. Classes can have attributes and methods. The __init__ method is the constructor. Example: class Dog: def __init__(self, name): self.name = name. Classes support inheritance, allowing child classes to inherit from parent classes.",
            metadata={"source": "python_classes.pdf", "topic": "oop"}
        ),
        Document(
            page_content="Python exception handling uses try, except, else, and finally blocks. The try block contains code that might raise an exception. The except block handles specific exceptions. Example: try: result = 10/0 except ZeroDivisionError: print('Cannot divide by zero'). The finally block always executes.",
            metadata={"source": "python_exceptions.pdf", "topic": "error_handling"}
        ),
        Document(
            page_content="Python file operations use the open() function. Files can be opened in read mode ('r'), write mode ('w'), or append mode ('a'). Always use context managers (with statement) for file operations: with open('file.txt', 'r') as f: content = f.read(). This ensures files are properly closed.",
            metadata={"source": "python_files.pdf", "topic": "file_io"}
        ),
        Document(
            page_content="Python list comprehension is a concise way to create lists. Syntax: [expression for item in iterable if condition]. Example: squares = [x**2 for x in range(10)] creates a list of squares. List comprehensions are more readable and often faster than equivalent for loops.",
            metadata={"source": "python_comprehensions.pdf", "topic": "advanced"}
        ),
    ]
    
    print(f"📚 Loaded {len(documents)} Python documentation pages\n")
    
    # Create embeddings
    print("Creating embeddings (HuggingFace - free)...")
    embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Embeddings created\n")
    
    # Create vector store
    print("Creating vector store (FAISS - free)...")
    vector_store = create_faiss_vector_store(
        documents=documents,
        embeddings=embeddings
    )
    print("✅ Vector store created\n")
    
    # Create LLM
    print("Creating LLM (HuggingFace Inference API - free tier)...")
    llm = create_huggingface_llm(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        use_inference_api=True,
        temperature=0,
        max_length=512
    )
    print("✅ LLM ready\n")
    
    # Setup tracking
    cost_tracker = CostTracker()
    metrics = MetricsCollector()
    
    # Create retriever
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
    
    # Queries about Python
    queries = [
        "How do I create a list in Python?",
        "What is a dictionary and how do I use it?",
        "How do I handle errors in Python?",
        "Explain Python classes",
    ]
    
    print("="*70)
    print("Querying Python Documentation")
    print("="*70)
    
    for i, question in enumerate(queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {question}")
        print(f"{'─'*70}")
        
        result = retriever.query_with_fallback(
            question=question,
            return_intermediate_steps=True
        )
        
        print(f"\n✅ Answer: {result.answer}")
        print(f"📊 Confidence: {result.confidence:.2%}")
        print(f"📄 Source: {result.source}")
        print(f"🔄 Attempts: {result.attempts}")
        
        if result.intermediate_steps:
            print(f"\n📋 Query Variations:")
            for step in result.intermediate_steps[:2]:  # Show first 2
                print(f"   Attempt {step['attempt']}: '{step['query'][:60]}...'")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    stats = metrics.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")


if __name__ == "__main__":
    main()

