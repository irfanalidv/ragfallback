"""
Complete Example - Demonstrating All ragfallback Features

This example shows:
1. Open-source setup (HuggingFace + FAISS)
2. Query variations fallback
3. Cost tracking
4. Metrics collection
5. Intermediate steps
6. Error handling
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document


def main():
    """Complete example demonstrating all features."""
    print("="*70)
    print("ragfallback - Complete Feature Demonstration")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Setup Documents (Python Documentation)
    # ========================================================================
    print("\n📄 STEP 1: Setting up documents...")
    documents = [
        Document(
            page_content="Python is a high-level, interpreted programming language with dynamic semantics. Its simple syntax emphasizes readability, reducing program maintenance costs. Python supports modules and packages, encouraging modularity and code reuse.",
            metadata={"source": "python_intro.pdf", "topic": "introduction"}
        ),
        Document(
            page_content="Python lists are mutable sequences created with square brackets: my_list = [1, 2, 3]. Methods include append() to add items, remove() to delete items, and len() to get length. List comprehension provides concise syntax: [x**2 for x in range(10)].",
            metadata={"source": "python_lists.pdf", "topic": "data_structures"}
        ),
        Document(
            page_content="Python dictionaries store key-value pairs: person = {'name': 'Alice', 'age': 30}. Access values with keys: person['name']. Use get() for safe access: person.get('email', 'N/A'). Dictionaries are unordered in Python 3.7+.",
            metadata={"source": "python_dicts.pdf", "topic": "data_structures"}
        ),
        Document(
            page_content="Python functions are defined with def keyword: def greet(name='World'): return f'Hello, {name}'. Functions can return multiple values as tuples. Functions are first-class objects and can be passed as arguments or assigned to variables.",
            metadata={"source": "python_functions.pdf", "topic": "functions"}
        ),
        Document(
            page_content="Python classes use the class keyword: class Dog: def __init__(self, name): self.name = name. Classes support inheritance, allowing child classes to inherit from parents. Methods can be instance methods, class methods, or static methods.",
            metadata={"source": "python_classes.pdf", "topic": "oop"}
        ),
    ]
    print(f"✅ Created {len(documents)} documents")
    
    # ========================================================================
    # STEP 2: Create Embeddings (Open-Source)
    # ========================================================================
    print("\n🔤 STEP 2: Creating embeddings (HuggingFace - free)...")
    try:
        embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embeddings created successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Install with: pip install sentence-transformers")
        return
    
    # ========================================================================
    # STEP 3: Create Vector Store (FAISS - Open-Source)
    # ========================================================================
    print("\n🗄️  STEP 3: Creating vector store (FAISS - free)...")
    try:
        vector_store = create_faiss_vector_store(
            documents=documents,
            embeddings=embeddings,
            persist_directory="./faiss_index_demo"
        )
        print("✅ Vector store created successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Install with: pip install faiss-cpu")
        return
    
    # ========================================================================
    # STEP 4: Create LLM (HuggingFace Inference API - Free Tier)
    # ========================================================================
    print("\n🤖 STEP 4: Creating LLM (HuggingFace Inference API - free tier)...")
    try:
        llm = create_huggingface_llm(
            model_id="mistralai/Mistral-7B-Instruct-v0.1",
            use_inference_api=True,
            temperature=0,
            max_length=512
        )
        print("✅ LLM created successfully")
        print("💡 Using HuggingFace Inference API (free tier available)")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Install with: pip install huggingface-hub")
        print("💡 Or use Ollama: ollama pull llama3")
        return
    
    # ========================================================================
    # STEP 5: Setup Cost Tracking
    # ========================================================================
    print("\n💰 STEP 5: Setting up cost tracking...")
    cost_tracker = CostTracker(budget=10.0)  # $10 budget
    print("✅ Cost tracker initialized (budget: $10.00)")
    
    # ========================================================================
    # STEP 6: Setup Metrics Collection
    # ========================================================================
    print("\n📊 STEP 6: Setting up metrics collection...")
    metrics = MetricsCollector()
    print("✅ Metrics collector initialized")
    
    # ========================================================================
    # STEP 7: Create Adaptive Retriever
    # ========================================================================
    print("\n🔄 STEP 7: Creating adaptive retriever with fallback strategies...")
    retriever = AdaptiveRAGRetriever(
        vector_store=vector_store,
        llm=llm,
        embedding_model=embeddings,
        fallback_strategy="query_variations",
        cost_tracker=cost_tracker,
        metrics_collector=metrics,
        max_attempts=3,
        min_confidence=0.7,
        enable_logging=True
    )
    print("✅ Adaptive retriever created")
    print("   - Fallback strategy: Query Variations")
    print("   - Max attempts: 3")
    print("   - Min confidence: 0.7")
    
    # ========================================================================
    # STEP 8: Test Queries
    # ========================================================================
    print("\n" + "="*70)
    print("🧪 TESTING: Running queries with fallback")
    print("="*70)
    
    test_queries = [
        {
            "question": "How do I create a list in Python?",
            "context": {},
            "description": "Direct question about Python lists"
        },
        {
            "question": "What is a dictionary in Python?",
            "context": {},
            "description": "Question about Python dictionaries - tests query variations"
        },
        {
            "question": "How do I define a function?",
            "context": {},
            "description": "Question about Python functions"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {test['description']}")
        print(f"{'─'*70}")
        print(f"Question: {test['question']}")
        print(f"Context: {test['context']}")
        
        try:
            result = retriever.query_with_fallback(
                question=test['question'],
                context=test['context'],
                return_intermediate_steps=True
            )
            
            print(f"\n✅ Result:")
            print(f"   Answer: {result.answer}")
            print(f"   Source: {result.source}")
            print(f"   Confidence: {result.confidence:.2%}")
            print(f"   Attempts: {result.attempts}")
            print(f"   Cost: ${result.cost:.4f}")
            
            if result.intermediate_steps:
                print(f"\n   📋 Intermediate Steps:")
                for step in result.intermediate_steps:
                    print(f"      Attempt {step['attempt']}:")
                    print(f"        Query: {step['query'][:60]}...")
                    print(f"        Documents found: {step['documents']}")
                    print(f"        Confidence: {step['confidence']:.2%}")
                    print(f"        Cost: ${step['cost']:.4f}")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # STEP 9: Display Metrics
    # ========================================================================
    print("\n" + "="*70)
    print("📊 METRICS SUMMARY")
    print("="*70)
    stats = metrics.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Average Attempts: {stats['avg_attempts']:.2f}")
    if stats['avg_latency_ms'] > 0:
        print(f"Average Latency: {stats['avg_latency_ms']:.0f}ms")
    
    # ========================================================================
    # STEP 10: Display Cost Report
    # ========================================================================
    print("\n" + "="*70)
    print("💰 COST REPORT")
    print("="*70)
    report = cost_tracker.get_report()
    print(f"Total Cost: ${report['total_cost']:.4f}")
    print(f"Total Tokens: {report['total_tokens']['input']} input, {report['total_tokens']['output']} output")
    if report['budget_remaining'] is not None:
        print(f"Budget Remaining: ${report['budget_remaining']:.4f}")
        print(f"Budget Usage: {report['budget_usage_percent']:.1f}%")
    
    print("\n" + "="*70)
    print("✅ Complete example finished successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

