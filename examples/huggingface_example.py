"""
HuggingFace Transformers Example

This example shows how to use HuggingFace transformers with ragfallback.
Two options:
1. HuggingFace Inference API (easier, free tier available, no local installation)
2. Local HuggingFace models (requires transformers and torch)
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document


def main():
    """HuggingFace transformers example."""
    print("="*60)
    print("ragfallback - HuggingFace Transformers Example")
    print("="*60)
    print("\nThis example uses:")
    print("  - HuggingFace LLM (via Inference API or local)")
    print("  - HuggingFace embeddings (local, free)")
    print("  - FAISS vector store (local, free)")
    print("\nNo API keys required for Inference API free tier! 🎉\n")
    
    # Machine learning documentation content
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions. Common types include supervised learning, unsupervised learning, and reinforcement learning.",
            metadata={"source": "ml_intro.pdf", "topic": "introduction"}
        ),
        Document(
            page_content="Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. The input layer receives data, hidden layers process it, and the output layer produces results. Training involves adjusting weights through backpropagation.",
            metadata={"source": "neural_networks.pdf", "topic": "deep_learning"}
        ),
        Document(
            page_content="Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. Applications include language translation, sentiment analysis, chatbots, and text summarization. Modern NLP uses transformer models like BERT and GPT that can understand context and meaning.",
            metadata={"source": "nlp_basics.pdf", "topic": "nlp"}
        ),
        Document(
            page_content="Vector embeddings represent words, sentences, or documents as dense numerical vectors in high-dimensional space. Similar words have similar vectors. Embeddings capture semantic meaning and relationships. Popular embedding models include Word2Vec, GloVe, and modern transformer-based embeddings like sentence-transformers.",
            metadata={"source": "embeddings.pdf", "topic": "embeddings"}
        ),
    ]
    
    # Create HuggingFace embeddings (local, free)
    print("Creating HuggingFace embeddings...")
    embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    
    # Create FAISS vector store
    print("Creating FAISS vector store...")
    vector_store = create_faiss_vector_store(
        documents=documents,
        embeddings=embeddings
    )
    
    # Choose HuggingFace LLM mode
    print("\nChoose HuggingFace mode:")
    print("1. Inference API (easier, free tier, no local installation)")
    print("2. Local model (requires transformers and torch)")
    
    choice = input("Enter choice (1 or 2, default: 1): ").strip() or "1"
    
    if choice == "1":
        # Use HuggingFace Inference API (easier, free tier available)
        print("\nUsing HuggingFace Inference API...")
        print("Note: Free tier available, no API key needed for public models!")
        
        llm = create_huggingface_llm(
            model_id="mistralai/Mistral-7B-Instruct-v0.1",  # Popular model
            use_inference_api=True,  # Use Inference API
            temperature=0,
            max_length=512
        )
        print("✅ Using HuggingFace Inference API (free tier)")
    else:
        # Use local HuggingFace model
        print("\nUsing local HuggingFace model...")
        print("Note: This requires transformers and torch installed.")
        print("      First run will download the model (~7GB for Mistral-7B).")
        
        try:
            llm = create_huggingface_llm(
                model_id="google/flan-t5-base",  # Smaller model for demo
                use_inference_api=False,  # Load locally
                device="cpu",  # Use "cuda" if GPU available
                temperature=0,
                max_length=512
            )
            print("✅ Using local HuggingFace model")
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            print("Falling back to Inference API...")
            llm = create_huggingface_llm(
                model_id="mistralai/Mistral-7B-Instruct-v0.1",
                use_inference_api=True,
                temperature=0
            )
    
    # Setup cost tracking
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
        question="What is machine learning?",
        return_intermediate_steps=True
    )
    
    # Print results
    print(f"\n✅ Answer: {result.answer}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print(f"📄 Source: {result.source}")
    print(f"🔄 Attempts: {result.attempts}")
    print(f"💰 Cost: ${result.cost:.4f} (Free with Inference API!)")
    
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

