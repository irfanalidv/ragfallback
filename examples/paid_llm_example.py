"""
Paid LLM Example - Using OpenAI or Anthropic

This example shows how to use paid LLMs (OpenAI, Anthropic) with
open-source vector stores and embeddings.

You can mix and match:
- Paid LLM + Open-source embeddings + Open-source vector store
- Open-source LLM + Paid embeddings + Paid vector store
- Any combination you prefer!
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_openai_llm,
    create_anthropic_llm,
    create_open_source_embeddings,  # Still using open-source embeddings!
    create_faiss_vector_store  # Still using open-source vector store!
)
from langchain.docstore.document import Document


def main():
    """Example using paid LLM with open-source vector store."""
    print("="*60)
    print("ragfallback - Paid LLM Example")
    print("="*60)
    print("\nThis example uses:")
    print("  - OpenAI/Anthropic LLM (paid, requires API key)")
    print("  - HuggingFace embeddings (local, free)")
    print("  - FAISS vector store (local, free)")
    print("\nYou can mix paid and open-source components! 🎉\n")
    
    # Technical documentation content
    documents = [
        Document(
            page_content="REST APIs use HTTP methods: GET to retrieve data, POST to create resources, PUT to update resources, and DELETE to remove resources. API endpoints are URLs that represent resources. Responses typically use JSON format. Status codes indicate success (200) or errors (400, 404, 500).",
            metadata={"source": "rest_api.pdf", "topic": "api"}
        ),
        Document(
            page_content="Authentication methods include API keys, OAuth 2.0, and JWT tokens. API keys are simple strings sent in headers. OAuth 2.0 provides secure authorization flows. JWT tokens are self-contained and include user information. Always use HTTPS for API communication.",
            metadata={"source": "api_auth.pdf", "topic": "security"}
        ),
        Document(
            page_content="Rate limiting prevents API abuse by limiting requests per time period. Common limits are 1000 requests per hour or 100 requests per minute. When exceeded, APIs return 429 Too Many Requests status. Implement exponential backoff for retries.",
            metadata={"source": "rate_limiting.pdf", "topic": "api"}
        ),
        Document(
            page_content="API versioning allows maintaining multiple API versions simultaneously. Common approaches include URL versioning (/api/v1/users) or header versioning (Accept: application/vnd.api+json;version=1). Versioning prevents breaking changes for existing clients.",
            metadata={"source": "api_versioning.pdf", "topic": "api"}
        ),
    ]
    
    # Create open-source embeddings (no API key needed!)
    print("Creating open-source embeddings...")
    embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    
    # Create FAISS vector store (local, free)
    print("Creating FAISS vector store...")
    vector_store = create_faiss_vector_store(
        documents=documents,
        embeddings=embeddings
    )
    
    # Choose your LLM provider:
    print("\nChoose LLM provider:")
    print("1. OpenAI (requires OPENAI_API_KEY)")
    print("2. Anthropic (requires ANTHROPIC_API_KEY)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Use OpenAI (paid)
        print("\nCreating OpenAI LLM...")
        llm = create_openai_llm(
            model="gpt-4o-mini",  # Cheaper option
            temperature=0
        )
    elif choice == "2":
        # Use Anthropic (paid)
        print("\nCreating Anthropic LLM...")
        llm = create_anthropic_llm(
            model="claude-3-haiku-20240307",  # Cheaper option
            temperature=0
        )
    else:
        print("Invalid choice, using OpenAI as default")
        llm = create_openai_llm(model="gpt-4o-mini", temperature=0)
    
    # Setup cost tracking with budget
    cost_tracker = CostTracker(budget=10.0)  # $10 budget
    
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
        question="How do I authenticate API requests?",
        return_intermediate_steps=True,
        enforce_budget=True  # Stop if budget exceeded
    )
    
    # Print results
    print(f"\n✅ Answer: {result.answer}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print(f"📄 Source: {result.source}")
    print(f"🔄 Attempts: {result.attempts}")
    print(f"💰 Cost: ${result.cost:.4f}")
    
    # Print cost report
    print("\n" + "="*60)
    print("Cost Report:")
    print("="*60)
    report = cost_tracker.get_report()
    print(f"Total Cost: ${report['total_cost']:.4f}")
    if report['budget_remaining'] is not None:
        print(f"Budget Remaining: ${report['budget_remaining']:.4f}")
        print(f"Budget Usage: {report['budget_usage_percent']:.1f}%")


if __name__ == "__main__":
    main()

