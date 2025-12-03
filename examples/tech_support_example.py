"""
Technical Support Knowledge Base Example

This example demonstrates ragfallback with a technical support knowledge base.
Customer support Q&A system with fallback strategies.
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document


def main():
    """Technical support knowledge base example."""
    print("="*70)
    print("ragfallback - Technical Support Knowledge Base")
    print("="*70)
    print("\nUse case: Customer support Q&A system")
    print("Uses HuggingFace Inference API (free tier)\n")
    
    # Technical support knowledge base content
    documents = [
        Document(
            page_content="To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and check your inbox for a reset link. The link expires in 24 hours. If you don't receive the email, check your spam folder.",
            metadata={"source": "password_reset.pdf", "category": "account"}
        ),
        Document(
            page_content="Two-factor authentication (2FA) adds an extra layer of security. Enable it in Settings > Security > Two-Factor Authentication. You'll need to install an authenticator app like Google Authenticator or Authy. Scan the QR code and enter the 6-digit code to verify.",
            metadata={"source": "2fa_setup.pdf", "category": "security"}
        ),
        Document(
            page_content="To cancel your subscription, log into your account, go to Billing > Subscriptions, and click Cancel. Your subscription will remain active until the end of the billing period. You'll continue to have access to all features until then. No refunds are provided for partial months.",
            metadata={"source": "cancel_subscription.pdf", "category": "billing"}
        ),
        Document(
            page_content="API rate limits are 1000 requests per hour for free tier accounts and 10000 requests per hour for paid accounts. If you exceed the limit, you'll receive a 429 Too Many Requests error. Wait until the next hour or upgrade your plan for higher limits.",
            metadata={"source": "api_limits.pdf", "category": "technical"}
        ),
        Document(
            page_content="To export your data, go to Settings > Privacy > Download My Data. The export includes all your account information, files, and activity history. Large exports may take up to 48 hours to process. You'll receive an email when your export is ready for download.",
            metadata={"source": "data_export.pdf", "category": "privacy"}
        ),
        Document(
            page_content="Our service supports Windows 10/11, macOS 10.15+, and Linux (Ubuntu 20.04+). Minimum system requirements: 4GB RAM, 10GB free disk space, and an internet connection. For mobile, we support iOS 14+ and Android 8+. Download the app from the App Store or Google Play.",
            metadata={"source": "system_requirements.pdf", "category": "technical"}
        ),
        Document(
            page_content="Payment methods accepted include credit cards (Visa, Mastercard, American Express), debit cards, PayPal, and bank transfers. All payments are processed securely through Stripe. We don't store your full card details. You can update your payment method in Settings > Billing > Payment Methods.",
            metadata={"source": "payment_methods.pdf", "category": "billing"}
        ),
        Document(
            page_content="If you're experiencing slow performance, try clearing your browser cache, disabling browser extensions, or using a different browser. Check your internet connection speed. For mobile apps, ensure you have the latest version installed. If issues persist, contact support with your account details and device information.",
            metadata={"source": "troubleshooting.pdf", "category": "technical"}
        ),
    ]
    
    print(f"📚 Loaded {len(documents)} knowledge base articles\n")
    
    # Setup components
    print("Setting up components...")
    embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    vector_store = create_faiss_vector_store(documents=documents, embeddings=embeddings)
    llm = create_huggingface_llm(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        use_inference_api=True,
        temperature=0
    )
    print("✅ Components ready\n")
    
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
    
    # Customer support questions (various phrasings)
    customer_questions = [
        "How do I reset my password?",
        "I forgot my password, what should I do?",
        "How can I enable two-factor authentication?",
        "I want to cancel my subscription",
        "What payment methods do you accept?",
        "The app is running slow, how do I fix it?",
    ]
    
    print("="*70)
    print("Customer Support Questions")
    print("="*70)
    
    for i, question in enumerate(customer_questions, 1):
        print(f"\n{'─'*70}")
        print(f"Customer Question {i}: {question}")
        print(f"{'─'*70}")
        
        result = retriever.query_with_fallback(
            question=question,
            return_intermediate_steps=True
        )
        
        print(f"\n✅ Support Answer: {result.answer}")
        print(f"📊 Confidence: {result.confidence:.2%}")
        print(f"📄 Source: {result.source}")
        print(f"🔄 Attempts: {result.attempts}")
        
        if result.intermediate_steps and len(result.intermediate_steps) > 1:
            print(f"\n🔄 Fallback used: Query was rephrased {len(result.intermediate_steps)-1} time(s)")
    
    # Summary
    print("\n" + "="*70)
    print("Support System Metrics")
    print("="*70)
    stats = metrics.get_stats()
    print(f"Total Questions: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Average Attempts: {stats['avg_attempts']:.2f}")


if __name__ == "__main__":
    main()

