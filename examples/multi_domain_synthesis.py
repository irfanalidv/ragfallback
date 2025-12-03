"""
Multi-Domain Knowledge Synthesis - Advanced Production Use Case

This example demonstrates ragfallback for complex multi-domain queries requiring:
- Cross-domain knowledge synthesis
- Handling conflicting information
- Priority-based source selection
- Complex reasoning chains
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document


def main():
    """Multi-domain knowledge synthesis for complex queries."""
    print("="*80)
    print("ragfallback - Multi-Domain Knowledge Synthesis")
    print("="*80)
    print("\nScenario: Enterprise knowledge base with multiple domains")
    print("Challenges: Cross-domain queries, conflicting info, priority resolution\n")
    
    # Multi-domain enterprise knowledge base
    documents = [
        # IT/Security Domain
        Document(
            page_content="SECURITY POLICY: All production systems require MFA (Multi-Factor Authentication). SSH access restricted to VPN-only. Database access requires approval from data owner. Secrets managed via HashiCorp Vault. Security incidents must be reported within 1 hour. Regular penetration testing quarterly.",
            metadata={"source": "security_policy.pdf", "domain": "IT", "priority": "critical"}
        ),
        Document(
            page_content="INFRASTRUCTURE: Production runs on AWS (us-east-1, eu-west-1). Auto-scaling enabled for web tier (2-10 instances). Database uses RDS Multi-AZ with automated backups. CDN via CloudFront. Monitoring via CloudWatch and Datadog. Incident response SLA: P1 (critical) 15min, P2 (high) 1hr, P3 (medium) 4hr.",
            metadata={"source": "infrastructure_guide.pdf", "domain": "IT", "priority": "high"}
        ),
        
        # Compliance Domain
        Document(
            page_content="GDPR COMPLIANCE: Personal data processing requires legal basis (consent, contract, legal obligation). Right to access, rectification, erasure must be honored within 30 days. Data Protection Impact Assessment required for high-risk processing. Data breach notification to authorities within 72 hours. Records of processing activities mandatory.",
            metadata={"source": "gdpr_compliance.pdf", "domain": "compliance", "priority": "critical"}
        ),
        Document(
            page_content="SOC 2 REQUIREMENTS: Annual SOC 2 Type II audit required. Access controls must be documented and tested. Change management process must be followed. Vendor risk assessments required for third-party services. Security awareness training mandatory for all employees annually.",
            metadata={"source": "soc2_requirements.pdf", "domain": "compliance", "priority": "high"}
        ),
        
        # Business Operations Domain
        Document(
            page_content="CUSTOMER SUPPORT SLA: Response time targets: Email 4 hours, Chat 2 minutes, Phone 30 seconds. Escalation path: L1 → L2 → L3 → Engineering. Customer satisfaction target: NPS > 50. Refund policy: Full refund within 30 days, partial refund 30-90 days. Support hours: 24/7 for enterprise, business hours for standard.",
            metadata={"source": "support_sla.pdf", "domain": "operations", "priority": "high"}
        ),
        Document(
            page_content="PRICING MODEL: Enterprise tier: $5000/month, includes dedicated support, SLA guarantees, custom integrations. Business tier: $500/month, includes priority support, API access. Standard tier: $50/month, includes email support. Annual contracts receive 20% discount. Volume discounts available for 100+ seats.",
            metadata={"source": "pricing_model.pdf", "domain": "sales", "priority": "medium"}
        ),
        
        # Product Domain
        Document(
            page_content="API RATE LIMITS: Free tier: 1000 requests/day. Standard tier: 10000 requests/day. Enterprise tier: 100000 requests/day with custom limits available. Rate limit headers included in responses. 429 status code when exceeded. Exponential backoff recommended for retries. Webhook rate limits: 1000 events/hour per webhook.",
            metadata={"source": "api_documentation.pdf", "domain": "product", "priority": "high"}
        ),
        Document(
            page_content="FEATURE ROADMAP: Q1: Advanced analytics dashboard, Q2: Custom workflows, Q3: AI-powered insights, Q4: Mobile app. Beta features available to enterprise customers. Feature requests tracked in public roadmap. Deprecation notices provided 90 days in advance.",
            metadata={"source": "product_roadmap.pdf", "domain": "product", "priority": "medium"}
        ),
        
        # Legal Domain
        Document(
            page_content="TERMS OF SERVICE: Service provided 'as-is' with no warranties. Limitation of liability: total liability capped at amount paid in last 12 months. Indemnification: customer indemnifies provider for third-party claims. Dispute resolution: binding arbitration in Delaware. Governing law: Delaware state law.",
            metadata={"source": "terms_of_service.pdf", "domain": "legal", "priority": "critical"}
        ),
        Document(
            page_content="DATA PROCESSING AGREEMENT: Customer is data controller, provider is data processor. Processing activities documented in DPA. Sub-processors require prior approval. Data location: US and EU regions available. Data retention: as specified in customer agreement, minimum 7 years for financial records.",
            metadata={"source": "dpa.pdf", "domain": "legal", "priority": "critical"}
        ),
    ]
    
    print(f"📚 Loaded {len(documents)} documents across multiple domains\n")
    
    # Setup
    embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    vector_store = create_faiss_vector_store(documents=documents, embeddings=embeddings)
    llm = create_huggingface_llm(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        use_inference_api=True,
        temperature=0,
        max_length=1024
    )
    
    cost_tracker = CostTracker(budget=50.0)
    metrics = MetricsCollector()
    
    retriever = AdaptiveRAGRetriever(
        vector_store=vector_store,
        llm=llm,
        embedding_model=embeddings,
        fallback_strategy="query_variations",
        cost_tracker=cost_tracker,
        metrics_collector=metrics,
        max_attempts=5,
        min_confidence=0.80
    )
    print("✅ Multi-domain knowledge system ready\n")
    
    # Complex cross-domain queries
    complex_queries = [
        {
            "question": "What security measures are in place and how do they comply with regulations?",
            "description": "Cross-domain: IT + Compliance"
        },
        {
            "question": "What happens if a customer wants a refund after 30 days?",
            "description": "Cross-domain: Operations + Legal"
        },
        {
            "question": "How does pricing work for enterprise customers with high API usage?",
            "description": "Cross-domain: Sales + Product"
        },
        {
            "question": "What are my data protection obligations and how do I meet them?",
            "description": "Cross-domain: Legal + Compliance + IT"
        },
        {
            "question": "What support can I expect and what are the limitations?",
            "description": "Cross-domain: Operations + Legal + Sales"
        },
    ]
    
    print("="*80)
    print("Multi-Domain Knowledge Synthesis")
    print("="*80)
    
    for i, query_info in enumerate(complex_queries, 1):
        question = query_info["question"]
        description = query_info["description"]
        
        print(f"\n{'─'*80}")
        print(f"Query {i}: {description}")
        print(f"Question: {question}")
        print(f"{'─'*80}")
        
        result = retriever.query_with_fallback(
            question=question,
            return_intermediate_steps=True,
            enforce_budget=True
        )
        
        print(f"\n✅ Synthesized Answer:")
        print(f"   {result.answer}")
        print(f"\n📊 Confidence: {result.confidence:.2%}")
        print(f"📄 Primary Source: {result.source}")
        print(f"🔄 Attempts: {result.attempts}")
        print(f"💰 Cost: ${result.cost:.4f}")
        
        if result.intermediate_steps:
            domains_consulted = set()
            for step in result.intermediate_steps:
                # Extract domain from source if available
                if 'domain' in step.get('metadata', {}):
                    domains_consulted.add(step['metadata']['domain'])
            
            if domains_consulted:
                print(f"\n🌐 Domains Consulted: {', '.join(domains_consulted)}")
    
    # Summary
    print("\n" + "="*80)
    print("Multi-Domain Analysis Summary")
    print("="*80)
    stats = metrics.get_stats()
    report = cost_tracker.get_report()
    
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Average Attempts: {stats['avg_attempts']:.2f}")
    print(f"Total Cost: ${report['total_cost']:.4f}")
    print(f"Budget Remaining: ${report['budget_remaining']:.4f}")


if __name__ == "__main__":
    main()

