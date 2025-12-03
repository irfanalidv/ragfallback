"""
Legal Document Analysis - Advanced Use Case

This example demonstrates ragfallback for legal document analysis with:
- Complex multi-part queries
- Ambiguous question handling
- Cross-document synthesis
- Confidence-based fallback for critical decisions
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document


def main():
    """Legal document analysis with complex queries."""
    print("="*80)
    print("ragfallback - Legal Document Analysis (Advanced Use Case)")
    print("="*80)
    print("\nScenario: Analyzing legal contracts and regulations")
    print("Challenges: Ambiguous queries, multi-part questions, cross-references\n")
    
    # Legal document content (contract terms, regulations)
    documents = [
        Document(
            page_content="TERMINATION CLAUSE: Either party may terminate this agreement with 30 days written notice. Upon termination, all outstanding invoices become immediately due. Confidential information must be returned or destroyed within 15 days. Non-compete provisions remain in effect for 12 months post-termination.",
            metadata={"source": "service_agreement.pdf", "section": "termination", "type": "contract"}
        ),
        Document(
            page_content="DATA PROTECTION: Customer data must be encrypted at rest and in transit using AES-256. Data retention period is 7 years for financial records, 3 years for general records. GDPR compliance requires explicit consent for data processing. Right to deletion must be honored within 30 days of request.",
            metadata={"source": "data_protection_policy.pdf", "section": "privacy", "type": "regulation"}
        ),
        Document(
            page_content="LIABILITY LIMITATIONS: Service provider's total liability shall not exceed the amount paid by customer in the 12 months preceding the claim. Exclusions include indirect damages, lost profits, and consequential damages. Force majeure events exempt both parties from performance obligations.",
            metadata={"source": "service_agreement.pdf", "section": "liability", "type": "contract"}
        ),
        Document(
            page_content="INTELLECTUAL PROPERTY: All pre-existing IP remains with original owner. Work product created during engagement becomes customer property. Service provider retains rights to general knowledge and methodologies. Disputes resolved through binding arbitration in accordance with AAA rules.",
            metadata={"source": "service_agreement.pdf", "section": "ip", "type": "contract"}
        ),
        Document(
            page_content="PAYMENT TERMS: Invoices are due net 30 days. Late payments incur 1.5% monthly interest. Service suspension occurs after 60 days of non-payment. Reinstatement requires payment of all outstanding amounts plus reinstatement fee of $500. Refunds only available within first 30 days of service.",
            metadata={"source": "service_agreement.pdf", "section": "payment", "type": "contract"}
        ),
        Document(
            page_content="COMPLIANCE REQUIREMENTS: SOC 2 Type II certification required annually. Regular security audits must be conducted quarterly. Incident response plan must be tested semi-annually. Compliance violations may result in immediate termination and potential legal action. Regulatory changes require contract amendments within 90 days.",
            metadata={"source": "compliance_requirements.pdf", "section": "compliance", "type": "regulation"}
        ),
        Document(
            page_content="SERVICE LEVEL AGREEMENTS: Uptime guarantee of 99.9% measured monthly. Scheduled maintenance excluded from uptime calculation. Credits issued for downtime exceeding SLA thresholds: 5% credit for 99.0-99.9% uptime, 10% for 98.0-98.9%, 25% for below 98%. Credits must be requested within 30 days.",
            metadata={"source": "sla_document.pdf", "section": "sla", "type": "contract"}
        ),
        Document(
            page_content="DISPUTE RESOLUTION: All disputes subject to mediation before arbitration. Mediation must be completed within 60 days. Arbitration conducted under AAA Commercial Rules. Each party bears own costs. Class action waivers are enforceable. Governing law is State of Delaware. Venue is Wilmington, Delaware.",
            metadata={"source": "service_agreement.pdf", "section": "disputes", "type": "contract"}
        ),
    ]
    
    print(f"📚 Loaded {len(documents)} legal documents\n")
    
    # Setup components
    print("Setting up advanced RAG system...")
    embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    vector_store = create_faiss_vector_store(documents=documents, embeddings=embeddings)
    llm = create_huggingface_llm(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        use_inference_api=True,
        temperature=0,
        max_length=1024  # Longer for complex legal queries
    )
    
    # Higher confidence threshold for legal accuracy
    cost_tracker = CostTracker()
    metrics = MetricsCollector()
    
    retriever = AdaptiveRAGRetriever(
        vector_store=vector_store,
        llm=llm,
        embedding_model=embeddings,
        fallback_strategy="query_variations",
        cost_tracker=cost_tracker,
        metrics_collector=metrics,
        max_attempts=5,  # More attempts for complex queries
        min_confidence=0.85  # Higher threshold for legal accuracy
    )
    print("✅ System ready\n")
    
    # Complex legal queries
    complex_queries = [
        {
            "question": "What happens if I want to cancel the contract?",
            "description": "Ambiguous query - needs to find termination clause"
        },
        {
            "question": "How long do I have to pay after receiving an invoice?",
            "description": "Specific payment terms query"
        },
        {
            "question": "What are my rights regarding data deletion?",
            "description": "Cross-referencing privacy and contract terms"
        },
        {
            "question": "What happens if the service goes down?",
            "description": "Needs to find SLA and liability sections"
        },
        {
            "question": "Can I sue if something goes wrong?",
            "description": "Complex query requiring dispute resolution and liability analysis"
        },
    ]
    
    print("="*80)
    print("Complex Legal Queries Analysis")
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
            return_intermediate_steps=True
        )
        
        print(f"\n✅ Legal Analysis:")
        print(f"   {result.answer}")
        print(f"\n📊 Confidence: {result.confidence:.2%}")
        print(f"📄 Source: {result.source}")
        print(f"🔄 Attempts: {result.attempts}")
        
        if result.intermediate_steps:
            print(f"\n🔄 Query Evolution:")
            for step in result.intermediate_steps:
                print(f"   Attempt {step['attempt']}: '{step['query'][:70]}...' (Confidence: {step['confidence']:.2%})")
        
        # Warning for low confidence in legal context
        if result.confidence < 0.85:
            print(f"\n⚠️  WARNING: Low confidence for legal query. Manual review recommended.")
    
    # Summary
    print("\n" + "="*80)
    print("Legal Analysis Summary")
    print("="*80)
    stats = metrics.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Average Attempts: {stats['avg_attempts']:.2f}")
    print(f"\n💡 Note: Legal queries require high confidence thresholds")
    print(f"   Low confidence results should be reviewed by legal professionals")


if __name__ == "__main__":
    main()

