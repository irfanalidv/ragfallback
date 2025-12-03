"""
Financial Risk Analysis - Production-Grade Use Case

This example demonstrates ragfallback for financial analysis with:
- Multi-source data synthesis
- Risk assessment queries
- Regulatory compliance checking
- High-stakes decision support
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document


def main():
    """Financial risk analysis with regulatory compliance."""
    print("="*80)
    print("ragfallback - Financial Risk Analysis (Production Use Case)")
    print("="*80)
    print("\nScenario: Financial institution risk assessment")
    print("Challenges: Regulatory compliance, multi-factor analysis, risk quantification\n")
    
    # Financial and regulatory content
    documents = [
        Document(
            page_content="BASEL III REQUIREMENTS: Tier 1 capital ratio must be at least 6%. Total capital ratio minimum 8%. Leverage ratio minimum 3%. Banks must maintain capital conservation buffer of 2.5%. Countercyclical buffer ranges from 0-2.5% based on credit growth. Non-compliance triggers restrictions on distributions.",
            metadata={"source": "basel_iii_regulations.pdf", "type": "regulation", "priority": "high"}
        ),
        Document(
            page_content="RISK ASSESSMENT FRAMEWORK: Credit risk assessed using PD (Probability of Default), LGD (Loss Given Default), and EAD (Exposure at Default). Market risk measured via VaR (Value at Risk) with 99% confidence, 10-day holding period. Operational risk includes fraud, system failures, legal risks. Liquidity risk measured via LCR (Liquidity Coverage Ratio) minimum 100%.",
            metadata={"source": "risk_framework.pdf", "type": "policy", "priority": "high"}
        ),
        Document(
            page_content="STRESS TESTING REQUIREMENTS: Annual stress tests required for banks with assets over $50B. Scenarios include: severe recession (GDP -8%, unemployment 12%), market crash (equity -50%, credit spreads +300bps), and combined scenario. Capital adequacy must be maintained under all scenarios. Results reported to regulators quarterly.",
            metadata={"source": "stress_testing_requirements.pdf", "type": "regulation", "priority": "high"}
        ),
        Document(
            page_content="CREDIT RISK MITIGATION: Collateral reduces exposure by haircut percentage: government bonds 0%, corporate bonds 15-50%, equities 15-50%, real estate 15-40%. Guarantees from eligible guarantors reduce risk weight. Credit derivatives can transfer risk but require proper documentation. Netting agreements reduce exposure for offsetting positions.",
            metadata={"source": "credit_mitigation.pdf", "type": "policy", "priority": "medium"}
        ),
        Document(
            page_content="OPERATIONAL RISK EVENTS: High-severity events (>$10M loss) require immediate reporting to board and regulators within 24 hours. Root cause analysis must be completed within 30 days. Remediation plans required within 60 days. Annual operational risk loss data collection mandatory. Key risk indicators monitored monthly.",
            metadata={"source": "operational_risk_policy.pdf", "type": "policy", "priority": "high"}
        ),
        Document(
            page_content="LIQUIDITY RISK MANAGEMENT: LCR (Liquidity Coverage Ratio) requires high-quality liquid assets to cover net cash outflows over 30 days. NSFR (Net Stable Funding Ratio) ensures stable funding over 1 year. Contingency funding plan must be tested quarterly. Intraday liquidity monitoring required. Central bank facilities available as backstop.",
            metadata={"source": "liquidity_management.pdf", "type": "policy", "priority": "high"}
        ),
        Document(
            page_content="REGULATORY REPORTING: Call reports filed quarterly with detailed balance sheet, income statement, risk metrics. FFIEC 031 for banks, FFIEC 041 for smaller institutions. Data must be accurate and submitted within 30 days of quarter end. Errors subject to penalties. Automated reporting systems recommended for large institutions.",
            metadata={"source": "regulatory_reporting.pdf", "type": "regulation", "priority": "high"}
        ),
        Document(
            page_content="CAPITAL PLANNING: Capital plans must cover 9-quarter horizon, updated annually. Scenarios include baseline, adverse, and severely adverse. Capital actions (dividends, buybacks) must be justified. CCAR (Comprehensive Capital Analysis and Review) required for large banks. Failure results in restrictions on capital distributions.",
            metadata={"source": "capital_planning.pdf", "type": "regulation", "priority": "high"}
        ),
    ]
    
    print(f"📚 Loaded {len(documents)} financial and regulatory documents\n")
    
    # Setup with strict requirements
    embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    vector_store = create_faiss_vector_store(documents=documents, embeddings=embeddings)
    llm = create_huggingface_llm(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        use_inference_api=True,
        temperature=0,
        max_length=1024
    )
    
    # Budget tracking for production use
    cost_tracker = CostTracker(budget=100.0)  # $100 budget for analysis session
    metrics = MetricsCollector()
    
    # Very high confidence for financial decisions
    retriever = AdaptiveRAGRetriever(
        vector_store=vector_store,
        llm=llm,
        embedding_model=embeddings,
        fallback_strategy="query_variations",
        cost_tracker=cost_tracker,
        metrics_collector=metrics,
        max_attempts=5,
        min_confidence=0.88  # High threshold for financial accuracy
    )
    print("✅ Financial analysis system ready\n")
    
    # Complex financial risk queries
    risk_queries = [
        {
            "question": "What are the capital requirements I need to meet?",
            "description": "Regulatory compliance query"
        },
        {
            "question": "How do I calculate my risk exposure?",
            "description": "Risk quantification query"
        },
        {
            "question": "What happens if I fail a stress test?",
            "description": "Consequence analysis"
        },
        {
            "question": "How can I reduce my credit risk?",
            "description": "Risk mitigation strategies"
        },
        {
            "question": "What reporting requirements do I have?",
            "description": "Regulatory compliance"
        },
    ]
    
    print("="*80)
    print("Financial Risk Analysis Queries")
    print("="*80)
    
    for i, query_info in enumerate(risk_queries, 1):
        question = query_info["question"]
        description = query_info["description"]
        
        print(f"\n{'─'*80}")
        print(f"Analysis Query {i}: {description}")
        print(f"Question: {question}")
        print(f"{'─'*80}")
        
        result = retriever.query_with_fallback(
            question=question,
            return_intermediate_steps=True,
            enforce_budget=True  # Stop if budget exceeded
        )
        
        print(f"\n✅ Risk Analysis:")
        print(f"   {result.answer}")
        print(f"\n📊 Confidence: {result.confidence:.2%}")
        print(f"📄 Source: {result.source}")
        print(f"💰 Cost: ${result.cost:.4f}")
        print(f"🔄 Attempts: {result.attempts}")
        
        if result.intermediate_steps:
            print(f"\n🔄 Query Refinement:")
            for step in result.intermediate_steps[:3]:  # Show first 3
                print(f"   Attempt {step['attempt']}: '{step['query'][:65]}...'")
        
        if result.confidence < 0.88:
            print(f"\n⚠️  WARNING: Lower confidence. Regulatory review recommended.")
    
    # Summary
    print("\n" + "="*80)
    print("Financial Analysis Summary")
    print("="*80)
    stats = metrics.get_stats()
    report = cost_tracker.get_report()
    
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Total Cost: ${report['total_cost']:.4f}")
    print(f"Budget Remaining: ${report['budget_remaining']:.4f}")
    print(f"\n💡 Financial Disclaimer: This is for analysis purposes only.")
    print(f"   All financial decisions should be reviewed by qualified professionals.")


if __name__ == "__main__":
    main()

