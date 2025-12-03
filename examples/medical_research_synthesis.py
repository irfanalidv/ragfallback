"""
Medical Research Synthesis - Advanced Multi-Document Analysis

This example demonstrates ragfallback for medical research with:
- Multi-document synthesis
- Handling conflicting information
- Evidence-based answers
- Source citation requirements
"""

from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document


def main():
    """Medical research synthesis with evidence-based answers."""
    print("="*80)
    print("ragfallback - Medical Research Synthesis (Advanced Use Case)")
    print("="*80)
    print("\nScenario: Synthesizing medical research papers")
    print("Challenges: Conflicting studies, evidence levels, source attribution\n")
    
    # Medical research content (simplified for example)
    documents = [
        Document(
            page_content="STUDY 1 (2023, n=500): Metformin reduces HbA1c by 1.2% on average in Type 2 diabetes patients over 6 months. Side effects include gastrointestinal issues in 20% of patients. Study limitations: single-center, short follow-up period. Evidence level: Moderate.",
            metadata={"source": "metformin_study_2023.pdf", "year": "2023", "n": "500", "evidence": "moderate"}
        ),
        Document(
            page_content="STUDY 2 (2022, n=1200): Comprehensive analysis shows metformin effectiveness varies by patient age. Patients under 60 show 1.5% HbA1c reduction, while patients over 60 show 0.8% reduction. No significant cardiovascular risks observed. Evidence level: High.",
            metadata={"source": "metformin_meta_2022.pdf", "year": "2022", "n": "1200", "evidence": "high"}
        ),
        Document(
            page_content="STUDY 3 (2024, n=300): Long-term metformin use (5+ years) associated with vitamin B12 deficiency in 30% of patients. Regular monitoring recommended. No association with cognitive decline found. Evidence level: Moderate.",
            metadata={"source": "metformin_longterm_2024.pdf", "year": "2024", "n": "300", "evidence": "moderate"}
        ),
        Document(
            page_content="GUIDELINE (2023): ADA recommends metformin as first-line therapy for Type 2 diabetes unless contraindicated. Starting dose 500mg twice daily, titrate to 2000mg daily. Monitor renal function and B12 levels. Contraindications: severe renal impairment, metabolic acidosis.",
            metadata={"source": "ada_guidelines_2023.pdf", "type": "guideline", "evidence": "high"}
        ),
        Document(
            page_content="STUDY 4 (2023, n=800): Combination therapy (metformin + SGLT2 inhibitor) shows superior HbA1c reduction (1.8%) compared to metformin alone (1.1%). Cardiovascular benefits observed in combination group. Evidence level: High.",
            metadata={"source": "combination_therapy_2023.pdf", "year": "2023", "n": "800", "evidence": "high"}
        ),
        Document(
            page_content="REVIEW (2024): Systematic review of 15 studies confirms metformin safety profile. Most common side effects: diarrhea (10-15%), nausea (5-10%), metallic taste (5%). Rare but serious: lactic acidosis in patients with renal impairment. Evidence level: High.",
            metadata={"source": "metformin_safety_review_2024.pdf", "year": "2024", "type": "review", "evidence": "high"}
        ),
    ]
    
    print(f"📚 Loaded {len(documents)} medical research documents\n")
    
    # Setup with higher precision requirements
    embeddings = create_open_source_embeddings(model_name="all-MiniLM-L6-v2")
    vector_store = create_faiss_vector_store(documents=documents, embeddings=embeddings)
    llm = create_huggingface_llm(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        use_inference_api=True,
        temperature=0,
        max_length=1024
    )
    
    cost_tracker = CostTracker()
    metrics = MetricsCollector()
    
    # High confidence threshold for medical accuracy
    retriever = AdaptiveRAGRetriever(
        vector_store=vector_store,
        llm=llm,
        embedding_model=embeddings,
        fallback_strategy="query_variations",
        cost_tracker=cost_tracker,
        metrics_collector=metrics,
        max_attempts=5,
        min_confidence=0.90  # Very high threshold for medical
    )
    print("✅ Research synthesis system ready\n")
    
    # Complex medical research queries
    research_queries = [
        {
            "question": "What is the effectiveness of metformin for diabetes?",
            "description": "Requires synthesizing multiple studies"
        },
        {
            "question": "Are there any long-term side effects I should know about?",
            "description": "Needs to find long-term safety data"
        },
        {
            "question": "What do the guidelines recommend?",
            "description": "Should prioritize guideline documents"
        },
        {
            "question": "Is metformin safe for elderly patients?",
            "description": "Requires age-specific analysis"
        },
        {
            "question": "What about combining metformin with other medications?",
            "description": "Needs combination therapy information"
        },
    ]
    
    print("="*80)
    print("Medical Research Synthesis")
    print("="*80)
    
    for i, query_info in enumerate(research_queries, 1):
        question = query_info["question"]
        description = query_info["description"]
        
        print(f"\n{'─'*80}")
        print(f"Research Question {i}: {description}")
        print(f"Question: {question}")
        print(f"{'─'*80}")
        
        result = retriever.query_with_fallback(
            question=question,
            return_intermediate_steps=True
        )
        
        print(f"\n✅ Evidence-Based Answer:")
        print(f"   {result.answer}")
        print(f"\n📊 Confidence: {result.confidence:.2%}")
        print(f"📄 Primary Source: {result.source}")
        print(f"🔄 Attempts: {result.attempts}")
        
        if result.intermediate_steps:
            print(f"\n📚 Sources Consulted:")
            sources = set()
            for step in result.intermediate_steps:
                if 'source' in step:
                    sources.add(step.get('source', 'Unknown'))
            for source in sources:
                print(f"   - {source}")
        
        if result.confidence < 0.90:
            print(f"\n⚠️  CAUTION: Lower confidence. Multiple sources should be consulted.")
    
    # Summary
    print("\n" + "="*80)
    print("Research Synthesis Summary")
    print("="*80)
    stats = metrics.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"\n💡 Medical Disclaimer: This is for research purposes only.")
    print(f"   Always consult healthcare professionals for medical decisions.")


if __name__ == "__main__":
    main()

