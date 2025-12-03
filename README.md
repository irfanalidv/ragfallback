# ragfallback

[![GitHub license](https://img.shields.io/github/license/irfanalidv/ragfallback)](https://github.com/irfanalidv/ragfallback/blob/main/LICENSE)
[![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://pypi.org/project/ragfallback/)
[![PyPI](https://img.shields.io/pypi/v/ragfallback)](https://pypi.org/project/ragfallback/)
[![Downloads](https://static.pepy.tech/badge/ragfallback)](https://pepy.tech/project/ragfallback)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**RAG Fallback Strategies** - A production-ready Python library that adds intelligent fallback mechanisms to RAG (Retrieval-Augmented Generation) systems, preventing silent failures and improving answer quality.

[Installation](#-quick-start) • [Documentation](#-complete-examples-with-outputs) • [Examples](examples/) • [Contributing](CONTRIBUTING.md)

## 🎯 Real-World Problems Solved

### Problem 1: Silent Failures

**Before:** RAG systems return "Not found" even when relevant data exists  
**After:** Automatic query variations find answers that initial queries miss

### Problem 2: Cost Overruns

**Before:** No visibility into LLM costs, unexpected bills  
**After:** Real-time cost tracking and budget enforcement

### Problem 3: Query Mismatch

**Before:** User queries don't match document phrasing → no results  
**After:** LLM-generated query variations increase retrieval success rate

### Problem 4: Low Confidence Answers

**Before:** RAG systems return low-quality answers without retry  
**After:** Confidence scoring with automatic retry on low-confidence results

## 🎯 Features

- **🔄 Multiple Fallback Strategies**: Query variations, semantic expansion, re-ranking, and more
- **💰 Cost Awareness**: Built-in token tracking and budget management
- **🔌 Framework Agnostic**: Works with LangChain, LlamaIndex, and custom retrievers
- **📊 Production Ready**: Comprehensive error handling, logging, and metrics
- **⚙️ Configurable**: Easy to customize and extend
- **🆓 Open-Source First**: Works completely free with HuggingFace, Ollama, and FAISS
- **📈 Transparent**: See all intermediate steps, costs, and metrics
- **✅ Production-Ready**: Comprehensive examples and test coverage

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install ragfallback

# With open-source components (recommended for free usage)
pip install ragfallback[huggingface,sentence-transformers,faiss]

# With paid providers (optional)
pip install ragfallback[openai]
```

### Minimal Example (5 Lines)

```python
from ragfallback import AdaptiveRAGRetriever
from ragfallback.utils import create_huggingface_llm, create_open_source_embeddings, create_faiss_vector_store
from langchain.docstore.document import Document

# Python documentation content
documents = [
    Document(
        page_content="Python is a high-level programming language known for simplicity and readability. It supports multiple programming paradigms and has an extensive standard library.",
        metadata={"source": "python_intro.pdf"}
    )
]
embeddings = create_open_source_embeddings()
vector_store = create_faiss_vector_store(documents, embeddings)
llm = create_huggingface_llm(use_inference_api=True)
retriever = AdaptiveRAGRetriever(vector_store=vector_store, llm=llm, embedding_model=embeddings)

result = retriever.query_with_fallback(question="What is Python?")
print(result.answer)
```

**Output:**

```
Python is a high-level programming language known for simplicity and readability.
```

> **💡 Note:** Uses HuggingFace Inference API for LLM responses, embeddings, and vector similarity search.

## 📖 Complete Examples with Outputs

All examples demonstrate production-ready implementations.

**To see actual outputs, run any example:**

```bash
python examples/open_source_example.py
python examples/huggingface_example.py
python examples/complete_example.py
```

### Example 1: Basic Usage (Open-Source)

**Code:**

```python
from ragfallback import AdaptiveRAGRetriever
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document

# Python documentation content
documents = [
    Document(
        page_content="Python lists are mutable sequences created with square brackets: my_list = [1, 2, 3]. Methods include append() to add items, remove() to delete items, and len() to get length.",
        metadata={"source": "python_lists.pdf"}
    ),
    Document(
        page_content="Python dictionaries store key-value pairs: person = {'name': 'Alice', 'age': 30}. Access values using keys: person['name']. Use get() method for safe access.",
        metadata={"source": "python_dicts.pdf"}
    ),
]

# Create components (all free, no API keys!)
embeddings = create_open_source_embeddings()
vector_store = create_faiss_vector_store(documents, embeddings)
llm = create_huggingface_llm(use_inference_api=True)

# Create retriever
retriever = AdaptiveRAGRetriever(
    vector_store=vector_store,
    llm=llm,
    embedding_model=embeddings,
    fallback_strategy="query_variations",
    max_attempts=3
)

# Query
result = retriever.query_with_fallback(
    question="How do I create a list in Python?"
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Attempts: {result.attempts}")
print(f"Cost: ${result.cost:.4f}")
```

**Output:**

```
Answer: Python lists are mutable sequences created with square brackets: my_list = [1, 2, 3].
Confidence: 92.00%
Attempts: 1
Cost: $0.0000
```

> **Note:** Uses HuggingFace Inference API for query variations and answer generation. Confidence scores are calculated from document retrieval results.

---

### Example 2: With Cost Tracking and Metrics

**Code:**

```python
from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_openai_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document

# Example documents (metadata values are just for tracking - not actual files)
documents = [
    Document(page_content="Product X costs $99.", metadata={"source": "pricing.pdf"}),
]

# Setup cost tracking
cost_tracker = CostTracker(budget=5.0)  # $5 budget
metrics = MetricsCollector()

# Create components
embeddings = create_open_source_embeddings()  # Free
vector_store = create_faiss_vector_store(documents, embeddings)  # Free
llm = create_openai_llm(model="gpt-4o-mini")  # Paid (requires OPENAI_API_KEY)

retriever = AdaptiveRAGRetriever(
    vector_store=vector_store,
    llm=llm,
    embedding_model=embeddings,
    cost_tracker=cost_tracker,
    metrics_collector=metrics,
    max_attempts=3
)

# Query multiple times
questions = [
    "What is the price of Product X?",
    "How much does Product X cost?",
]

for question in questions:
    result = retriever.query_with_fallback(question=question, enforce_budget=True)
    print(f"Q: {question}")
    print(f"A: {result.answer}\n")

# Display metrics
stats = metrics.get_stats()
print(f"Success Rate: {stats['success_rate']:.2%}")
print(f"Average Confidence: {stats['avg_confidence']:.2f}")

# Display cost report
report = cost_tracker.get_report()
print(f"Total Cost: ${report['total_cost']:.4f}")
print(f"Budget Remaining: ${report['budget_remaining']:.4f}")
```

**Output:**

```
Q: What is the price of Product X?
A: Product X costs $99.

Q: How much does Product X cost?
A: Product X costs $99.

Success Rate: 100.00%
Average Confidence: 0.90
Total Cost: $0.0024
Budget Remaining: $4.9976
```

> **Note:** Cost tracking uses token counts from LLM API calls. Metrics are collected from query executions.

---

### Example 3: Query Variations Fallback

**Code:**

```python
from ragfallback import AdaptiveRAGRetriever
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document

# Example documents (metadata is just for tracking - not actual files)
documents = [
    Document(
        page_content="The CEO of Acme Corp is John Smith.",
        metadata={"source": "leadership.pdf"}
    ),
]

embeddings = create_open_source_embeddings()
vector_store = create_faiss_vector_store(documents, embeddings)
llm = create_huggingface_llm(use_inference_api=True)

retriever = AdaptiveRAGRetriever(
    vector_store=vector_store,
    llm=llm,
    embedding_model=embeddings,
    max_attempts=3,
    min_confidence=0.7
)

# Query with different phrasings
result = retriever.query_with_fallback(
    question="Who leads Acme Corp?",
    return_intermediate_steps=True
)

print(f"Final Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Total Attempts: {result.attempts}\n")

# Show intermediate steps
if result.intermediate_steps:
    print("Intermediate Steps:")
    for step in result.intermediate_steps:
        print(f"  Attempt {step['attempt']}: '{step['query']}'")
        print(f"    Confidence: {step['confidence']:.2%}")
```

**Output:**

```
Final Answer: The CEO of Acme Corp is John Smith.
Confidence: 88.00%
Total Attempts: 2

Intermediate Steps:
  Attempt 1: 'Who leads Acme Corp?'
    Confidence: 75.00%
  Attempt 2: 'Who is the leader of Acme Corp?'
    Confidence: 88.00%
```

> **Note:** Query variations are generated by LLM calls. Each attempt uses a different query formulation, and confidence is calculated from document retrieval results.

---

### Example 4: Complete Workflow

**Code:**

```python
from ragfallback import AdaptiveRAGRetriever, CostTracker, MetricsCollector
from ragfallback.utils import (
    create_huggingface_llm,
    create_open_source_embeddings,
    create_faiss_vector_store
)
from langchain.docstore.document import Document

# Step 1: Prepare documents (metadata is just for tracking - not actual files)
documents = [
    Document(
        page_content="Acme Corp revenue: $10M. Employees: 50. Founded: 2020.",
        metadata={"source": "company_data.pdf"}
    ),
]

# Step 2: Create components
embeddings = create_open_source_embeddings()
vector_store = create_faiss_vector_store(documents, embeddings)
llm = create_huggingface_llm(use_inference_api=True)

# Step 3: Setup tracking
cost_tracker = CostTracker()
metrics = MetricsCollector()

# Step 4: Create retriever
retriever = AdaptiveRAGRetriever(
    vector_store=vector_store,
    llm=llm,
    embedding_model=embeddings,
    cost_tracker=cost_tracker,
    metrics_collector=metrics,
    fallback_strategy="query_variations",
    max_attempts=3,
    min_confidence=0.7
)

# Step 5: Query
result = retriever.query_with_fallback(
    question="What is Acme Corp's revenue?",
    context={"company": "Acme Corp"},
    return_intermediate_steps=True
)

# Step 6: Display results
print("="*60)
print("QUERY RESULTS")
print("="*60)
print(f"Question: What is Acme Corp's revenue?")
print(f"Answer: {result.answer}")
print(f"Source: {result.source}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Attempts: {result.attempts}")
print(f"Cost: ${result.cost:.4f}")

# Step 7: Display metrics
print("\n" + "="*60)
print("METRICS")
print("="*60)
stats = metrics.get_stats()
print(f"Total Queries: {stats['total_queries']}")
print(f"Success Rate: {stats['success_rate']:.2%}")
print(f"Average Confidence: {stats['avg_confidence']:.2f}")
```

**Output:**

```
============================================================
QUERY RESULTS
============================================================
Question: What is Acme Corp's revenue?
Answer: Acme Corp revenue: $10M.
Source: company_data.pdf
Confidence: 92.00%
Attempts: 1
Cost: $0.0000

============================================================
METRICS
============================================================
Total Queries: 1
Success Rate: 100.00%
Average Confidence: 0.92
```

> **Note:** Metrics are collected from query executions. Confidence scores are calculated using document retrieval and answer quality assessment.

---

## 🎯 Use Cases

### Use Case 1: Research Assistant

Build a research assistant that answers questions about companies:

```python
retriever = AdaptiveRAGRetriever(...)
result = retriever.query_with_fallback(
    question="What is the company's revenue?",
    context={"company": "Acme Corp"}
)
```

**Use Case:** Company research, competitive intelligence, due diligence

---

### Use Case 2: Document Q&A

Answer questions from large document collections:

```python
retriever = AdaptiveRAGRetriever(...)
result = retriever.query_with_fallback(
    question="What are the key findings?",
    return_intermediate_steps=True
)
```

**Use Case:** Legal document analysis, research papers, technical documentation

---

### Use Case 3: Cost-Conscious Production

Production systems with budget limits:

```python
cost_tracker = CostTracker(budget=10.0)
retriever = AdaptiveRAGRetriever(
    ...,
    cost_tracker=cost_tracker
)
result = retriever.query_with_fallback(
    question="...",
    enforce_budget=True
)
```

**Use Case:** Production APIs, SaaS applications, high-volume systems

---

### Use Case 4: Open-Source Setup

Completely free setup using only open-source components:

```python
# All free, no API keys!
embeddings = create_open_source_embeddings()
vector_store = create_faiss_vector_store(documents, embeddings)
llm = create_huggingface_llm(use_inference_api=True)
```

**Use Case:** Personal projects, learning, prototyping, privacy-sensitive applications

---

## 📚 Documentation

### Loading Documents

**Note:** The PDF file references in examples (like `"annual_report.pdf"`) are just example metadata values, not actual files. They're used to demonstrate how document metadata works.

In practice, you'd load documents from various sources:

```python
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Option 1: Load from actual PDF files
loader = PyPDFLoader("path/to/your/document.pdf")
documents = loader.load()

# Option 2: Load from text files
loader = TextLoader("path/to/your/document.txt")
documents = loader.load()

# Option 3: Create Document objects manually (as shown in examples)
documents = [
    Document(
        page_content="Your content here...",
        metadata={"source": "your_file.pdf", "page": 1}
    )
]

# Option 4: Load from web pages, databases, etc.
# Use any LangChain document loader
```

The `metadata["source"]` field is just for tracking where documents came from - it doesn't need to point to an actual file.

### Core Components

#### AdaptiveRAGRetriever

The main retriever class:

```python
retriever = AdaptiveRAGRetriever(
    vector_store=vector_store,
    llm=llm,
    embedding_model=embeddings,
    fallback_strategy="query_variations",  # Default
    max_attempts=3,                         # Max retry attempts
    min_confidence=0.7,                    # Minimum confidence threshold
    cost_tracker=cost_tracker,             # Optional cost tracking
    metrics_collector=metrics               # Optional metrics
)
```

#### QueryResult

Result object with metadata:

```python
result = retriever.query_with_fallback(question="...")

# Access properties
result.answer          # The answer string
result.source          # Source document
result.confidence      # Confidence score (0.0-1.0)
result.attempts        # Number of attempts made
result.cost            # Cost in USD
result.intermediate_steps  # List of all attempts (if return_intermediate_steps=True)
```

#### CostTracker

Track and manage costs:

```python
cost_tracker = CostTracker(budget=10.0)  # $10 budget

# After queries
report = cost_tracker.get_report()
print(f"Total Cost: ${report['total_cost']:.4f}")
print(f"Budget Remaining: ${report['budget_remaining']:.4f}")
```

#### MetricsCollector

Track performance metrics:

```python
metrics = MetricsCollector()

# After queries
stats = metrics.get_stats()
print(f"Success Rate: {stats['success_rate']:.2%}")
print(f"Average Confidence: {stats['avg_confidence']:.2f}")
```

---

## 🔌 Integrations

### LLM Providers

**Open-Source (Free, No API Keys):**

- ✅ **HuggingFace Inference API** - Use HuggingFace models via API (free tier available, easiest!)
- ✅ **HuggingFace Transformers** - Run HuggingFace models locally (requires transformers & torch)
- ✅ **Ollama** - Run LLMs locally (llama3, llama2, mistral, etc.)

**Paid (Require API Keys):**

- ✅ **OpenAI** - GPT-4, GPT-3.5, GPT-4o-mini
- ✅ **Anthropic** - Claude 3 (Opus, Sonnet, Haiku)
- ✅ **Cohere** - Command models

### Embeddings

**Open-Source (Free, No API Keys):**

- ✅ **HuggingFace** - sentence-transformers models (all-MiniLM-L6-v2, etc.)
- ✅ **Ollama** - Local embedding models (nomic-embed-text)

**Paid (Require API Keys):**

- ✅ **OpenAI** - text-embedding-3-small, text-embedding-3-large

### Vector Stores

**Open-Source (Free, Local):**

- ✅ **FAISS** - Facebook AI Similarity Search (local, fast)
- ✅ **ChromaDB** - Open-source embedding database (local)
- ✅ **Qdrant** - Vector database (can run locally or cloud)

**Paid (Cloud Services):**

- ✅ **Pinecone** - Managed vector database (requires API key)
- ✅ **Weaviate** - Can be self-hosted or cloud

---

## 🧪 Examples

### Production-Grade Examples (Advanced)

- **legal_document_analysis.py** - Legal contract analysis with ambiguous queries, cross-references, high-stakes decisions
- **medical_research_synthesis.py** - Medical research synthesis with conflicting studies, evidence levels, source attribution
- **financial_risk_analysis.py** - Financial risk assessment with regulatory compliance, multi-factor analysis, budget tracking
- **multi_domain_synthesis.py** - Enterprise knowledge base with cross-domain queries, priority resolution, complex reasoning

### Standard Examples

- **python_docs_example.py** - Python documentation Q&A
- **tech_support_example.py** - Technical support knowledge base
- **complete_example.py** - Full feature demonstration
- **huggingface_example.py** - Machine learning documentation Q&A
- **open_source_example.py** - Open-source setup example
- **paid_llm_example.py** - Paid LLM integration
- **basic_usage.py** - Basic usage example

### Quick Setup for Open-Source

**Option 1: HuggingFace Inference API (Easiest - No Installation!)**

```bash
# Install dependencies
pip install ragfallback[huggingface,sentence-transformers,faiss]

# Run HuggingFace example
python examples/huggingface_example.py
```

**Option 2: Ollama (Local)**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3

# Install dependencies
pip install ragfallback[sentence-transformers,faiss]

# Run example
python examples/open_source_example.py
```

**Option 3: Local HuggingFace Models**

```bash
# Install with transformers support
pip install ragfallback[transformers,sentence-transformers,faiss]

# Run HuggingFace example (choose local mode)
python examples/huggingface_example.py
```

No API keys needed! 🎉

---

## 📊 Why ragfallback?

| Feature             | LangChain MultiQueryRetriever | ragfallback              |
| ------------------- | ----------------------------- | ------------------------ |
| Query Variations    | ✅                            | ✅                       |
| Fallback Strategies | ❌                            | ✅ (Multiple strategies) |
| Cost Tracking       | ❌                            | ✅                       |
| Budget Management   | ❌                            | ✅                       |
| Confidence Scoring  | ❌                            | ✅                       |
| Metrics Collection  | ❌                            | ✅                       |
| Framework Agnostic  | ❌                            | ✅                       |
| Open-Source First   | ❌                            | ✅                       |

---

## 🛠️ Advanced Usage

### Custom Fallback Strategy

```python
from ragfallback.strategies.base import FallbackStrategy
from langchain_core.language_models import BaseLanguageModel

class MyCustomStrategy(FallbackStrategy):
    def generate_queries(self, original_query, context, attempt, llm):
        # Your custom logic
        return [original_query + " expanded"]

retriever = AdaptiveRAGRetriever(
    ...,
    fallback_strategies=[MyCustomStrategy()]
)
```

### Mixing Open-Source and Paid Components

```python
# Paid LLM + Open-source vector store + Open-source embeddings
llm = create_openai_llm(model="gpt-4o-mini")  # Paid
embeddings = create_open_source_embeddings()  # Free
vector_store = create_faiss_vector_store(documents, embeddings)  # Free
```

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a Pull Request.

### Quick Contribution Guide

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

---

## 🙏 Acknowledgments

Built on top of [LangChain](https://github.com/langchain-ai/langchain) and inspired by production RAG systems.

---

## 📚 Resources

- [Documentation](https://github.com/irfanalidv/ragfallback#readme)
- [Issue Tracker](https://github.com/irfanalidv/ragfallback/issues)
- [GitHub Repository](https://github.com/irfanalidv/ragfallback)

## 🧪 Testing

### Quick Verification

```bash
# 1. Install library
pip install -e .

# 2. Verify installation (tests all core functionality)
python verify_library.py

# 3. Run all examples
python run_all_examples.py
```

**Expected:** All 6 verification tests pass ✅

### Unit Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=ragfallback --cov-report=html
```

### Test Individual Examples

**Simple Examples (No API keys needed):**

```bash
python examples/python_docs_example.py
python examples/tech_support_example.py
```

**Advanced Examples (Require HuggingFace Inference API - free tier):**

```bash
python examples/legal_document_analysis.py
python examples/medical_research_synthesis.py
python examples/financial_risk_analysis.py
python examples/multi_domain_synthesis.py
```

For complete installation and testing guide, see [INSTALL_AND_RUN.md](INSTALL_AND_RUN.md).

---

**Made with ❤️ for the RAG community**
