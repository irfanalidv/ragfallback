# Installing and Running ragfallback

## Install

```bash
# Core only (no vector store, no embeddings)
pip install ragfallback

# Golden path — runs all UC-1–UC-10 examples with no API keys
pip install ragfallback[chroma,huggingface]

# FAISS instead of Chroma
pip install ragfallback[faiss,huggingface]

# Add BM25 hybrid retrieval (UC-5)
pip install ragfallback[hybrid,huggingface]

# Real public dataset examples (financial, legal, medical, SQuAD)
pip install ragfallback[real-data]

# Everything open-source at once
pip install ragfallback[chroma,faiss,huggingface,hybrid,real-data]
```

## Run examples — no API keys needed

```bash
# End-to-end on 200 real Wikipedia passages (SQuAD dataset)
python examples/real_data_demo.py          # requires: [chroma,huggingface,real-data]

# Diagnostics
python examples/uc1_retrieval_health.py    # retrieval smoke test (ChromaDB)
python examples/uc2_embedding_guard.py     # dimension / NaN guard
python examples/uc3_chunk_quality.py       # chunk quality check
python examples/uc4_context_window.py      # context budget trimming
python examples/uc7_rag_evaluator.py       # answer quality scoring

# Retrieval patterns
python examples/uc8_context_stitcher.py    # merge adjacent chunks (ChromaDB)
python examples/uc9_embedding_probe.py     # domain mismatch heuristic
python examples/uc10_metadata_sanitizer.py # metadata normalization (ChromaDB)

# Domain examples (real public datasets — requires [real-data])
python examples/financial_risk_analysis.py  # nickmuchi/financial-classification
python examples/legal_document_analysis.py  # theatticusproject/cuad-qa
python examples/medical_research_synthesis.py  # qiaojin/PubMedQA
```

## Run with local LLM (Ollama — no paid keys)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3

# Full AdaptiveRAGRetriever demo with LLM generation
python examples/chroma_real_kb_demo.py

# Multi-hop strategy demo (requires LLM)
python examples/uc6_multi_hop_demo.py
```

## Examples that require setup

| Example | Requires | Install |
|---------|----------|---------|
| `uc5_hybrid_failover.py` | BM25 | `pip install ragfallback[hybrid]` |
| `uc6_multi_hop_demo.py` | Local LLM | `ollama pull llama3` |
| `qdrant_local_demo.py` | Qdrant | `docker run -p 6333:6333 qdrant/qdrant` |

## Run all examples

```bash
python run_all_examples.py   # skips examples needing Ollama/API keys, prints reasons
```

## Verify installation

```bash
python verify_library.py     # 9/9 checks — tests every subpackage
```

## Run tests

```bash
pip install -r requirements-dev.txt
pytest tests/unit/ -v        # 81 unit tests — no network, no API keys
pytest tests/integration/ -m integration -v   # 7 integration tests (needs sentence-transformers + chromadb)
```

## Development install

```bash
git clone https://github.com/irfanalidv/ragfallback
cd ragfallback
pip install -e ".[chroma,faiss,huggingface,hybrid,real-data]"
pip install -r requirements-dev.txt
pytest tests/unit/ -v
```
