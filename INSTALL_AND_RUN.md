# Install and Run ragfallback

## Quick Start

```bash
# 1. Install library
pip install -e .

# 2. Verify installation
python verify_library.py

# 3. Run all examples
python run_all_examples.py
```

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Optional Dependencies

```bash
# Open-source components (recommended)
pip install -e .[sentence-transformers,faiss]

# Or install manually
pip install sentence-transformers faiss-cpu huggingface-hub
```

## Verification

```bash
# Verify library installation and core functionality
python verify_library.py
```

**Expected:** All 6 tests pass ✅

## Run Examples

### Run All Examples

```bash
python run_all_examples.py
```

### Run Individual Examples

**Simple Examples:**

```bash
python examples/python_docs_example.py
python examples/tech_support_example.py
```

**Advanced Examples:**

```bash
python examples/legal_document_analysis.py
python examples/medical_research_synthesis.py
python examples/financial_risk_analysis.py
python examples/multi_domain_synthesis.py
python examples/complete_example.py
```

## Run Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run unit tests
pytest tests/ -v

# Run with coverage
pytest --cov=ragfallback --cov-report=html
```

## Library Structure

```
ragfallback/
├── ragfallback/          # Core library
├── examples/            # 11 example files
├── tests/               # Unit tests
├── verify_library.py    # Installation verification
├── run_all_examples.py  # Run all examples
├── pyproject.toml       # Package config
├── requirements-dev.txt # Dev dependencies
└── README.md           # Documentation
```

## Success Criteria

✅ `python verify_library.py` passes all tests  
✅ `pytest tests/` passes all unit tests  
✅ At least 3 examples run successfully  
✅ Library can be imported: `from ragfallback import AdaptiveRAGRetriever`
