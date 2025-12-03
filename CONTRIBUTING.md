# Contributing to ragfallback

Thank you for considering contributing to ragfallback! This document provides guidelines for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/irfanalidv/ragfallback.git
cd ragfallback

# Install in development mode
pip install -e .[sentence-transformers,faiss]

# Install development dependencies
pip install -r requirements-dev.txt
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=ragfallback --cov-report=html tests/

# Run verification
python verify_library.py
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, descriptive commits
3. **Add tests** for new functionality
4. **Update documentation** including README if needed
5. **Run tests** to ensure everything passes
6. **Submit a pull request** with a clear description

## Commit Message Format

```
<type>: <subject>

<body>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

Example:
```
feat: Add semantic expansion fallback strategy

Implements a new fallback strategy that expands queries
using semantic similarity to improve retrieval.
```

## Adding New Features

### Adding a New Fallback Strategy

1. Create a new file in `ragfallback/strategies/`
2. Inherit from `FallbackStrategy` base class
3. Implement the `generate_queries()` method
4. Add tests in `tests/`
5. Add example in `examples/`
6. Update README

### Adding Support for New LLM/Vector Store

1. Add factory function in appropriate file:
   - `ragfallback/utils/llm_factory.py` for LLMs
   - `ragfallback/utils/vector_store_factory.py` for vector stores
   - `ragfallback/utils/embedding_factory.py` for embeddings
2. Add optional dependency in `pyproject.toml`
3. Add example demonstrating usage
4. Update README integration section

## Testing Guidelines

- Write unit tests for all new functions
- Add integration tests for new features
- Ensure tests are independent and can run in any order
- Use descriptive test names: `test_<what>_<condition>_<expected>`

## Documentation

- Update README.md for new features
- Add docstrings with examples
- Update CHANGELOG.md
- Add examples for complex features

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about usage
- Discussion about improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

