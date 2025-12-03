"""Setup script for ragfallback."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ragfallback",
    version="0.1.0",
    author="Irfan Ali",
    author_email="irfanali29@hotmail.com",
    description="RAG Fallback Strategies - Intelligent fallback mechanisms for RAG systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/irfanalidv/ragfallback",
    project_urls={
        "Documentation": "https://github.com/irfanalidv/ragfallback#readme",
        "Repository": "https://github.com/irfanalidv/ragfallback",
        "Issues": "https://github.com/irfanalidv/ragfallback/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "pydantic>=2.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "ollama": ["langchain-community>=0.0.20"],
        "huggingface": ["huggingface-hub>=0.16.0"],
        "transformers": ["transformers>=4.30.0", "torch>=2.0.0"],
        "sentence-transformers": ["sentence-transformers>=2.2.0"],
        "faiss": ["faiss-cpu>=1.7.4"],
        "chroma": ["chromadb>=0.4.0"],
        "qdrant": ["qdrant-client>=1.7.0"],
        "openai": ["langchain-openai>=0.0.5", "openai>=1.0.0"],
        "anthropic": ["langchain-anthropic>=0.1.0", "anthropic>=0.18.0"],
        "pinecone": ["pinecone-client>=2.2.0"],
        "weaviate": ["weaviate-client>=3.25.0"],
        "cohere": ["cohere>=4.0.0"],
        "open-source": [
            "huggingface-hub>=0.16.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.4",
            "chromadb>=0.4.0",
        ],
        "paid": [
            "langchain-openai>=0.0.5",
            "openai>=1.0.0",
            "langchain-anthropic>=0.1.0",
            "anthropic>=0.18.0",
            "pinecone-client>=2.2.0",
        ],
    },
    keywords="rag retrieval llm fallback query-variations langchain",
    license="MIT",
    package_data={
        "ragfallback": ["py.typed"],
    },
    include_package_data=True,
)
