from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-rag-milvus",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade RAG system with Milvus for multi-stage information retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-rag-milvus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "embeddings": [
            "openai>=1.0.0",
            "cohere>=4.0.0",
            "sentence-transformers>=2.2.0",
        ],
        "reranking": [
            "transformers>=4.30.0",
            "torch>=2.0.0",
        ],
        "nlp": [
            "spacy>=3.5.0",
            "nltk>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-pipeline=example_usage:main",
        ],
    },
)
