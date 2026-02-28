"""Setup configuration for factcheck package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="factcheck",
    version="0.1.0",
    author="Marco Srhl",
    description="Automated fact-checking system using DBpedia and BERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarcoSrhl/factcheck",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "spacy>=3.5.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "SPARQLWrapper>=2.0.0",
        "requests>=2.28.0",
        "scikit-learn>=1.2.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
        ],
        "database": [
            "psycopg2-binary>=2.9.0",
            "python-dotenv>=1.0.0",
        ],
    },
)
