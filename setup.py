"""Setup script for Stock Market Sentiment Predictor package."""

import os
from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stock-market-sentiment-predictor",
    version="1.0.0",
    author="Jaimin Prajapati",
    author_email="jaimin.prajapati@example.com",
    description="Enterprise-grade ML system for stock market prediction using sentiment analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jaimin-prajapati-ds/stock-market-sentiment-predictor",
    project_urls={
        "Bug Tracker": "https://github.com/Jaimin-prajapati-ds/stock-market-sentiment-predictor/issues",
        "Documentation": "https://github.com/Jaimin-prajapati-ds/stock-market-sentiment-predictor/tree/main/docs",
        "Source Code": "https://github.com/Jaimin-prajapati-ds/stock-market-sentiment-predictor",
    },
    packages=find_packages(where=".", exclude=["tests", "*.tests", "*.tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-predictor=src.main:main",
            "train-sentiment=src.models.sentiment.train:main",
            "train-price=src.models.prediction.train:main",
            "run-dashboard=src.visualization.dashboard:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning deep-learning nlp sentiment-analysis stock-market prediction fintech mlops",
)
