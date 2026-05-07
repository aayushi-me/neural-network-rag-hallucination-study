# Neural Network Classifier with RAG Hallucination Study

A binary neural network classifier on real loan data combined with an empirical RAG evaluation framework that quantifies hallucination reduction through retrieval grounding.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

## Overview

This project has two parts. The first is a binary neural network classifier trained on real loan approval data with full preprocessing and evaluation. The second is an empirical RAG study comparing grounded versus ungrounded DistilGPT-2 outputs on identical queries to measure hallucination reduction.

## Features

- Binary neural network classifier with 81% test accuracy on loan approval data
- Full preprocessing pipeline with binary encoding, standard scaling, and one-hot encoding for mixed data types
- Model evaluation using precision, recall, F1-score, and confusion matrix for imbalanced classification
- RAG pipeline using SentenceTransformers and FAISS for retrieval-grounded response generation
- Side-by-side comparison of grounded versus ungrounded LLM outputs on identical queries

## Getting Started

### Installation

```bash
pip install torch scikit-learn pandas sentence-transformers faiss-cpu transformers
```

### Run

```bash
jupyter notebook neural-network-rag-hallucination-study.ipynb
```

## License

MIT License
