<div align="center">

# Dense Retrieval on the BEIR FiQA Benchmark

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: BEIR FiQA](https://img.shields.io/badge/Dataset-BEIR%20FiQA-green.svg)](https://github.com/beir-cellar/beir)

*Semantic search over financial questions with sentence-transformer embeddings and a FAISS HNSW index, built on Pyserini.*

</div>

## Overview

This repository contains a single Jupyter notebook that walks through a neural information retrieval pipeline end to end: downloading a standard IR benchmark, encoding its corpus into dense vectors, building an approximate nearest neighbor index, and answering free-text queries with embedding similarity instead of keyword matching. It was developed as a course project and is written to run on Google Colab with a GPU.

## Data

[FiQA](https://sites.google.com/view/fiqa/) (financial question answering), downloaded through the [BEIR](https://github.com/beir-cellar/beir) benchmark utilities and converted to Pyserini's JSONL corpus format. The notebook fetches the data itself; nothing is committed to this repository.

## Methods

- **Encoder**: `microsoft/xtremedistil-l6-h256-uncased` (256-dimensional embeddings), run via `pyserini.encode` with the sentence-transformers encoder class.
- **Index**: FAISS HNSW index built with `pyserini.index.faiss`.
- **Search**: `FaissSearcher` with an `AutoQueryEncoder` retrieves the top-scoring documents for natural language queries.

## Quick Start

```bash
git clone https://github.com/mohammadi-hadi/better-than-bing.git
cd better-than-bing
pip install -r requirements.txt

jupyter notebook XAI_Project.ipynb
```

The notebook was written for Google Colab (it mounts Google Drive and encodes on `cuda:0`), but runs anywhere with a GPU. Run the cells in order: corpus preparation, encoding, indexing, then search. Encoding the FiQA corpus is the slow step; a GPU is strongly recommended.

## Repository Structure

```
better-than-bing/
├── XAI_Project.ipynb   # Full pipeline: download, encode, index, search
├── requirements.txt    # Python dependencies
├── CONTRIBUTING.md     # Contribution guidelines
├── LICENSE             # MIT License
└── README.md           # This file
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Hadi Mohammadi** — Utrecht University
- Website: [mohammadi.cv](https://mohammadi.cv)
