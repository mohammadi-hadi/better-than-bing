# BetterThanBing

A dense (semantic) retrieval experiment on the BEIR FiQA benchmark using Pyserini, sentence-transformer embeddings, and a FAISS HNSW index.

## Overview

This repository contains a single Jupyter notebook that walks through a neural information retrieval pipeline end to end: downloading a standard IR benchmark, encoding its corpus into dense vectors, building an approximate nearest neighbor index, and answering free-text queries with embedding similarity instead of keyword matching. It was developed as a course project and is written to run on Google Colab with a GPU.

## Data / Methods

- **Dataset**: [FiQA](https://sites.google.com/view/fiqa/) (financial question answering), downloaded through the [BEIR](https://github.com/beir-cellar/beir) benchmark utilities and converted to Pyserini's JSONL corpus format.
- **Encoder**: `microsoft/xtremedistil-l6-h256-uncased` (256-dimensional embeddings), run via `pyserini.encode` with the sentence-transformers encoder class.
- **Index**: FAISS HNSW index built with `pyserini.index.faiss`.
- **Search**: `FaissSearcher` with an `AutoQueryEncoder` retrieves the top-scoring documents for natural language queries.

## Repository structure

```
better-than-bing/
├── XAI_Project.ipynb   # Full pipeline: download, encode, index, search
├── CONTRIBUTING.md     # Contribution guidelines
├── LICENSE             # MIT License
└── README.md           # This file
```

## How to run

The notebook was written for Google Colab (it mounts Google Drive and encodes on `cuda:0`), but runs anywhere with a GPU:

```bash
pip install beir pyserini tensorflow_text
jupyter notebook XAI_Project.ipynb
```

Run the cells in order: corpus preparation, encoding, indexing, then search. Encoding the FiQA corpus is the slow step; a GPU is strongly recommended.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
