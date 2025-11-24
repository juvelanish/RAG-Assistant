# Retrieval Evaluation Guide

This document describes a minimal, reproducible evaluation procedure for the RAG Assistant retrieval pipeline.

1) Dataset
- Prepare a held-out set of queries with ground-truth document or chunk identifiers (CSV with columns: `query,oracle_id`).

2) Metrics
- Precision@k (k=1,5): fraction of retrieved chunks in top-k that are relevant.
- Recall@k: fraction of oracle documents recovered in top-k.
- MRR: mean reciprocal rank of first relevant chunk.
- nDCG@k: useful for graded relevance.
- Latency: median and 95th percentile of end-to-end retrieval+LLM time.

3) Example workflow (Python/pseudocode)

1. Load held-out queries.
2. For each query, call `VectorDB.search(query, n_results=5)` and collect returned `ids`/`documents`.
3. Compare returned ids to oracle_id and compute metrics.

4) Reporting
- Run experiments varying `chunk_size` and `chunk_overlap` and report precision@1, precision@5, MRR, and median latency in a table.

5) Human evaluation
- For a subset (50–200 queries), collect human judgments on answer usefulness and faithfulness (3 raters per query).

More details: use this guide to create simple scripts that compute these metrics and store results as CSV for reproducibility.