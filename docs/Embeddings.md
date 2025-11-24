# Embedding Model Selection

This file explains tradeoffs and recommended embedding models for the RAG Assistant.

1) Tradeoffs
- Managed embeddings (OpenAI, Anthropic, etc.): higher semantic quality, paid API, low maintenance.
- Open-source models (sentence-transformers): lower cost, can run locally, requires GPU for large corpora.

2) Recommendations
- Small experimental setup: use `sentence-transformers/all-MiniLM-L6-v2` for speed and reasonable quality.
- Production/high-quality: use vendor embeddings (OpenAI `text-embedding-3-large`) and measure MRR vs local models.

3) Evaluation
- Use the `docs/Evaluation.md` workflow to compare embedding models by computing MRR and precision@k on your held-out set.

4) Integration notes
- The current code uses ChromaDB default embeddings. To use a custom HF model, compute embeddings offline and insert them into Chroma.