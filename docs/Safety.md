# Safety & Limitations

This project is a demo and contains no production-grade safeguards by default. The following recommendations help reduce risks when deploying or sharing results:

- Do not commit API keys to the repository. Use the `.env` file (example included) and platform secrets stores for CI.
- Surface retrieved citations with answers. Always show the document name and page number for PDF excerpts.
- Keep LLM temperature at 0.0 for deterministic results when using RAG for factual answers.
- Rate-limit requests to LLM providers and handle transient errors.
- Remove or redact PII from ingested documents if not needed for the task.

Limitations:
- RAG reduces hallucinations but does not eliminate them. Use human review for high-stakes decisions.
- Embedding quality strongly affects retrieval — validate using the `docs/Evaluation.md` metrics.