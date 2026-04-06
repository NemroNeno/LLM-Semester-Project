# Project Documentation

This folder contains the formal project documentation for the LLM Semester Project.

## Documents

- SRS: [SRS.md](SRS.md)
- Architecture and Tech Stack: [ARCHITECTURE.md](ARCHITECTURE.md)

## Scope Covered

The docs cover:

- End-to-end chatbot and RAG functionality
- Knowledge graph functionality (Graphiti + Neo4j) including static and dynamic ingestion
- Dynamic admin ingestion workflow
- Data persistence and storage layout
- Retrieval, summarization, guardrails, generation behavior, and KG fallback search
- API endpoints and UI capabilities
- Fine-tuning pipeline artifacts and deployment integration via Ollama
- Non-functional requirements, constraints, risks, and operational guidance

## Newly Added KG-Related Operational Scripts

- `ingest_markdown_kg.py`
One-time static KG ingestion from `sheets_markdown/*.md` with chunking and progress bar.

- `clear_neo4j.py`
Guarded utility to clear local Neo4j graph data before re-ingestion experiments.
