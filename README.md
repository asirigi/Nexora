# Nexora
This repository contains the app that can ingest different files and can provide insights on that data when user asks any question

User stories:

User Story 1: Document Ingestion, Chunking, and Vectorization (Backend Core)

Sub-Tasks:
virtual environment setup 
Chunk documents
Use Docling/Marker for PDF/docx
Generate embeddings (e.g., Sentence-BERT)
Store in vector database (e.g., ChromaDB) with metadata

Sub-Tasks:
Extract metadata from CSV/Excel (e.g., via Pydantic models)
Csv agent

User Story 2: Pending