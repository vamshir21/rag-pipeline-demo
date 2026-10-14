# GitDocs RAG Engine

A local Retrieval-Augmented Generation (RAG) pipeline that turns Markdown documentation into an interactive knowledge base. 

Instead of manually `grep`-ing through cloned Git repositories, this tool indexes `.md` files into a local vector database and uses Google's Gemini models to generate precise, context-aware answers based strictly on the provided documentation.

## Tech Stack
* **Framework:** LangChain
* **Vector Database:** ChromaDB (Local SQLite)
* **LLM & Embeddings:** Google Gemini (`gemini-3.8-flash` & `gemini-embedding-2-preview`)
* **Environment:** Python (managed via `uv`)

## Architecture
This project is split into two distinct phases:

1. **Data Ingestion (`ingest.py`):** 
   Scans the target directory for Markdown files, splits the text into manageable overlapping chunks (to preserve sentence context), generates high-dimensional embeddings via Gemini, and persists them locally in ChromaDB.
   
2. **Retrieval & Generation (`query.py`):** 
   Takes a user query, embeds it using the same model, and performs a similarity search against the vector database. The top results are injected into a strict system prompt, forcing the LLM to answer the question using *only* the retrieved context.

## Local Setup
1. Clone the repository.
2. Initialize a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt