# RAG Paper Assistant

A Retrieval-Augmented Generation (RAG) document Q&A chatbot that answers questions grounded strictly in a set of uploaded documents, with source citations and a hallucination-prevention gate. Built as a portfolio project to demonstrate end-to-end RAG architecture — from document ingestion to LLM-backed, citation-aware generation.

## Why this project exists

Most RAG tutorials stop at "embed, retrieve, stuff into a prompt." This project makes deliberate architectural decisions at each layer instead — unified vector + relational storage, an explicit similarity gate to prevent hallucination, and structured citation metadata separate from the answer text — and documents the reasoning behind them below.

## Features

- **Document ingestion**: Loads and chunks PDF documents for embedding
- **Semantic retrieval**: pgvector-based cosine similarity search over document chunks
- **Grounded generation**: Claude API generates answers using *only* retrieved context, with explicit citation tagging
- **Hallucination gate**: Queries with no sufficiently relevant context are rejected before ever reaching the LLM
- **Structured citations**: Source file and page number metadata returned alongside every answer

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│   PDF Docs  │ --> │   Chunking   │ --> │  Embedding Model   │
└─────────────┘     └──────────────┘     │ (all-MiniLM-L6-v2) │
                                          └──────────┬─────────┘
                                                      │
                                                      v
                                          ┌───────────────────┐
                                          │   pgvector table   │
                                          │  (Supabase/Postgres)│
                                          └──────────┬─────────┘
                                                      │
User Query ──> Embed ──> Cosine similarity search ───┘
                              │
                              v
                    Similarity threshold gate
                    (reject if best match too weak)
                              │
                              v
                    Claude API (grounded generation)
                              │
                              v
                    Answer + structured citations
```

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) |
| Vector storage | pgvector on Supabase-hosted PostgreSQL |
| LLM | Anthropic Claude API |
| Backend (planned) | FastAPI |
| Frontend (planned) | Streamlit |
| Core libraries | `psycopg2`, `pgvector`, `pypdf`, `python-dotenv`, `anthropic` |

## Key Architectural Decisions

**pgvector over a dedicated vector DB (e.g. ChromaDB).**
Storing document embeddings, user sessions, and chat history in a single Postgres instance avoids the operational overhead of running and syncing two separate databases. For a project of this scope, one well-indexed relational store outperforms the complexity of a split architecture.

**Similarity threshold as a hallucination gate.**
Before calling the LLM at all, the top retrieved chunk's cosine distance is checked against a threshold (`0.35`). If nothing sufficiently relevant is found, the pipeline returns a fixed "I don't have enough information" response — this avoids the LLM ever generating a plausible-sounding answer from irrelevant or absent context.

**Structured citations, separate from prose.**
Instead of asking the model to embed source links directly in free text, citation metadata (source file, page number) is tracked and returned as a distinct field. This keeps citations machine-readable and consistent regardless of how the model phrases its answer.

**DDL/DML separation.**
Schema definition (`CREATE TABLE`, `CREATE EXTENSION vector`) is handled once, manually, via the Supabase SQL editor. All Python scripts are DML-only (`INSERT`/`SELECT`), since Supabase's transaction-mode connection pooler doesn't support DDL statements.

## Project Structure

```
rag-paper-assistant/
├── load_documents.py       # PDF loading and text extraction
├── chunk_documents.py      # Splits documents into embeddable chunks
├── store_embeddings.py     # Generates and stores embeddings in pgvector
├── test_retrieval.py       # Standalone retrieval testing
├── generate_answer.py      # Core RAG loop: retrieve -> gate -> generate -> cite
├── .env                    # Environment variables (not committed)
├── .gitignore
└── README.md
```

> A `src/`-based package layout is planned once FastAPI is introduced.

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/rag-paper-assistant.git
cd rag-paper-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
PGDATABASE=your_db_name
PGUSER=your_db_user
PGPASSWORD=your_db_password
PGHOST=your_supabase_host
PGPORT=5432
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 4. Set up the database schema

Run this once in the Supabase SQL editor (not via Python — see DDL/DML note above):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    source_file TEXT NOT NULL,
    page_number INT,
    chunk_index INT
);
```

### 5. Run the pipeline

```bash
python load_documents.py
python chunk_documents.py
python store_embeddings.py
python generate_answer.py
```

## Usage Example

```python
from generate_answer import generate_answer

result = generate_answer("What optimizer does the paper use for training?")
print(result["answer"])
print(result["citations"])
```

## Roadmap

- [x] Document loading and chunking
- [x] Embedding generation and pgvector storage
- [x] Retrieval testing
- [x] Core RAG loop with citation tracking and hallucination gate
- [ ] FastAPI wrapper with connection pooling and chat history persistence
- [ ] Streamlit frontend
- [ ] Hybrid search (BM25 + semantic retrieval with Reciprocal Rank Fusion)
- [ ] LangChain integration
- [ ] Async request handling, SSE streaming, retry/backoff logic

## Known Limitations

- Citation tracking currently returns *all* retrieved chunks rather than only the ones the model explicitly referenced in its answer. A future pass will parse `[Source N]` tags from the generated text to return only cited sources.

## License

MIT