# RAG Paper Assistant

A Retrieval-Augmented Generation (RAG) app that answers questions strictly from your own PDF documents, with a source citation for every claim and a gate that refuses questions the documents can't answer. Built as a portfolio project to demonstrate an end-to-end RAG system: ingestion, hybrid retrieval, re-ranking, grounded generation, evaluation, and deployment.

**Live demo:** https://your-app-name.streamlit.app

## Why this project exists

Most RAG tutorials stop at "embed, retrieve, stuff into a prompt." This project makes deliberate decisions at each layer instead, and measures them: hybrid keyword and semantic search, cross-encoder re-ranking, an explicit relevance gate against hallucination, citations parsed from the model's own answer, and an evaluation harness with real numbers. The reasoning behind each choice is documented below.

## Features

- **PDF upload and indexing** from the web UI: extract text per page, chunk, embed, store
- **Hybrid retrieval**: pgvector cosine search and BM25 keyword search, merged with Reciprocal Rank Fusion (RRF)
- **Cross-encoder re-ranking** of the fused candidates for higher precision
- **Hallucination gate**: if no chunk is relevant enough, the app answers "I don't know" and never calls the LLM
- **Grounded generation** through Groq, with numbered context chunks and inline `[n]` citations
- **Citations that match the answer**: only the passages the model actually cited are returned, shown as expandable sources with page numbers and relevance scores
- **Streamlit interface** with a document library, chat history, and a clear empty-state guide
- **Evaluation harness** measuring retrieval hit rate, answer accuracy, citation accuracy, and refusal behaviour
- **Swappable LLM provider**: Groq in the cloud, or local Ollama, selected by one setting

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        A[PDF upload] --> B[Extract text per page]
        B --> C[Chunk text]
        C --> D["Embed with all-MiniLM-L6-v2"]
        D --> E[("Supabase pgvector: chunks table")]
    end
    subgraph Query
        Q[User question] --> S1["Semantic search: pgvector cosine"]
        Q --> S2["Keyword search: BM25"]
        S1 --> F[Reciprocal Rank Fusion]
        S2 --> F
        F --> R[Cross-encoder re-rank]
        R --> G{"Any chunk above relevance threshold?"}
        G -- No --> N["Refuse: I don't know based on the provided documents"]
        G -- Yes --> L["LLM via Groq with numbered context"]
        L --> P[Parse cited chunk numbers]
        P --> O[Answer plus cited passages]
    end
    E -. vectors .-> S1
    E -. text .-> S2
```

### How a question is answered

1. The question is embedded with MiniLM and matched against the `chunks` table by cosine distance (top 20).
2. In parallel, BM25 scores every chunk on keyword overlap (top 20). This catches exact names, acronyms, and numbers that embeddings can blur.
3. RRF merges the two ranked lists by rank position, so cosine distances and BM25 scores never need to be put on the same scale.
4. A cross-encoder scores the top 15 fused candidates jointly with the question and returns the best 10.
5. Chunks scoring below the relevance threshold are dropped. If none remain, the app refuses and the LLM is never called.
6. The top remaining chunks are sent to the LLM as numbered context (`[1]`, `[2]`, ...). The model is told to answer only from that context and cite chunk numbers.
7. Cited numbers are parsed from the answer, and only those passages are shown as sources.

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` (384-dim) |
| Vector storage | pgvector on Supabase-hosted PostgreSQL |
| Keyword search | `rank-bm25` (in-memory index, rebuilt after new uploads) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq (`openai/gpt-oss-20b` by default); local Ollama (`llama3.2`) optional |
| PDF parsing | `pypdf` |
| Frontend | Streamlit |
| Hosting | Streamlit Community Cloud |
| Core libraries | `psycopg2`, `pgvector`, `python-dotenv`, `groq` |

## Evaluation

Run with `python src/run_eval.py`. The test set has 10 answerable questions drawn from the indexed papers (each with an expected source, page, and keywords) and 4 unanswerable questions the system should refuse.

<!-- Update these numbers after re-running src/run_eval.py -->

| Metric | Result |
|---|---|
| Hit@10 (right source, nearby page or matching keywords) | 100% |
| Hit@10, exact page only | 90% |
| Answer accuracy (all expected keywords present) | 50% |
| Citation accuracy (cited passage is from the right paper) | 100% |
| Refusal accuracy (unanswerable questions refused) | 100% |
| False refusal rate (answerable questions wrongly refused) | 0% |
| Eval set | 10 answerable + 4 unanswerable questions |

**How to read these numbers:**
- Retrieval and citations are strong: the right paper was found for every answerable question, and every citation pointed to the right paper.
- Answer accuracy is a strict keyword-match check, not human grading. It can understate correctness when the model words an answer differently from the expected keywords.
- The set is small, so treat the figures as a baseline rather than a benchmark. Re-run the eval whenever chunking, thresholds, or models change.

## Key Architectural Decisions

**pgvector over a dedicated vector DB (e.g. ChromaDB).**
Storing embeddings, text, and metadata in one Postgres instance avoids running and syncing two databases. For this scope, one well-indexed relational store is simpler and easier to deploy.

**Hybrid search with RRF instead of embeddings alone.**
Small bi-encoders blur exact terms such as method names, acronyms, and numbers. BM25 catches those, embeddings catch paraphrases, and RRF combines them without any score tuning.

**Cross-encoder re-ranking.**
A cross-encoder reads the question and chunk together, which is slower than comparing embeddings but much more accurate. It only runs on the small fused candidate pool, so latency stays low.

**Relevance gate on the re-rank score.**
The first version gated on cosine distance. After RRF fusion that signal is unreliable (BM25-only hits have no distance at all), so the gate now uses the cross-encoder score. These scores are unbounded logits, so the threshold (`CROSS_ENCODER_THRESHOLD`, default `-2.0`) is tuned with the eval's refusal accuracy and false-refusal rate.

**Citations parsed from the answer.**
Chunks are numbered in the prompt and the model cites them as `[n]`. The code parses those numbers and returns only the cited passages with their source file, page, and score. If the model forgets to cite, the top two chunks are returned as a fallback.

**Provider-agnostic generation.**
All LLM calls go through one function (`_call_llm`). Groq is the default because Ollama can't run on Streamlit Community Cloud; setting `LLM_PROVIDER=ollama` switches to a local model with no other code change.

**DDL/DML separation.**
The schema is created once, manually, in the Supabase SQL editor. The Python code only runs `SELECT` and `INSERT` statements.

## Project Structure

```
rag-paper-assistant/
├── src/
│   ├── app.py                # Streamlit UI: upload, library, chat, sources
│   ├── generate_answers.py   # Retrieval (semantic + BM25 + RRF + re-rank), gate, generation, citations
│   ├── ingest.py             # Upload pipeline: PDF -> chunks -> embeddings -> database
│   ├── utils.py              # get_secret(): reads .env locally and st.secrets in the cloud
│   ├── load_documents.py     # Batch PDF loading
│   ├── chunk_documents.py    # Batch chunking
│   ├── store_embeddings.py   # Batch embedding and storage
│   ├── test_retrieval.py     # Standalone retrieval testing
│   └── run_eval.py           # Evaluation harness
├── .streamlit/
│   └── config.toml           # Theme and server settings
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

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

Copy `.env.example` to `.env` and fill it in:

```env
DATABASE_URL=postgresql://user:password@host:5432/postgres
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
# Optional: use a local Ollama model instead of Groq
# LLM_PROVIDER=ollama
```

Use the Supabase session-mode pooler connection string (port 5432). Get a free Groq API key at [console.groq.com](https://console.groq.com); model names change over time, so check which models your key can use.

### 4. Set up the database schema

Run this once in the Supabase SQL editor (not via Python):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    page INT,
    embedding VECTOR(384) NOT NULL
);
```

### 5. Add documents and run the app

Either upload PDFs in the app's sidebar, or index a folder with the batch scripts:

```bash
python src/load_documents.py
python src/chunk_documents.py
python src/store_embeddings.py
```

Then start the app:

```bash
streamlit run src/app.py
```

### Other commands

```bash
python src/generate_answers.py   # ask questions in the terminal
python src/run_eval.py           # run the evaluation and write eval_results.json
```

## Usage Example

```python
# run from the src/ folder
from generate_answers import generate_answer

result = generate_answer("What baseline methods did SpaCellAgent compare against?")
print(result["answer"])
for c in result["citations"]:
    print(c["chunk_number"], c["source_file"], c["page_number"], c["rerank_score"])
```

## Deployment

The app is deployed on Streamlit Community Cloud.

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repository with the main file path `src/app.py`.
3. Under Advanced settings, add the secrets:
   ```toml
   DATABASE_URL = "postgresql://..."
   GROQ_API_KEY = "gsk_..."
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```
4. Deploy. The first build takes several minutes while the ML libraries install.

## Roadmap

- [x] Document loading, chunking, and embedding into pgvector
- [x] Hybrid search (BM25 + semantic) with Reciprocal Rank Fusion
- [x] Cross-encoder re-ranking
- [x] Relevance gate against hallucination
- [x] Citation parsing: only passages the model cited are returned
- [x] Groq generation with a swappable provider
- [x] Evaluation harness (retrieval, answers, citations, refusals)
- [x] Streamlit frontend with PDF upload
- [x] Deployment on Streamlit Community Cloud
- [ ] FastAPI wrapper with connection pooling and chat history persistence
- [ ] Per-user document isolation
- [ ] Larger evaluation set and LLM-as-judge answer grading
- [ ] Streaming responses
- [ ] OCR support for scanned PDFs

## Known Limitations

- **Shared index:** all uploaded documents go into one shared index, so anyone using the app can add documents and every question searches all of them.
- **Free-tier infrastructure:** Streamlit Community Cloud puts idle apps to sleep, Supabase's free tier pauses inactive projects, and Groq's free tier is rate limited. The first visit after a quiet period can be slow or fail until the services wake up.
- **Privacy:** don't upload sensitive documents; the free tiers offer no data-privacy guarantees.
- **Scanned PDFs:** only PDFs with a text layer are supported. Uploads are limited to 15 MB and 150 pages.
- **Chunking:** uploads use a character-based chunker (1,000 characters with 150 overlap), which can split tables and lists awkwardly.
- **Evaluation:** the test set is small and answer accuracy uses keyword matching rather than human or model grading.
- **Chat history:** conversations live in the browser session and are not persisted.
