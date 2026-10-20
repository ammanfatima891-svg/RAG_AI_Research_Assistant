# generate_answers.py

import re
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from utils import get_secret

# ---- Config ----
DB_URL = get_secret("DATABASE_URL")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM provider: "groq" (cloud, default) or "ollama" (local). Switch via the
# LLM_PROVIDER setting in .env / Streamlit secrets, no code change needed.
LLM_PROVIDER = get_secret("LLM_PROVIDER", "groq")
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-20b")  # fallback: "llama-3.1-8b-instant"
OLLAMA_MODEL = "llama3.2"

TOP_K = 10                   # final number of chunks returned to the caller
SEMANTIC_CANDIDATES = 20     # how many candidates to pull from pgvector
BM25_CANDIDATES = 20         # how many candidates to pull from BM25
RRF_K = 60                   # standard RRF damping constant
RERANK_POOL_SIZE = 15        # how many fused candidates go into the cross-encoder
CROSS_ENCODER_THRESHOLD = -2.0  # gate: below this, treat as "not relevant enough"
# ^ ms-marco-MiniLM-L-6-v2 scores are unbounded logits, not 0-1 similarities.
#   Tune this using run_eval.py (refusal accuracy vs false refusal rate).

LLM_CONTEXT_CHUNKS = 5       # how many top relevant chunks are sent to the LLM
REFUSAL_TEXT = "I don't know based on the provided documents."

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Module-level cache for the BM25 index, built once on first use rather than
# per query (rebuilding it every call would be slow and pointless — the
# corpus only changes when you re-ingest documents).
_bm25_index = None
_bm25_corpus_ids = None          # list[int], same order as the BM25 corpus
_bm25_chunk_lookup = None        # dict[id] -> {"content", "source_file", "page_number"}


def get_connection():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    return conn


def _tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def _build_bm25_index():
    """Load every chunk once and build an in-memory BM25 index. Cached at
    module level — call _reset_bm25_index() after re-ingesting documents so
    it picks up new/changed chunks on the next query."""
    global _bm25_index, _bm25_corpus_ids, _bm25_chunk_lookup

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, text, source, page FROM chunks;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    _bm25_corpus_ids = []
    _bm25_chunk_lookup = {}
    tokenized_corpus = []

    for chunk_id, text, source, page in rows:
        _bm25_corpus_ids.append(chunk_id)
        _bm25_chunk_lookup[chunk_id] = {
            "content": text,
            "source_file": source,
            "page_number": page,
        }
        tokenized_corpus.append(_tokenize(text))

    _bm25_index = BM25Okapi(tokenized_corpus)


def _reset_bm25_index():
    """Call this after re-ingesting/updating documents so the BM25 index
    rebuilds from the current chunks table on the next retrieval call."""
    global _bm25_index
    _bm25_index = None


def embed_query(query: str):
    return embed_model.encode(query).tolist()


def _semantic_search(query: str, top_n: int):
    """Dense retrieval via pgvector cosine distance. Returns candidates with
    an 'id' key so they can be merged with BM25 results downstream."""
    query_vector = embed_query(query)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, text, source, page, embedding <=> %s::vector AS distance
        FROM chunks
        ORDER BY distance ASC
        LIMIT %s;
        """,
        (query_vector, top_n)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for chunk_id, text, source, page, distance in rows:
        results.append({
            "id": chunk_id,
            "content": text,
            "source_file": source,
            "page_number": page,
            "distance": distance,
        })
    return results


def _bm25_search(query: str, top_n: int):
    """Sparse keyword retrieval. Catches exact terms (names, acronyms,
    specific phrases) that a small bi-encoder can blur across documents."""
    if _bm25_index is None:
        _build_bm25_index()

    tokens = _tokenize(query)
    scores = _bm25_index.get_scores(tokens)

    ranked = sorted(
        zip(_bm25_corpus_ids, scores), key=lambda x: x[1], reverse=True
    )[:top_n]

    results = []
    for chunk_id, score in ranked:
        if score <= 0:
            continue  # no actual term overlap — not a real candidate
        chunk = _bm25_chunk_lookup[chunk_id]
        results.append({
            "id": chunk_id,
            "content": chunk["content"],
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
            "distance": None,  # not applicable for BM25 hits
        })
    return results


def _reciprocal_rank_fusion(semantic_results, bm25_results, k: int = RRF_K):
    """Merge two ranked lists by rank position, not raw score — avoids the
    problem of cosine distance and BM25 scores living on incomparable
    scales. Each list contributes 1/(k+rank) to a chunk's fused score;
    chunks appearing in both lists get a boost from both terms."""
    fused_scores = {}
    chunk_data = {}

    for rank, chunk in enumerate(semantic_results, start=1):
        fused_scores[chunk["id"]] = fused_scores.get(chunk["id"], 0) + 1 / (k + rank)
        chunk_data[chunk["id"]] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        fused_scores[chunk["id"]] = fused_scores.get(chunk["id"], 0) + 1 / (k + rank)
        # Prefer the semantic version if we already have it (it carries a
        # real distance value); otherwise use the BM25 version.
        chunk_data.setdefault(chunk["id"], chunk)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [chunk_data[cid] for cid in ranked_ids]


def _cross_encoder_rerank(query: str, candidates: list, top_k: int):
    """Score each (query, chunk) pair jointly — slower than embedding
    similarity but far more accurate, which is why it only runs on the
    small fused candidate pool rather than the whole corpus."""
    if not candidates:
        return []

    pairs = [(query, c["content"]) for c in candidates]
    scores = cross_encoder.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        enriched = dict(chunk)
        enriched["rerank_score"] = float(score)
        results.append(enriched)
    return results


def retrieve_chunks(query: str, top_k: int = TOP_K):
    """Hybrid retrieval: semantic (pgvector) + keyword (BM25) candidates,
    fused via RRF, then re-ranked by a cross-encoder for the final top_k.
    Returns the same shape as before (content/source_file/page_number/
    distance) plus an added 'rerank_score' field."""
    semantic_results = _semantic_search(query, SEMANTIC_CANDIDATES)
    bm25_results = _bm25_search(query, BM25_CANDIDATES)

    fused = _reciprocal_rank_fusion(semantic_results, bm25_results)
    reranked = _cross_encoder_rerank(query, fused[:RERANK_POOL_SIZE], top_k)

    return reranked


# ---------------------------------------------------------------
# Generation
# ---------------------------------------------------------------

def build_prompt(query: str, chunks: list):
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[{i}] (Source: {chunk['source_file']} | Page: {chunk['page_number']})\n"
            f"{chunk['content']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a research assistant answering questions using ONLY the numbered context chunks below.

Context:
{context_text}

Question: {query}

Instructions:
- Answer using only the information in the context above.
- After each claim, cite the supporting chunk number(s) in square brackets, like [1] or [2][3]. Cite only chunks you actually used.
- If the context does not contain enough information to answer, respond exactly with: "{REFUSAL_TEXT}" and add no citations.
- Do not use outside knowledge.
- Be concise and precise.
"""
    return prompt


def _call_llm(prompt: str) -> str:
    """Single place where the LLM provider is chosen."""
    if LLM_PROVIDER == "ollama":
        import ollama  # imported lazily so the cloud deployment doesn't need it
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    from groq import Groq
    client = Groq(api_key=get_secret("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


_CITE_RE = re.compile(r"\[(?:Chunk\s*)?(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)


def _parse_citations(answer: str, n_chunks: int):
    """Return sorted chunk numbers (1-based) the model actually cited."""
    nums = set()
    for group in _CITE_RE.findall(answer):
        for x in re.findall(r"\d+", group):
            nums.add(int(x))
    return sorted(n for n in nums if 1 <= n <= n_chunks)


def generate_answer(query: str):
    chunks = retrieve_chunks(query)

    # Gate on the cross-encoder's relevance judgment rather than raw cosine
    # distance — after RRF fusion, distance isn't a reliable signal on its
    # own (BM25-only hits don't have one at all).
    relevant_chunks = [c for c in chunks if c["rerank_score"] >= CROSS_ENCODER_THRESHOLD]

    if not relevant_chunks:
        return {"answer": REFUSAL_TEXT, "citations": []}

    context_chunks = relevant_chunks[:LLM_CONTEXT_CHUNKS]
    prompt = build_prompt(query, context_chunks)

    try:
        answer_text = _call_llm(prompt)
    except Exception as e:
        return {
            "answer": f"Sorry, the language model is unavailable right now ({type(e).__name__}). Please try again.",
            "citations": [],
            "error": str(e),
        }

    # Model said it doesn't know -> no citations
    if "don't know based on the provided documents" in answer_text.lower():
        return {"answer": REFUSAL_TEXT, "citations": []}

    # Only return the chunks the answer actually cited.
    used = _parse_citations(answer_text, len(context_chunks))
    if not used:
        used = [1, 2][: len(context_chunks)]  # fallback if the model forgot to cite

    citations = [
        {
            "chunk_number": n,
            "source_file": context_chunks[n - 1]["source_file"],
            "page_number": context_chunks[n - 1]["page_number"],
            "rerank_score": round(context_chunks[n - 1]["rerank_score"], 4),
            "content": context_chunks[n - 1]["content"],
        }
        for n in used
    ]

    return {"answer": answer_text, "citations": citations}


if __name__ == "__main__":
    query = input("Ask a question about your documents: ")
    result = generate_answer(query)

    print("\n--- ANSWER ---")
    print(result["answer"])

    print("\n--- CITATIONS ---")
    for c in result["citations"]:
        print(f"- [{c['chunk_number']}] {c['source_file']} (page {c['page_number']}, rerank_score {c['rerank_score']})")