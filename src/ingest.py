# ingest.py
# Upload-time pipeline: PDF bytes -> text per page -> chunks -> MiniLM embeddings -> Supabase pgvector.
# Writes to the same `chunks` table that generate_answers.py reads (text, source, page, embedding).

import io
import re
import uuid

import numpy as np
from pypdf import PdfReader
from psycopg2.extras import execute_values

from generate_answers import embed_model, get_connection, _reset_bm25_index

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks
MAX_PAGES = 150        # guard for the free-tier database


def _clean(text: str) -> str:
    text = text.replace("\x00", "")          # Postgres rejects NUL characters
    return re.sub(r"\s+", " ", text).strip()


def _split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start + size // 2, end)   # prefer breaking at a space
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def source_exists(filename: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM chunks WHERE source = %s LIMIT 1;", (filename,))
    found = cur.fetchone() is not None
    cur.close()
    conn.close()
    return found


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    """Index one PDF. Returns the number of chunks stored.
    Raises ValueError with a readable message for expected problems."""
    if source_exists(filename):
        raise ValueError("already indexed (rename the file to re-upload)")

    reader = PdfReader(io.BytesIO(file_bytes))
    if len(reader.pages) > MAX_PAGES:
        raise ValueError(f"too many pages ({len(reader.pages)}; limit {MAX_PAGES})")

    texts, pages = [], []
    for page_number, page in enumerate(reader.pages, start=1):   # 1-based, same as your existing data
        page_text = _clean(page.extract_text() or "")
        for piece in _split(page_text):
            texts.append(piece)
            pages.append(page_number)

    if not texts:
        raise ValueError("no extractable text (scanned PDF? OCR is not supported)")

    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=False)
    conn = get_connection()
    try:
        cur = conn.cursor()
        # chunks.id is a text column with no default, so generate a unique string id per chunk.
        rows = [
            (str(uuid.uuid4()), t, filename, p, np.asarray(e, dtype=np.float32))
            for t, p, e in zip(texts, pages, embeddings)
        ]
        execute_values(
            cur,
            "INSERT INTO chunks (id, text, source, page, embedding) VALUES %s",
            rows,
        )
        conn.commit()
        cur.close()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    _reset_bm25_index()   # so keyword search sees the new chunks on the next query
    return len(rows)