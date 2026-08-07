import os
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads .env into environment variables


def test_query(question, n_results=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode(question).tolist()

    conn = psycopg2.connect(
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        sslmode="require"
    )
    register_vector(conn)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT text, source, page, embedding <=> %s::vector AS distance
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (question_embedding, question_embedding, n_results)
    )

    results = cur.fetchall()
    print(f"\nQuestion: {question}\n")
    for i, (text, source, page, distance) in enumerate(results):
        print(f"--- Match {i+1} (distance: {distance:.3f}) ---")
        print(f"Source: {source}, Page: {page}")
        print(f"Text: {text[:300]}...")
        print()

    cur.close()
    conn.close()


if __name__ == "__main__":
    test_query("What datasets were used for evaluation?")