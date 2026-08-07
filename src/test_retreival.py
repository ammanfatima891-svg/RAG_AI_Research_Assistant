import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

def test_query(question, n_results=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode(question).tolist()

    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.tyezxdhjyaapraatlsmg",
        password="admin1234@AmanFatima",
        host="aws-0-ap-northeast-1.pooler.supabase.com",
        port="5432",          # session pooler, not 6543
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