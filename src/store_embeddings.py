import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from load_documents import load_pdfs
from chunk_documents import chunk_documents


def clean(value):
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def build_vector_store():
    docs = load_pdfs("../data")
    chunks = chunk_documents(docs)
    print(f"Total chunks to insert: {len(chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

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

    for i, c in enumerate(chunks):
        chunk_id = clean(c["chunk_id"])
        text = clean(c["text"])
        source = clean(c["source"])
        page = c["page"]

        embedding = model.encode(text).tolist()
        cur.execute(
            """
            INSERT INTO chunks(id,text,source,page,embedding)
            VALUES(%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO NOTHING;
            """,
            (chunk_id, text, source, page, embedding)
        )

        if (i + 1) % 20 == 0:
            conn.commit()
            print(f"Committed {i + 1} chunks so far...")

    conn.commit()  # final commit for any remaining rows
    print(f"Stored {len(chunks)} chunks in pgvector")

    cur.close()
    conn.close()


if __name__ == "__main__":
    build_vector_store()