# debug_page_text.py
from generate_answers import get_connection

conn = get_connection()
cur = conn.cursor()
cur.execute(
    "SELECT page, text FROM chunks WHERE source = %s AND page IN (2, 5) ORDER BY page;",
    ("paper10.pdf",)
)
for page, text in cur.fetchall():
    print(f"\n--- page {page} ---")
    print(text)
cur.close()
conn.close()