# debug_q9.py
from generate_answers import _semantic_search, _bm25_search, _reciprocal_rank_fusion, SEMANTIC_CANDIDATES, BM25_CANDIDATES

query = "What baseline methods did SpaCellAgent compare against for trajectory inference?"

sem = _semantic_search(query, SEMANTIC_CANDIDATES)
bm25 = _bm25_search(query, BM25_CANDIDATES)
fused = _reciprocal_rank_fusion(sem, bm25)

for i, c in enumerate(fused, start=1):
    marker = " <-- target page" if c["page_number"] == 5 else ""
    print(f"{i}. {c['source_file']} p{c['page_number']} (len={len(c['content'])}){marker}")