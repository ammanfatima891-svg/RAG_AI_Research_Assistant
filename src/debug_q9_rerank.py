# debug_q9_rerank.py
from generate_answers import _semantic_search, _bm25_search, _reciprocal_rank_fusion, _cross_encoder_rerank, SEMANTIC_CANDIDATES, BM25_CANDIDATES, RERANK_POOL_SIZE

query = "What baseline methods did SpaCellAgent compare against for trajectory inference?"

sem = _semantic_search(query, SEMANTIC_CANDIDATES)
bm25 = _bm25_search(query, BM25_CANDIDATES)
fused = _reciprocal_rank_fusion(sem, bm25)
reranked = _cross_encoder_rerank(query, fused[:RERANK_POOL_SIZE], top_k=15)  # rank ALL 15, not just 5

for i, c in enumerate(reranked, start=1):
    marker = " <-- target page" if c["page_number"] == 5 else ""
    print(f"{i}. {c['source_file']} p{c['page_number']} score={c['rerank_score']:.3f} (len={len(c['content'])}){marker}")