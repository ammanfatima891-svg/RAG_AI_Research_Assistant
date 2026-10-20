# debug_chunks.py — one-off diagnostic, not part of the pipeline
from generate_answers import retrieve_chunks

chunks = retrieve_chunks("Which model performed best overall on the ImagingBench leaderboard under the Expert protocol?")
for c in chunks:
    print(f"\n--- {c['source_file']} p{c['page_number']} (dist {c['distance']:.3f}) ---")
    print(c["content"])