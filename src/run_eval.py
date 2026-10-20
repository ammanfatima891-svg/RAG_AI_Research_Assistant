# run_eval.py
import json
import time
from generate_answers import retrieve_chunks, generate_answer, TOP_K

# ---------------------------------------------------------------
# Answerable questions (verified against your chunks table)
# ---------------------------------------------------------------
TEST_SET = [
    {"question": "What is ImagingBench?", "expected_source": "paper1.pdf", "expected_page": 1,
     "expected_keywords": ["benchmark", "computational imaging", "agentic"]},
    {"question": "What three evaluation settings does ImagingBench use?", "expected_source": "paper1.pdf", "expected_page": 1,
     "expected_keywords": ["Expert", "Planner", "Forward"]},
    {"question": "Which model performed best overall on the ImagingBench leaderboard under the Expert protocol?",
     "expected_source": "paper1.pdf", "expected_page": 10, "expected_keywords": ["Gemini"]},
    {"question": "What is SpaCellAgent?", "expected_source": "paper10.pdf", "expected_page": 1,
     "expected_keywords": ["multi-agent", "trajectory", "LLM"]},
    {"question": "What are the specialized agent roles in the SpaCellAgent framework?",
     "expected_source": "paper10.pdf", "expected_page": 2,
     "expected_keywords": ["planner", "executor", "evaluator", "reporter"]},
    {"question": "How much does SpaCellAgent improve analytical efficiency compared to manual expert workflows?",
     "expected_source": "paper10.pdf", "expected_page": 2, "expected_keywords": ["41.2%"]},
    {"question": "What is the end-to-end API cost per image for the Expert protocol using OpenAI in ImagingBench?",
     "expected_source": "paper1.pdf", "expected_page": 11, "expected_keywords": ["0.048"]},
    {"question": "What five categories of tasks does ImagingBench cover?", "expected_source": "paper1.pdf", "expected_page": 3,
     "expected_keywords": ["Ray and Wave Optics", "Computational sensing", "Image signal processing",
                           "Inverse reconstruction", "Calibration"]},
    {"question": "What baseline methods did SpaCellAgent compare against for trajectory inference?",
     "expected_source": "paper10.pdf", "expected_page": 5, "expected_keywords": ["DPT", "RaceID", "StemID"]},
    {"question": "According to ImagingBench, are visually plausible outputs from agentic models always physically accurate?",
     "expected_source": "paper1.pdf", "expected_page": 13, "expected_keywords": ["plausible", "physically"]},
    # TODO: add 10-15 more answerable questions (especially from paper5.pdf).
    # Copy a fact you can see on a specific page; don't guess pages.
]

# ---------------------------------------------------------------
# Unanswerable questions: the system SHOULD refuse these.
# ---------------------------------------------------------------
UNANSWERABLE = [
    "What is the capital of France?",
    "Who won the 2022 FIFA World Cup?",
    "Give me a recipe for chocolate chip cookies.",
    "What is the current stock price of Tesla?",
]

REFUSAL_MARKERS = (
    "i don't know based on",
    "don't know based on the provided documents",
    "couldn't find", "could not find", "cannot find", "can't find",
)

PAUSE_SECONDS = 2      # spacing between questions (Groq free-tier rate limits)
MAX_RETRIES = 3        # retries when the LLM call fails (rate limit etc.)
RETRY_WAIT = 10
_TEXT_KEYS = ("text", "chunk_text", "content", "page_content")


def _chunk_text(chunk):
    for key in _TEXT_KEYS:
        if chunk.get(key):
            return chunk[key]
    return ""


def is_refusal(answer_text):
    a = answer_text.lower()
    return any(m in a for m in REFUSAL_MARKERS)


def keyword_fraction(text, keywords):
    t = text.lower()
    return sum(kw.lower() in t for kw in keywords) / len(keywords) if keywords else 0


def ask_with_retry(question):
    """Call generate_answer, retrying if the LLM call returned an error."""
    result = {}
    for attempt in range(1, MAX_RETRIES + 1):
        result = generate_answer(question)
        if "error" not in result:
            return result
        print(f"  LLM error (attempt {attempt}/{MAX_RETRIES}): {result['error'][:120]}")
        time.sleep(RETRY_WAIT)
    return result


def eval_retrieval(tc, top_k=TOP_K):
    """
    Relaxed hit: right source AND (page within +/-1 OR >=50% of keywords in the chunk).
    Strict hit:  right source AND exact page.
    Returns (relaxed_hit, strict_hit, relaxed_rank)
    """
    chunks = retrieve_chunks(tc["question"], top_k=top_k)
    relaxed_rank, strict = None, False
    for rank, c in enumerate(chunks, start=1):
        if c.get("source_file") != tc["expected_source"]:
            continue
        page = c.get("page_number")
        page_exact = page == tc["expected_page"]
        page_near = isinstance(page, int) and abs(page - tc["expected_page"]) <= 1
        kw_ok = keyword_fraction(_chunk_text(c), tc["expected_keywords"]) >= 0.5
        if page_exact:
            strict = True
        if relaxed_rank is None and (page_near or kw_ok):
            relaxed_rank = rank
    return relaxed_rank is not None, strict, relaxed_rank


def pct(x):
    return f"{x:.1%}"


def run_eval():
    answerable_results, refusal_results = [], []

    # ---------- answerable ----------
    for i, tc in enumerate(TEST_SET, start=1):
        print(f"\n[{i}/{len(TEST_SET)}] {tc['question']}")
        try:
            hit, strict, rank = eval_retrieval(tc)
            result = ask_with_retry(tc["question"])
            answer = result["answer"]
            llm_error = "error" in result
            kw = keyword_fraction(answer, tc["expected_keywords"])
            citations = result.get("citations", [])
            citation_hit = any(c.get("source_file") == tc["expected_source"] for c in citations)
            err = result.get("error")
        except Exception as e:
            hit, strict, rank, answer, kw = False, False, None, "", 0
            citations, citation_hit, llm_error, err = [], False, True, str(e)
            print(f"  ERROR: {e}")

        refused = (not llm_error) and is_refusal(answer)
        answerable_results.append({
            "question": tc["question"], "hit": hit, "strict_hit": strict, "rank": rank,
            "answer": answer, "keyword_score": kw, "all_keywords": kw == 1.0,
            "false_refusal": refused, "citation_hit": citation_hit,
            "citations": [{k: v for k, v in c.items() if k != "content"} for c in citations],
            "llm_error": llm_error, "error": err,
        })
        print(f"  Hit@{TOP_K}: {hit} (rank {rank}, strict page: {strict}) | keywords: {kw:.2f} "
              f"| cited right source: {citation_hit} | refused: {refused}")
        time.sleep(PAUSE_SECONDS)

    # ---------- unanswerable ----------
    for j, q in enumerate(UNANSWERABLE, start=1):
        print(f"\n[unanswerable {j}/{len(UNANSWERABLE)}] {q}")
        try:
            result = ask_with_retry(q)
            answer = result["answer"]
            llm_error = "error" in result
            refused = (not llm_error) and is_refusal(answer)
        except Exception as e:
            answer, refused, llm_error = f"ERROR: {e}", False, True
        refusal_results.append({"question": q, "answer": answer, "refused": refused, "llm_error": llm_error})
        print(f"  Refused correctly: {refused}")
        time.sleep(PAUSE_SECONDS)

    # ---------- aggregate ----------
    n, m = len(answerable_results), len(refusal_results)

    def rate(rows, key):
        return sum(bool(r[key]) for r in rows) / len(rows) if rows else 0

    summary = {
        "n_answerable": n,
        "n_unanswerable": m,
        "top_k": TOP_K,
        "hit_relaxed": rate(answerable_results, "hit"),
        "hit_strict_page": rate(answerable_results, "strict_hit"),
        "answer_all_keywords": rate(answerable_results, "all_keywords"),
        "answer_avg_keyword_score": (sum(r["keyword_score"] for r in answerable_results) / n) if n else 0,
        "citation_source_accuracy": rate(answerable_results, "citation_hit"),
        "false_refusal_rate": rate(answerable_results, "false_refusal"),
        "refusal_accuracy": rate(refusal_results, "refused"),
        "llm_errors": sum(r["llm_error"] for r in answerable_results) + sum(r["llm_error"] for r in refusal_results),
    }

    print("\n" + "=" * 58)
    print("EVAL SUMMARY")
    print("=" * 58)
    print(f"Answerable: {n} | Unanswerable: {m} | LLM errors: {summary['llm_errors']}")
    print(f"Hit@{TOP_K} (relaxed):            {pct(summary['hit_relaxed'])}")
    print(f"Hit@{TOP_K} (strict page):        {pct(summary['hit_strict_page'])}")
    print(f"Answer accuracy (all keywords):   {pct(summary['answer_all_keywords'])}")
    print(f"Avg keyword score:                {pct(summary['answer_avg_keyword_score'])}")
    print(f"Citation points to right source:  {pct(summary['citation_source_accuracy'])}")
    print(f"False refusal rate:               {pct(summary['false_refusal_rate'])}")
    print(f"Refusal accuracy:                 {pct(summary['refusal_accuracy'])}")

    print("\nREADME table (copy-paste):\n")
    print("| Metric | Value |\n|---|---|")
    print(f"| Hit@{TOP_K} (retrieval) | {pct(summary['hit_relaxed'])} |")
    print(f"| Answer accuracy | {pct(summary['answer_all_keywords'])} |")
    print(f"| Citation accuracy | {pct(summary['citation_source_accuracy'])} |")
    print(f"| Refusal accuracy | {pct(summary['refusal_accuracy'])} |")
    print(f"| Eval set | {n} answerable + {m} unanswerable questions |")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "answerable": answerable_results, "unanswerable": refusal_results},
                  f, indent=2)
    print("\nFull results saved to eval_results.json")


if __name__ == "__main__":
    run_eval()