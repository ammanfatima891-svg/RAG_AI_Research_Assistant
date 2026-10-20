# app.py
import html
import time
import streamlit as st

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")

from generate_answers import generate_answer, get_connection, REFUSAL_TEXT  # noqa: E402
from ingest import ingest_pdf  # noqa: E402

MAX_UPLOAD_MB = 15

# ---------------------------------------------------------------------------
# Look and feel (styling only, no behaviour lives in this block)
# ---------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #14213D;
  --muted: #5B6678;
  --paper: #F7F8FA;
  --card: #FFFFFF;
  --line: #E1E5EC;
  --accent: #2F5BEA;
  --accent-soft: #E8EEFF;
}

/* page width: keeps answer lines readable */
.block-container { max-width: 880px; padding-top: 2.4rem; padding-bottom: 6rem; }
footer { visibility: hidden; }

/* text (scoped, so icon fonts keep working) */
.stMarkdown p, .stMarkdown li, [data-testid="stChatMessage"] p,
[data-testid="stCaptionContainer"], [data-testid="stSidebar"] label,
.stButton button, [data-testid="stChatInput"] textarea {
  font-family: 'Instrument Sans', system-ui, sans-serif;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { line-height: 1.65; }

/* header block */
.hero { margin: 0 0 1.6rem 0; }
.hero h1 {
  font-family: 'Newsreader', Georgia, serif;
  font-weight: 600; font-size: 2.6rem; line-height: 1.1;
  letter-spacing: -0.015em; color: var(--ink); margin: 0 0 .55rem 0; padding: 0;
}
.hero p { font-size: 1.05rem; color: var(--muted); margin: 0; max-width: 34rem; line-height: 1.55; }

/* empty state */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin: 1.2rem 0 2rem 0; counter-reset: s; }
.step { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .95rem 1rem; counter-increment: s; }
.step::before {
  content: counter(s); display: inline-flex; align-items: center; justify-content: center;
  width: 1.5rem; height: 1.5rem; border-radius: 50%; margin-bottom: .55rem;
  background: var(--accent-soft); color: var(--accent); font-weight: 600; font-size: .85rem;
}
.step b { display: block; color: var(--ink); font-size: .98rem; margin-bottom: .2rem; }
.step span { color: var(--muted); font-size: .88rem; line-height: 1.45; }
@media (max-width: 700px) { .steps { grid-template-columns: 1fr; } .hero h1 { font-size: 2.1rem; } }

/* chat messages */
[data-testid="stChatMessage"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 1rem 1.15rem; margin-bottom: .8rem;
}

/* source passages */
[data-testid="stExpander"] {
  background: var(--card); border: 1px solid var(--line);
  border-left: 4px solid var(--accent); border-radius: 10px; margin-top: .45rem;
}
[data-testid="stExpander"] summary { font-size: .92rem; font-weight: 500; color: var(--ink); }

/* inputs and buttons */
.stButton button { border-radius: 10px; border: 1px solid var(--line); font-weight: 500; }
.stButton button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
[data-testid="stChatInput"] { border-radius: 14px; }

/* sidebar */
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.brand { font-family: 'Newsreader', Georgia, serif; font-weight: 600; font-size: 1.45rem; color: var(--ink); margin-bottom: 1.1rem; }
.side-title { font-weight: 600; font-size: .95rem; color: var(--ink); margin: .2rem 0 .45rem 0; }
.hint { color: var(--muted); font-size: .82rem; margin: -.2rem 0 .7rem 0; }
.stats { display: flex; gap: .7rem; margin-bottom: .8rem; }
.stat { flex: 1; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: .55rem .75rem; }
.stat b { display: block; font-size: 1.25rem; color: var(--ink); line-height: 1.2; }
.stat span { font-size: .78rem; color: var(--muted); }
.doclist { max-height: 340px; overflow-y: auto; background: var(--card); border: 1px solid var(--line); border-radius: 10px; }
.doc { display: flex; justify-content: space-between; gap: .6rem; padding: .5rem .75rem; border-bottom: 1px solid var(--line); font-size: .86rem; }
.doc:last-child { border-bottom: none; }
.doc-name { color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-n { color: var(--muted); flex-shrink: 0; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data helpers (unchanged behaviour)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def list_indexed_sources():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception:
        return None


def render_answer(answer):
    if answer == REFUSAL_TEXT:
        st.info("I couldn't find this in your documents. Try rephrasing, or upload a file that covers it.")
    else:
        st.markdown(answer)


def render_citations(citations):
    for c in citations:
        label = f"[{c['chunk_number']}]  {c['source_file']}, page {c['page_number']}"
        with st.expander(label):
            st.caption(f"Relevance score {c['rerank_score']}")
            st.write(c["content"])


def meta_line(n_citations, seconds):
    if n_citations:
        noun = "passage" if n_citations == 1 else "passages"
        return f"Based on {n_citations} {noun}, answered in {seconds:.1f}s"
    return f"Answered in {seconds:.1f}s"


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">Document Q&amp;A</div>', unsafe_allow_html=True)

    # upload runs before the list below so the list shows new files right away
    st.markdown('<div class="side-title">Add documents</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
    )
    st.markdown(f'<div class="hint">PDF files, up to {MAX_UPLOAD_MB} MB each.</div>', unsafe_allow_html=True)
    if st.button("Index uploaded files", disabled=not files, type="primary", use_container_width=True):
        for f in files:
            if f.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"{f.name}: larger than {MAX_UPLOAD_MB} MB")
                continue
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    n = ingest_pdf(f.getvalue(), f.name)
                    st.success(f"{f.name}: {n} chunks indexed")
                except ValueError as e:
                    st.warning(f"{f.name}: {e}")
                except Exception as e:
                    st.error(f"{f.name}: failed ({type(e).__name__}: {str(e)[:150]})")
        list_indexed_sources.clear()

    st.divider()
    st.markdown('<div class="side-title">Your library</div>', unsafe_allow_html=True)
    sources = list_indexed_sources()
    if sources is None:
        st.error("Could not reach the database.")
    elif not sources:
        st.info("No documents yet. Upload a PDF above to get started.")
    else:
        total = sum(n for _, n in sources)
        st.markdown(
            f'<div class="stats">'
            f'<div class="stat"><b>{len(sources)}</b><span>documents</span></div>'
            f'<div class="stat"><b>{total:,}</b><span>passages</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        rows = "".join(
            f'<div class="doc"><span class="doc-name" title="{html.escape(name)}">{html.escape(name)}</span>'
            f'<span class="doc-n">{n}</span></div>'
            for name, n in sources
        )
        st.markdown(f'<div class="doclist">{rows}</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>Ask your documents</h1>'
    '<p>Answers come only from your PDFs, and every claim points to the page it came from.</p></div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        '<div class="steps">'
        '<div class="step"><b>Add your PDFs</b><span>Use the sidebar on the left, then click Index.</span></div>'
        '<div class="step"><b>Ask a question</b><span>Plain language works best, for example "What datasets were used?"</span></div>'
        '<div class="step"><b>Check the sources</b><span>Open a source to read the exact passage behind the answer.</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_answer(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])
        else:
            st.markdown(m["content"])
        if m.get("citations"):
            render_citations(m["citations"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.time()
        try:
            with st.spinner("Searching your documents..."):
                result = generate_answer(question)
        except Exception as e:
            st.error(f"Something went wrong ({type(e).__name__}). Check the database connection and try again.")
            st.stop()

        answer, citations = result["answer"], result["citations"]
        latency = time.time() - start
        meta = meta_line(len(citations), latency)

        render_answer(answer)
        st.caption(meta)
        if citations:
            render_citations(citations)
        print(f"[query] latency={latency:.2f}s cited={len(citations)} q={question!r}")  # visible in Streamlit Cloud logs

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations, "meta": meta}
    )