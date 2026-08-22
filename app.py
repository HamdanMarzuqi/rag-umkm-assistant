"""Streamlit demo untuk RAG UMKM Assistant.

Upgrade v2: Conversation memory (N-turn) — user bisa follow-up tanpa ulang konteks.

Jalankan:
    .venv/Scripts/python.exe -m streamlit run app.py

UI: chat interface + pilih strategi retrieval,
tampilkan jawaban + sumber + metrik latensi.
"""

import os
import sys
import uuid
from pathlib import Path

# Bootstrap project path so 'src' package terbaca
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src import rag

st.set_page_config(
    page_title="RAG UMKM Assistant",
    page_icon=".",
    layout="centered",
)

# ── Session ID untuk conversation memory ──────────────────────────────
if "sid" not in st.session_state:
    st.session_state.sid = f"ui_{uuid.uuid4().hex[:8]}"

# ── Chat history (untuk UI) ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Cache pipeline agar tidak reload tiap interaksi ──
@st.cache_resource(show_spinner=False)
def _warm(strategy: str):
    """Panggil retriever sekali untuk warm-up (model load)."""
    rag.answer("warmup", strategy=strategy, session_id="warmup")
    return True


STRATEGIES = ["naive", "rerank", "hyde", "hybrid"]

# ── Sidebar: strategi + benchmark ringkas ──
with st.sidebar:
    st.markdown("**RAG UMKM Assistant**")
    st.caption("Retrieval-Augmented Generation untuk regulasi UMKM Indonesia.")
    st.divider()
    strategy = st.radio(
        "Strategi retrieval",
        STRATEGIES,
        index=STRATEGIES.index("rerank"),
        help="naive: cosine. rerank: LLM urutkan top-20. hyde: hipotetis jawab. hybrid: BM25+semantic.",
    )
    top_k = st.slider("top_k", min_value=2, max_value=10, value=5)
    _warm(strategy)
    st.divider()

    bm_path = ROOT / "results" / "benchmark.md"
    if bm_path.exists():
        st.markdown("**Benchmark (29 pertanyaan)**")
        st.code(bm_path.read_text(encoding="utf-8").split("## Delta")[0], language=None)

    st.divider()
    st.caption(f"Session: `{st.session_state.sid}`")
    if st.button("🗑️ Reset percakapan"):
        st.session_state.messages = []
        # Clear memory di rag.py (restart module)
        import importlib
        import src.rag as rag_module
        importlib.reload(rag_module)
        st.rerun()

# ── Header & contoh pertanyaan ──
st.title("RAG UMKM Assistant")
st.caption("Pengetahuan: 32 dokumen regulasi UMKM (Kemenkop/OSS). Evaluator: RAGAS, model: multilingual-e5-small.")
st.caption("💡 **Upgrade v2:** Conversation memory — tanya follow-up tanpa ulang konteks.")

EXAMPLES = [
    "Apa saja kriteria usaha mikro?",
    "Berapa batas modal usaha kecil?",
    "Apa itu kemitraan distribusi dan keagenan?",
    "Kapan laporan KUR harus disampaikan?",
    "Apa tujuan pemberdayaan UMKM?",
]
st.markdown("**Coba pertanyaan:**")
for i, ex in enumerate(EXAMPLES):
    if st.button(ex, key=f"ex_{i}"):
        st.session_state["pending_question"] = ex

# ── Render chat history ──
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            srcs = ", ".join(sorted({s["source"] for s in m["sources"]}))
            st.caption(f"📚 Sumber: {srcs}")

# ── Input utama ──
question = st.text_input(
    "Pertanyaan",
    value=st.session_state.get("pending_question", ""),
    placeholder="contoh: Berapa batas modal usaha menengah?",
    key="question_input",
)
run = st.button("Tanyakan", type="primary")

if run and question.strip():
    # Tambah user message ke history
    st.session_state.messages.append({"role": "user", "content": question.strip()})

    with st.spinner(f"Menjawab dengan strategi '{strategy}' ..."):
        r = rag.answer(question.strip(), strategy=strategy, top_k=top_k, session_id=st.session_state.sid)

    # Tambah assistant message ke history
    st.session_state.messages.append({
        "role": "assistant",
        "content": r["answer"],
        "sources": r["sources"],
    })

    # Re-render
    st.rerun()
