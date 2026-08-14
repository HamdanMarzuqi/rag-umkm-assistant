"""Streamlit demo untuk RAG UMKM Assistant.

Jalankan:
    .venv/Scripts/python.exe -m streamlit run app.py

UI: input pertanyaan + pilih strategi retrieval,
tampilkan jawaban + sumber + metrik latensi.
"""

import os
import sys
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

# ── Cache pipeline agar tidak reload tiap interaksi ──
@st.cache_resource(show_spinner=False)
def _warm(strategy: str):
    """Panggil retriever sekali untuk warm-up (model load)."""
    rag.answer("warmup", strategy=strategy)
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

# ── Header & contoh pertanyaan ──
st.title("RAG UMKM Assistant")
st.caption("Pengetahuan: 32 dokumen regulasi UMKM (Kemenkop/OSS). Evaluator: RAGAS, model: multilingual-e5-small.")

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

# ── Input utama ──
question = st.text_input(
    "Pertanyaan",
    value=st.session_state.get("pending_question", ""),
    placeholder="contoh: Berapa batas modal usaha menengah?",
)
run = st.button("Tanyakan", type="primary")

if run and question.strip():
    with st.spinner(f"Menjawab dengan strategi '{strategy}' ..."):
        r = rag.answer(question.strip(), strategy=strategy, top_k=top_k)

    tab_answer, tab_sources = st.tabs(["Jawaban", "Sumber"])
    with tab_answer:
        st.markdown(r["answer"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"{r['latency_ms']/1000:.2f} s")
        c2.metric("Retrieval", f"{r['retrieval_latency_ms']/1000:.2f} s")
        c3.metric("LLM", f"{r['llm_latency_ms']/1000:.2f} s")
        st.caption(f"Provider: {r['provider']} · Strategi: {r['strategy']}")

    with tab_sources:
        for i, s in enumerate(r["sources"], 1):
            with st.expander(f"[{i}] {s['source']} — hal.{s['page']} (score {s['score']:.3f})"):
                st.text(r["contexts"][i-1][:1500])
