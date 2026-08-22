"""Pipeline RAG: retrieve -> susun prompt -> LLM jawab + sitasi sumber.

Upgrade v2: Conversation memory (N-turn) — user bisa follow-up tanpa ulang konteks.
"""
import time
from collections import defaultdict, deque

from src import config, llm, retriever

SYSTEM = (
    "Anda asisten regulasi UMKM Indonesia. Jawab HANYA berdasarkan KONTEKS "
    "yang diberikan. Jika konteks tidak memuat jawaban, katakan: 'Maaf, "
    "informasi tidak ditemukan dalam dokumen.' Jangan mengarang. Jawab ringkas "
    "dalam Bahasa Indonesia dan sebutkan dasar hukumnya bila ada."
)

# ── Conversation memory: session_id -> deque N-turn terakhir ─────────
_MEMORY = defaultdict(lambda: deque(maxlen=config.CONVERSATION_TURNS))


def _history_text(session_id):
    """Kembalikan riwayat percakapan sebagai teks (untuk prompt)."""
    turns = _MEMORY.get(session_id)
    if not turns:
        return ""
    lines = [f"User: {q}\nAsisten: {a}" for q, a in turns]
    return "RIWAYAT PERCAKAPAN:\n" + "\n".join(lines) + "\n\n"


def _build_prompt(question, chunks, session_id):
    """Susun prompt dengan konteks + riwayat percakapan."""
    ctx = "\n\n".join(
        f"[{i}] (Sumber: {c['source']}, hal.{c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    hist = _history_text(session_id)
    return f"{hist}KONTEKS:\n{ctx}\n\nPERTANYAAN: {question}\n\nJAWABAN:"


def answer(question, strategy="naive", top_k=None, session_id="default"):
    """Return answer, sources, provider, and split latency measurements."""
    t0 = time.perf_counter()
    kw = {} if top_k is None else {"top_k": top_k}

    retrieve_t0 = time.perf_counter()
    chunks = retriever.retrieve(question, strategy=strategy, **kw)
    retrieval_latency_ms = round((time.perf_counter() - retrieve_t0) * 1000, 1)

    prompt = _build_prompt(question, chunks, session_id)
    llm_t0 = time.perf_counter()
    text, provider = llm.generate(prompt, system=SYSTEM)
    llm_latency_ms = round((time.perf_counter() - llm_t0) * 1000, 1)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    sources = [
        {"source": c["source"], "page": c["page"], "score": round(c["score"], 4)}
        for c in chunks
    ]

    # Simpan ke memory (question + answer pair)
    _MEMORY[session_id].append((question, text.strip()))

    return {
        "question": question,
        "answer": text.strip(),
        "provider": provider,
        "strategy": strategy,
        "sources": sources,
        "contexts": [c["text"] for c in chunks],  # dipakai eval RAGAS
        "latency_ms": latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        "llm_latency_ms": llm_latency_ms,
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Apa saja kriteria usaha mikro?"
    r = answer(q)
    print(f"Q: {r['question']}")
    print(f"A [{r['provider']}, {r['latency_ms']}ms]: {r['answer']}\n")
    print("Sumber:")
    for s in r["sources"]:
        print(f"  - {s['source']} hal.{s['page']} (score {s['score']})")