"""Retriever: strategi pengambilan chunk dari Qdrant.

Fase 2: mode `naive` (embed query -> top-k cosine).
Fase 4.1: mode `rerank` (LLM-as-reranker, RankGPT-style) — retrieve top-20,
          LLM urutkan ulang, ambil top-5. Tidak butuh download model reranker.
Fase 4.2: mode `hyde` (HyDE — Hypothetical Document Embeddings).
Fase 4.3: mode `hybrid` (BM25 + semantic via Reciprocal Rank Fusion).

Embedding model di-cache (singleton) supaya tak reload tiap query.
"""
from functools import lru_cache

from src import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _client():
    client, _mode = config.get_qdrant_client()
    return client


def embed_query(text):
    # e5 minta prefix "query: " untuk pertanyaan
    return _model().encode("query: " + text, normalize_embeddings=True).tolist()


def naive(question, top_k=config.TOP_K):
    """Return list dict {text, source, page, score}."""
    qv = embed_query(question)
    hits = _client().query_points(config.COLLECTION, query=qv, limit=top_k).points
    return [
        {
            "text": h.payload["text"],
            "source": h.payload["source"],
            "page": h.payload["page"],
            "score": float(h.score),
        }
        for h in hits
    ]


# ── 4.1 Reranking (LLM-as-reranker, RankGPT-style) ───────────────────
def rerank(question, top_k=config.TOP_K, fetch_k=20):
    """Retrieve top-`fetch_k` → LLM rerank → top-`top_k`.

    Menggunakan RankGPT-style reranking: LLM diberi daftar kandidat +
    pertanyaan, diminta kembalikan urutan nomor dari paling relevan.
    Tidak butuh download model reranker (pakai LLM yang sudah ada di stack).

    Trade-off: +latency ~1 LLM call vs potential improvement context_precision.
    """
    candidates = naive(question, top_k=fetch_k)
    if not candidates:
        return []

    from src import llm
    numbered = "\n".join(
        f"[{i + 1}] {c['text'][:300]}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"Pertanyaan: {question}\n\n"
        f"Berikut {len(candidates)} kandidat dokumen regulasi UMKM:\n{numbered}\n\n"
        f"Urutkan nomor dokumen dari yang PALING relevan ke yang PALING tidak relevan "
        f"terhadap pertanyaan di atas. "
        f"Kembalikan HANYA nomor-nomor dalam urutan, dipisahkan koma. "
        f"Contoh format: 3,1,5,2,4,6,7,8,9,10"
    )
    ranking_text, _ = llm.generate(
        prompt,
        system="Kamu sistem reranking dokumen. Kembalikan HANYA daftar nomor dipisah koma, tanpa teks lain.",
    )

    # Parse urutan dari LLM
    try:
        raw = ranking_text.strip().split(",")
        order = [int(x.strip()) - 1 for x in raw if x.strip().lstrip("-").isdigit()]
        # Deduplicate, filter valid index
        seen: set = set()
        order = [
            i for i in order
            if 0 <= i < len(candidates) and i not in seen and not seen.add(i)  # type: ignore[func-returns-value]
        ]
    except Exception:
        order = []

    # Susun hasil rerank
    reranked = [candidates[i] for i in order[:top_k]]
    # Fallback: tambah dari urutan asli bila LLM parse kurang
    if len(reranked) < top_k:
        in_result = {id(c) for c in reranked}
        for c in candidates:
            if id(c) not in in_result:
                reranked.append(c)
                in_result.add(id(c))
            if len(reranked) >= top_k:
                break
    return reranked


# ── 4.2 HyDE (Hypothetical Document Embeddings) ────────────────────────
def hyde(question, top_k=config.TOP_K):
    """LLM generate hypothetical answer → embed → retrieve."""
    from src import llm
    prompt = (
        "Buat jawaban hipotetis SINGKAT (2-3 kalimat) untuk pertanyaan berikut. "
        "Jangan mengatakan kamu tidak tahu — buat jawaban yang masuk akal "
        "seolah-olah kamu ahli regulasi UMKM Indonesia.\n\n"
        f"Pertanyaan: {question}\n\nJawaban hipotetis:"
    )
    hyp, _ = llm.generate(prompt, system="Kamu ahli regulasi UMKM Indonesia.")
    # embed hypothetical doc (prefix "passage:" karena ini dokumen, bukan query)
    hyp_vec = _model().encode("passage: " + hyp, normalize_embeddings=True).tolist()
    hits = _client().query_points(config.COLLECTION, query=hyp_vec, limit=top_k).points
    return [
        {
            "text": h.payload["text"],
            "source": h.payload["source"],
            "page": h.payload["page"],
            "score": float(h.score),
        }
        for h in hits
    ]


# ── 4.3 Hybrid (BM25 + Semantic via RRF) ───────────────────────
@lru_cache(maxsize=1)
def _bm25_index():
    """Build BM25 index dari seluruh chunk di Qdrant (cached).

    Scroll semua point dari Qdrant, tokenisasi teks (split whitespace),
    lalu buat BM25Okapi index. Cached supaya hanya di-build sekali.
    Return (bm25, all_docs) di mana all_docs = list dict {text,source,page}.
    """
    from rank_bm25 import BM25Okapi
    client = _client()
    all_docs = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=config.COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, offset = result
        for p in points:
            all_docs.append({
                "text": p.payload["text"],
                "source": p.payload["source"],
                "page": p.payload["page"],
                "score": 0.0,
            })
        if offset is None:
            break
    tokenized = [d["text"].lower().split() for d in all_docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, all_docs


def hybrid(question, top_k=config.TOP_K, fetch_k=20, rrf_k=60):
    """Gabung semantic (Qdrant cosine) + BM25 via Reciprocal Rank Fusion.

    RRF score = 1/(rrf_k + rank_semantic) + 1/(rrf_k + rank_bm25)
    Lebih robust dari weighted sum karena tidak perlu normalisasi skor.

    Args:
        question: teks pertanyaan
        top_k: jumlah hasil akhir
        fetch_k: kandidat top-N dari tiap jalur (semantic & BM25)
        rrf_k: konstanta RRF (default 60, rekomendasi paper asli)
    """
    # Jalur 1: semantic top-fetch_k dari Qdrant
    qv = embed_query(question)
    sem_hits = _client().query_points(
        config.COLLECTION, query=qv, limit=fetch_k
    ).points
    sem_docs = [
        {
            "text": h.payload["text"],
            "source": h.payload["source"],
            "page": h.payload["page"],
            "score": float(h.score),
        }
        for h in sem_hits
    ]

    # Jalur 2: BM25 top-fetch_k dari seluruh corpus
    bm25, all_docs = _bm25_index()
    tokens = question.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    # Index top-fetch_k BM25
    top_bm25_idx = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:fetch_k]
    bm25_docs = [all_docs[i] for i in top_bm25_idx]

    # RRF fusion: buat lookup doc_id → rrf_score
    # Gunakan (source, page, text[:50]) sebagai key unik per chunk
    def _key(d):
        return (d["source"], d["page"], d["text"][:50])

    rrf: dict = {}
    for rank, d in enumerate(sem_docs):
        k = _key(d)
        rrf[k] = rrf.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, d in enumerate(bm25_docs):
        k = _key(d)
        rrf[k] = rrf.get(k, 0.0) + 1.0 / (rrf_k + rank + 1)

    # Kumpulkan semua kandidat unik
    seen_keys: set = set()
    candidates = []
    for d in sem_docs + bm25_docs:
        k = _key(d)
        if k not in seen_keys:
            seen_keys.add(k)
            candidates.append(d)

    # Sort by RRF score, ambil top_k
    candidates.sort(key=lambda d: rrf[_key(d)], reverse=True)
    results = []
    for d in candidates[:top_k]:
        results.append({**d, "score": round(rrf[_key(d)], 6)})
    return results


STRATEGIES = {"naive": naive, "rerank": rerank, "hyde": hyde, "hybrid": hybrid}


def retrieve(question, strategy="naive", top_k=config.TOP_K):
    if strategy not in STRATEGIES:
        raise ValueError(f"strategi tidak dikenal: {strategy}. Ada: {list(STRATEGIES)}")
    return STRATEGIES[strategy](question, top_k=top_k)


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "kriteria usaha mikro"
    for i, r in enumerate(retrieve(q), 1):
        print(f"[{i}] {r['score']:.3f} {r['source'][:40]} hal.{r['page']}")
        print("   ", r["text"][:120].replace("\n", " "))
