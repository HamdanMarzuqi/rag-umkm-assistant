# RAG Benchmark — UMKM Regulation Assistant

Evaluasi 4 strategi retrieval pada 29 pertanyaan golden dataset.
Evaluator: RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) via local-claude (9Router).
Embedding: intfloat/multilingual-e5-small (CPU). LLM: Groq llama-3.3-70b / local-claude fallback.

## Hasil Utama

| Strategy | n | Faithfulness | Answer Rel | Context Prec | Context Rec | Latency (ms) |
|---|---|---|---|---|---|---|
| Naive RAG (baseline) | 29 | 0.708 | 0.932 | 0.751 | 0.897 | 23,032 |
| + Reranking (LLM-as-reranker) | 29 | 0.668 | 0.893 | **0.825** | 0.897 | 33,035 |
| + HyDE | 29 | 0.630 | 0.886 | 0.667 | 0.793 | 32,908 |
| + Hybrid (BM25+semantic/RRF) | 29 | **0.717** | 0.848 | 0.753 | **0.897** | **18,215** |

## Delta vs Baseline (Naive)

| Strategy | Faithfulness | Answer Rel | Context Prec | Context Rec | Latency |
|---|---|---|---|---|---|
| + Reranking | -0.040 | -0.039 | **+0.074** | 0.000 | +10,003ms |
| + HyDE | -0.078 | -0.046 | -0.084 | -0.104 | +9,876ms |
| + Hybrid | **+0.009** | -0.084 | +0.002 | 0.000 | **-4,817ms** |

## Latency Breakdown

| Strategy | Avg Retrieval (ms) | Avg LLM (ms) | Avg Total (ms) |
|---|---|---|---|
| Naive | 2,799 | 20,233 | 23,032 |
| Reranking | 19,753 | 13,281 | 33,035 |
| HyDE | 20,308 | 12,600 | 32,908 |
| Hybrid | **1,157** | 17,058 | **18,215** |

> Hybrid retrieval paling cepat karena BM25 (pure Python, in-memory) sangat cepat dan tidak membutuhkan LLM call tambahan.
> Reranking dan HyDE mahal di retrieval karena keduanya membutuhkan 1 LLM call extra sebelum menjawab.

## Catatan Metodologi

- **Reranking**: LLM-as-reranker (RankGPT-style) — retrieve top-20 dari Qdrant, LLM urutkan ulang, ambil top-5.
  Tidak menggunakan BGE cross-encoder karena kendala download model (jaringan). Pendekatan ini valid
  secara akademis (RankGPT paper, LLM-as-judge).
- **HyDE**: LLM generate jawaban hipotetis → embed sebagai "passage:" → retrieve top-5 dari Qdrant.
- **Hybrid**: BM25 top-20 (rank-bm25, dari seluruh 771 chunk) + Semantic top-20 (Qdrant cosine),
  digabung via Reciprocal Rank Fusion (RRF, k=60). BM25 index di-cache setelah build pertama.
- RAGAS parser warnings (non-fatal) terjadi pada beberapa sampel — nilai NaN di-drop otomatis.
- Semua eval menggunakan golden_dataset.json (29 pertanyaan, verified manual).

## Analisis Trade-off

**Reranking menang di Context Precision (+7.4 poin)** — chunk yang masuk ke prompt lebih relevan.
Trade-off: +10 detik latency untuk 1 extra LLM call (reranking 20 kandidat).

**Hybrid menang di Latency (-4.8 detik dari naive)** dan marginally lebih baik di faithfulness.
Context Recall sama dengan naive (0.897) — RRF berhasil mempertahankan recall naive sambil lebih cepat.
Answer relevancy turun (-0.084) — kemungkinan karena BM25 kadang memasukkan chunk keyword-match
yang relevan secara leksikal tapi kurang relevan secara semantik untuk menjawab pertanyaan.

**HyDE tidak efektif di dataset ini** — pertanyaan faktual pendek (siapa, berapa, kapan) tidak banyak
terbantu oleh hypothetical document embedding. HyDE lebih efektif untuk pertanyaan konseptual/abstrak.

**Kesimpulan**:
- Jika prioritas **akurasi retrieval** → pakai **Reranking** (context precision terbaik)
- Jika prioritas **kecepatan** → pakai **Hybrid** (latency terbaik, recall sama dengan naive)
- HyDE tidak direkomendasikan untuk domain regulasi dengan pertanyaan faktual

## Provider Distribution

Semua strategi: 28-29 query via router_local (9Router/local-claude), 0-1 via Groq. Groq rate limit
menyebabkan fallback ke router_local untuk sebagian besar query.
