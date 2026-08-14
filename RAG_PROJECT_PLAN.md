# RAG Knowledge Assistant + Retrieval Evaluation Study

> Project portofolio untuk posisi **AI Engineer | LLM Integration & Agentic AI Developer**
> Owner: Hamdan Akbar Marzuqi
> Target: project yang dilirik HRD teknis — RAG betulan (vector DB) + evaluasi terukur.
> **Constraint: TANPA GPU** — semua jalan di CPU + API gratis (Groq/Gemini).

---

## 0. Tujuan & Killer Sentence

Bangun sistem RAG produksi untuk domain **panduan/regulasi UMKM Indonesia**, lalu **benchmark 3 strategi retrieval** dengan golden dataset. Buktikan peningkatan kualitas dengan angka.

**Kalimat target untuk CV/interview (diisi setelah benchmark):**
> "Built a RAG system for UMKM regulation docs; benchmarked naive vs reranking vs HyDE vs hybrid on a 29-query golden dataset with RAGAS — reranking improved context precision from 75.1% to 82.5% (+7.4 pts) at +10s latency; hybrid achieved faster retrieval (1,157ms vs 2,799ms) with equal context recall."

**Kenapa domain UMKM:** Hamdan owner Malika Kebab → narasi personal ("saya bikin karena saya sendiri kesulitan cari info izin/pajak UMKM"). Cerita personal = poin plus di mata HRD.

---

## 1. Tech Stack (CPU-friendly, standar 2026)

| Layer | Tool | Catatan CPU |
|---|---|---|
| Orchestration | **LangChain** | Standar industri, banyak contoh |
| Backend/API | **FastAPI** | ← skill yang WizAI & banyak lowongan AI minta |
| Vector DB | **Qdrant** (Docker lokal) | Production-grade, gratis self-host |
| Embedding | **intfloat/multilingual-e5-small** (via sentence-transformers) | ~470MB, jalan mulus di CPU, bagus utk Bahasa Indonesia |
| LLM | **Groq (Llama 3.3 70B)** utama, **Gemini** fallback | API gratis, tidak perlu GPU lokal |
| Reranker | **LLM-as-reranker** (RankGPT-style via local-claude) | Tidak butuh download model; retrieve top-20 → LLM urutkan → top-5 |
| Keyword search | **BM25** (rank-bm25) | Utk hybrid search |
| Evaluation | **RAGAS** | Framework eval RAG populer |
| PDF extract | **pymupdf** | Sudah dikuasai Hamdan |
| Demo UI | **Streamlit** | Tercepat utk demo |
| Deploy | **Docker + Fly.io** | Sudah dikuasai Hamdan |

**API key gratis yang perlu disiapkan:** Groq (console.groq.com), Gemini (aistudio.google.com). Simpan di `.env`, JANGAN commit.

---

## 2. Struktur Folder

```
rag-umkm-assistant/
├── data/
│   ├── raw/                 # PDF regulasi UMKM asli
│   └── processed/           # chunks JSON
├── src/
│   ├── ingest.py            # ekstrak PDF → chunk → embed → Qdrant
│   ├── retriever.py         # strategi: naive / rerank / hyde / hybrid
│   ├── rag.py               # pipeline retrieve → prompt → LLM
│   ├── llm.py               # wrapper Groq + Gemini fallback
│   └── config.py            # env, konstanta
├── eval/
│   ├── golden_dataset.json  # 50 (pertanyaan, jawaban, sumber)
│   ├── build_golden.py      # bantu generate draft golden set via LLM
│   └── run_eval.py          # RAGAS + tabel perbandingan
├── api/
│   └── main.py              # FastAPI /query endpoint
├── app.py                   # Streamlit demo
├── results/
│   └── benchmark.md         # tabel hasil (ISI INI = inti project)
├── docker-compose.yml       # Qdrant + app
├── requirements.txt
├── .env.example
├── .gitignore               # WAJIB: .env, __pycache__, data/raw besar
└── README.md                # arsitektur + hasil benchmark + cara jalankan
```

---

## 3. Fase Pengerjaan (checklist, ~3-4 minggu solo)

### FASE 1 — Data & Ingestion (minggu 1)
- [x] 1.1 Kumpulkan 20-50 dokumen regulasi/panduan UMKM (PDF resmi: Kemenkop, OSS, panduan pajak UMKM) — 32 PDF dari jdih.umkm.go.id
- [x] 1.2 Setup project: venv, `requirements.txt`, `.gitignore`, `.env.example`
- [x] 1.3 Qdrant — pakai **embedded on-disk** (Docker pull gagal: CDN reset; auto-detect server bila hidup)
- [x] 1.4 `ingest.py`: ekstrak teks PDF (pymupdf) → chunking (fixed 512 token + overlap 64) — 771 chunk; 4 PDF scan tanpa teks dilewati (perlu OCR)
- [x] 1.5 Embed tiap chunk (multilingual-e5-small) → upsert ke Qdrant — 771 vektor tersimpan
- [x] 1.6 Verifikasi: query manual, retrieval kembalikan chunk relevan — top-3 skor ~0.89 (PP 7/2021, PermenKop)

**Milestone F1:** ✅ dokumen terindeks di Qdrant, retrieval manual jalan.

> Catatan: 4 PDF scan (09, 18, 19, 22) belum ter-index — perlu OCR (skill `ocr-and-documents`) bila mau dimasukkan.

### FASE 2 — RAG Pipeline Dasar / Baseline (minggu 1-2)
- [x] 2.1 `llm.py`: wrapper Groq (utama) + Gemini (fallback) + 9Router local-claude (jaring pengaman terakhir) — triple fallback otomatis
- [x] 2.2 `retriever.py`: mode `naive` (embed query → top-k cosine dari Qdrant)
- [x] 2.3 `rag.py`: retrieve top-k → susun prompt → LLM jawab + sumber — jawaban grounded, anti-halusinasi terbukti
- [x] 2.4 `api/main.py`: FastAPI `/query` + `/health` — teruji via curl
- [x] 2.5 Test manual — 5+ pertanyaan terjawab grounded, pertanyaan di luar dokumen ditolak

**Milestone F2:** Naive RAG jalan end-to-end via API. Ini baseline.

### FASE 3 — Evaluation Harness (minggu 2-3) ← INTI
- [x] 3.1 `build_golden.py`: generate (pertanyaan, jawaban ideal, chunk sumber) dari dokumen → verifikasi manual tiap entri
- [x] 3.2 Simpan `golden_dataset.json` — 29 entri verified
- [x] 3.3 `run_eval.py`: jalankan golden dataset lewat pipeline → kumpulkan (question, answer, contexts, ground_truth)
- [x] 3.4 Ukur RAGAS: faithfulness, answer_relevancy, context_precision, context_recall — evaluator via 9Router local-claude
- [x] 3.5 Latency terukur (total, retrieval, LLM) + provider per query dicatat di naive.json
- [x] 3.6 Catat baseline di `results/benchmark.md`

**Milestone F3:** punya angka baseline objektif. Bisa bilang "Naive RAG: faithfulness X, context precision Y, latency Z ms".

### FASE 4 — Improvement & Comparison (minggu 3-4) ← yang bikin MENONJOL
- [x] 4.1 Strategi **+ Reranking**: retrieve top-20 → LLM-as-reranker (RankGPT-style) → top-5. Context Precision: 0.751 → 0.825 (+0.074). Latency: +10,003ms.
- [x] 4.2 Strategi **+ HyDE**: LLM bikin jawaban hipotetis → embed itu → retrieve. Context Recall: 0.897 → 0.793 (-0.104). HyDE kurang efektif untuk pertanyaan faktual pendek.
- [x] 4.3 Strategi **+ Hybrid** (semantic + BM25 via RRF): gabung skor. Faithfulness: +0.009, Context Recall sama (0.897), Latency: -4,817ms (lebih cepat dari naive!).
- [x] 4.4 Isi **tabel perbandingan** lengkap di `results/benchmark.md` — 4 strategi, delta table, latency breakdown.
- [x] 4.5 Tulis analisis singkat: Reranking menang context precision (+7.4 pts), Hybrid menang latency (-4.8s), HyDE tidak efektif untuk pertanyaan faktual.
- [x] 4.6 Streamlit demo (`app.py`): input pertanyaan → jawaban + sumber + pilih strategi (naive/rerank/hyde/hybrid)
- [x] 4.7 README profesional: arsitektur (ASCII diagram), hasil benchmark, 4 strategi, cara jalankan, keterbatasan — tanpa emoji
- [ ] 4.8 Deploy: Docker + Fly.io (atau demo video kalau Qdrant berat di free tier)
- [ ] 4.9 Push ke GitHub (repo publik, README tanpa emoji sesuai preferensi)

**Milestone F4:** repo publik + tabel benchmark + demo. Project siap dilirik HRD.

---

## 4. Contoh Tabel Benchmark (target di results/benchmark.md)

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Latency (ms) |
|---|---|---|---|---|---|
| Naive RAG (baseline) | _ | _ | _ | _ | _ |
| + Reranking | _ | _ | _ | _ | _ |
| + HyDE | _ | _ | _ | _ | _ |
| + Hybrid (semantic+BM25) | _ | _ | _ | _ | _ |

> Isi dengan angka riil hasil eval. INI aset utama project.

---

## 5. Bullet CV (isi angka setelah Fase 4)

```
RAG Knowledge Assistant — Retrieval Evaluation Study
Python • FastAPI • Qdrant • LangChain • sentence-transformers • RAGAS • Groq/Gemini • Docker
● Built a RAG system for Indonesian UMKM regulation documents with a FastAPI backend and
  Qdrant vector store, indexing N document chunks (multilingual-e5 embeddings, CPU-only).
● Benchmarked 4 retrieval strategies (naive, reranking, HyDE, hybrid) on a 50-query golden
  dataset using RAGAS — reranking improved context precision from X% to Y% (+Z pts) at +N ms.
● Deployed via Docker + Fly.io with a reproducible evaluation harness and Streamlit demo.
```

---

## 6. Pitfalls / Catatan Jujur

- **Golden dataset adalah kunci** — jangan malas verifikasi manual. Dataset jelek → angka tidak kredibel.
- **BGE-reranker di CPU agak lambat** — OK untuk eval batch, tapi catat latensinya apa adanya (justru itu data trade-off yang menarik).
- **Groq rate limit** — pakai fallback ke Gemini + retry backoff (Hamdan sudah paham pola ini dari agent sebelumnya).
- **Jangan over-claim** — tulis metrik apa adanya. Kalau HyDE tidak menang, laporkan jujur; analisis "kenapa tidak menang" justru menunjukkan kematangan.
- **README tanpa emoji** (preferensi Hamdan, berlaku semua repo).
- **.env jangan ke-commit** — cek `.gitignore` sebelum push pertama.

## 7. Nilai Ganda
- Sekalian belajar **FastAPI** (diminta WizAI, PWS, banyak lowongan AI).
- Naik dari "search-augmented generation" → **RAG klasik + evaluation** (lompatan kredibilitas CV).
- Bisa jadi bahan **AI Hackfest 2026** kalau domain diarahkan ke Business Automation / Public Good.
