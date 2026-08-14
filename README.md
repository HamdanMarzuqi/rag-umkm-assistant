# RAG Knowledge Assistant -- Retrieval Evaluation Study

A Retrieval-Augmented Generation (RAG) system for Indonesian UMKM (micro, small, and medium enterprise) regulation documents, with a rigorous evaluation of four retrieval strategies on a manually verified golden dataset.

Built as a portfolio project demonstrating end-to-end RAG implementation, quantitative evaluation with RAGAS, and retrieval strategy trade-off analysis -- all running on CPU without GPU.

---

## Table of Contents

- [Architecture](#architecture)
- [Benchmark Results](#benchmark-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Retrieval Strategies](#retrieval-strategies)
- [Limitations](#limitations)
- [License](#license)

---

## Architecture

```
PDF Documents (32 regulasi UMKM)
        |
        v
  [ingest.py] PDF -> text (pymupdf) -> chunk (512 words, overlap 64) -> embed (e5-small) -> Qdrant (771 vectors)
        |
        v
  [Qdrant Vector Store]  <---  embedded on-disk (no Docker required)
        |
        v
  [retriever.py]  User question -> embed -> retrieve top-k chunks
        |            |-- naive:   cosine similarity
        |            |-- rerank:  LLM-as-reranker (RankGPT-style, top-20 -> top-5)
        |            |-- hyde:    hypothetical document embedding
        |            |-- hybrid:  BM25 + semantic via Reciprocal Rank Fusion
        v
  [rag.py]  Retrieved chunks -> prompt augmentation -> LLM generates grounded answer
        |
        v
  [llm.py]  4-tier fallback: Opencode (Zen/DeepSeek) -> Groq (Llama 3.3 70B) -> Gemini -> local LLM
        |
        +-- [api/main.py]  FastAPI REST API (/query, /health)
        +-- [app.py]       Streamlit interactive demo
```

---

## Benchmark Results

Evaluated on 29 manually verified questions from the golden dataset using RAGAS metrics.

### Main Results

| Strategy | Faithfulness | Answer Rel | Context Prec | Context Rec | Avg Latency |
|---|---|---|---|---|---|
| Naive RAG (baseline) | 0.708 | 0.932 | 0.751 | 0.897 | 23,032 ms |
| + Reranking (LLM-as-reranker) | 0.668 | 0.893 | **0.825** | 0.897 | 33,035 ms |
| + HyDE | 0.630 | 0.886 | 0.667 | 0.793 | 32,908 ms |
| + Hybrid (BM25 + semantic/RRF) | **0.717** | 0.848 | 0.753 | **0.897** | **18,215 ms** |

### Delta vs Baseline

| Strategy | Context Prec | Context Rec | Faithfulness | Latency |
|---|---|---|---|---|
| + Reranking | **+0.074** | 0.000 | -0.040 | +10,003 ms |
| + HyDE | -0.084 | -0.104 | -0.078 | +9,876 ms |
| + Hybrid | +0.002 | 0.000 | **+0.009** | **-4,817 ms** |

### Key Findings

- **Reranking** achieves the best context precision (+7.4 pts), meaning the chunks fed to the LLM are more relevant. Trade-off: +10s latency per query.
- **Hybrid** is the fastest strategy overall, beating even naive by 4.8 seconds, while maintaining equal context recall. BM25 in-memory index eliminates the need for an extra LLM call.
- **HyDE** is not effective for this dataset -- factual, short-answer questions (who, how much, when) do not benefit from hypothetical document embeddings. HyDE is better suited for conceptual/abstract queries.

Full analysis: [results/benchmark.md](results/benchmark.md)

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Embedding | intfloat/multilingual-e5-small | 384-dim, CPU-friendly, good for Bahasa Indonesia |
| Vector DB | Qdrant (embedded on-disk) | Production-grade, no Docker required for dev |
| LLM | Opencode (Zen/DeepSeek) + Groq + Gemini | 4-tier fallback chain, no local GPU needed |
| Reranker | LLM-as-reranker (RankGPT-style) | Zero model download, uses existing LLM stack |
| Keyword Search | rank-bm25 (BM25Okapi) | For hybrid retrieval |
| PDF Extraction | pymupdf | Fast, reliable text extraction |
| API | FastAPI | REST endpoint with strategy selection |
| Demo UI | Streamlit | Interactive query interface |
| Evaluation | RAGAS | 4 metrics: faithfulness, answer relevancy, context precision, context recall |
| Language | Python 3.11+ | All CPU, no GPU dependencies |

---

## Project Structure

```
rag-umkm-assistant/
├── src/
│   ├── config.py           # centralized configuration, env vars, Qdrant client factory
│   ├── ingest.py           # PDF -> chunk -> embed -> Qdrant
│   ├── retriever.py        # 4 strategies: naive, rerank, hyde, hybrid
│   ├── rag.py              # retrieve -> prompt augmentation -> LLM answer
│   └── llm.py              # 4-tier fallback: Opencode -> Groq -> Gemini -> local LLM
├── eval/
│   ├── golden_dataset.json # 29 verified (question, ground_truth, source, page)
│   ├── build_golden.py     # generate draft golden set via LLM
│   └── run_eval.py         # RAGAS evaluation + benchmark table
├── api/
│   └── main.py             # FastAPI /query and /health endpoints
├── app.py                  # Streamlit interactive demo
├── results/
│   ├── benchmark.md        # full benchmark table with analysis
│   ├── naive.json          # per-query results for each strategy
│   ├── rerank.json
│   ├── hyde.json
│   └── hybrid.json
├── data/
│   ├── raw/                # source PDF documents
│   └── processed/          # (optional) intermediate chunks
├── qdrant_storage/         # embedded Qdrant on-disk data
├── requirements.txt
├── .env.example
├── .gitignore
└── docker-compose.yml
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- API keys: [Groq](https://console.groq.com/keys) and/or [Gemini](https://aistudio.google.com/app/apikey) (free tier)
- PDF documents in a local directory (32 UMKM regulation PDFs used in this study)

### Installation

```bash
git clone https://github.com/<your-username>/rag-umkm-assistant.git
cd rag-umkm-assistant

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   GROQ_API_KEY=your_key
#   GEMINI_API_KEY=your_key
```

### Ingest Documents

```bash
python -m src.ingest
```

This extracts text from PDFs, chunks them (512 words, 64 overlap), embeds with multilingual-e5-small, and stores 771 vectors in Qdrant (embedded on-disk, no Docker needed).

---

## Usage

### Streamlit Demo

```bash
python -m streamlit run app.py
```

Opens at http://localhost:8501. Select a retrieval strategy, type a question about UMKM regulations, and see the grounded answer with source citations.

### FastAPI

```bash
python -m uvicorn api.main:app --port 8000
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Apa saja kriteria usaha mikro?", "strategy": "rerank", "top_k": 5}'
```

### CLI (Quick Test)

```bash
python -m src.rag "Apa saja kriteria usaha mikro?"
```

---

## Evaluation

### Golden Dataset

29 question-answer pairs manually extracted and verified from the source regulation documents. Each entry includes:
- `question`: factual question in Bahasa Indonesia
- `ground_truth`: exact answer from the document
- `source`: PDF filename
- `page`: page number
- `verified`: manual verification flag

### Run Evaluation

```bash
# Full evaluation (29 questions)
python -m eval.run_eval --strategy naive
python -m eval.run_eval --strategy rerank
python -m eval.run_eval --strategy hyde
python -m eval.run_eval --strategy hybrid

# Smoke test (3 questions, quick verification)
python -m eval.run_eval --strategy rerank --limit 3
```

Results are saved to `results/<strategy>.json` and appended to `results/benchmark.md`.

### RAGAS Metrics

| Metric | What It Measures |
|---|---|
| Faithfulness | Is the answer grounded in the retrieved contexts? (no hallucination) |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are the retrieved chunks relevant to the question? |
| Context Recall | Do the retrieved chunks cover the ground truth? |

---

## Retrieval Strategies

### 1. Naive (Baseline)

Embed the question with e5-small, query Qdrant for top-k by cosine similarity.

### 2. Reranking (LLM-as-Reranker)

Retrieve top-20 from Qdrant, then ask the LLM to rank them by relevance (RankGPT-style). Return top-5. This is a valid alternative to cross-encoder rerankers (BGE, etc.) and requires no additional model download.

**Best at:** Context Precision (+7.4 pts vs baseline)

### 3. HyDE (Hypothetical Document Embeddings)

LLM generates a hypothetical answer, which is embedded and used as the query vector. The intuition is that a hypothetical answer is semantically closer to relevant passages than the original question.

**Finding:** Not effective for short factual questions in this regulatory domain.

### 4. Hybrid (BM25 + Semantic via RRF)

Two retrieval paths run in parallel: BM25 keyword search over all 771 chunks and semantic search via Qdrant. Results are fused using Reciprocal Rank Fusion (RRF, k=60).

**Best at:** Latency (fastest overall, -4.8s vs baseline) with equal context recall.

---

## Limitations

- **Golden dataset size**: 29 questions is sufficient for directional insights but not statistically robust. A larger dataset (50-100+) would strengthen conclusions.
- **CPU-only constraint**: All models run on CPU. Latency numbers reflect this and would be significantly lower with GPU acceleration.
- **LLM provider dependency**: Most queries fell back to local LLM (router_local) due to Groq rate limits. Production deployment should account for rate limiting and provider costs.
- **PDF quality**: 4 scanned PDFs (no extractable text) were excluded from indexing. OCR integration would increase document coverage.
- **Single domain**: Results are specific to Indonesian UMKM regulation documents. Strategy effectiveness may differ for other domains or languages.
- **Evaluation LLM**: RAGAS metrics were computed using a local LLM (not GPT-4), which may produce slightly different scores than the RAGAS default evaluator.

---

## Author

**Hamdan Akbar Marzuqi**

Built from personal experience as owner of Malika Kebab -- navigating UMKM regulations (permits, taxes, compliance) is genuinely difficult, and this system aims to make that knowledge more accessible.

---

## License

This project is for portfolio and educational purposes.
