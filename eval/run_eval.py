"""Jalankan evaluasi RAGAS pada strategi tertentu.

Usage:
    .venv/Scripts/python.exe -m eval.run_eval --strategy naive [--limit 3]

Mengukur: faithfulness, answer_relevancy, context_precision, context_recall,
avg_latency_ms. Hasil disimpan ke results/<strategy>.json + ringkasan markdown.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, rag  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATASET = Path(__file__).resolve().parent / "golden_dataset.json"


def _make_llm():
    """LLM evaluator via 9Router (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=config.EVAL_BASE_URL,
        api_key=config.EVAL_API_KEY,
        model=config.EVAL_MODEL,
        temperature=0,
    )


def _make_emb():
    """Embedding evaluator = e5 lokal yang sama dengan retriever."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def collect(limit=None, strategy="naive"):
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    if limit:
        items = items[:limit]
    samples = []
    for i, it in enumerate(items, 1):
        t0 = time.perf_counter()
        r = rag.answer(it["question"], strategy=strategy)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        samples.append({
            "user_input": it["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": it["ground_truth"],
            "reference_contexts": [],   # context untuk recall pakai ground_truth saja
            "latency_ms": elapsed,
            "retrieval_latency_ms": r["retrieval_latency_ms"],
            "llm_latency_ms": r["llm_latency_ms"],
            "provider": r["provider"],
        })
        print(f"[{i}/{len(items)}] {elapsed}ms | {it['question'][:55]}")
    return samples


def evaluate(samples, strategy):
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import EvaluationDataset, evaluate as ragas_evaluate
    from ragas.metrics import (
        Faithfulness, AnswerRelevancy,
        ContextPrecision, ContextRecall,
    )

    llm = LangchainLLMWrapper(_make_llm())
    emb = LangchainEmbeddingsWrapper(_make_emb())

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    ds = EvaluationDataset.from_list(samples)
    print(f"\nRunning RAGAS on {len(samples)} samples ({strategy}) ...")
    res = ragas_evaluate(ds, metrics=metrics)
    df = res.to_pandas()
    for field in ("latency_ms", "retrieval_latency_ms", "llm_latency_ms"):
        df[field] = [s[field] for s in samples]
    df["provider"] = [s["provider"] for s in samples]

    RESULTS.mkdir(exist_ok=True)
    df.to_json(RESULTS / f"{strategy}.json", orient="records", indent=2, force_ascii=False)

    summary = {col: float(df[col].mean()) for col in ["faithfulness", "answer_relevancy",
                                                       "context_precision", "context_recall"]}
    summary["avg_latency_ms"] = float(df["latency_ms"].mean())
    summary["avg_retrieval_ms"] = float(df["retrieval_latency_ms"].mean())
    summary["avg_llm_ms"] = float(df["llm_latency_ms"].mean())
    summary["providers"] = df["provider"].value_counts().to_dict()
    summary["n"] = len(samples)
    summary["strategy"] = strategy
    print("\n", summary)
    return summary, df


def render_markdown(rows):
    out = ["# RAG Benchmark", "", "| Strategy | n | Faithfulness | Answer Rel | Context Prec | Context Rec | Latency (ms) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['strategy']} | {r['n']} | {r['faithfulness']:.3f} | "
                   f"{r['answer_relevancy']:.3f} | {r['context_precision']:.3f} | "
                   f"{r['context_recall']:.3f} | {r['avg_latency_ms']:.0f} |")
    (RESULTS / "benchmark.md").write_text("\n".join(out), encoding="utf-8")
    print(f"saved -> {RESULTS / 'benchmark.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="naive")
    ap.add_argument("--limit", type=int, default=None,
                    help="batasi jumlah sample (untuk smoke test)")
    args = ap.parse_args()

    samples = collect(limit=args.limit, strategy=args.strategy)
    summary, _ = evaluate(samples, args.strategy)

    # append to benchmark.md
    md = RESULTS / "benchmark.md"
    if md.exists():
        text = md.read_text(encoding="utf-8")
        if f"| {args.strategy} |" in text:    # replace row
            lines = text.splitlines()
            lines = [l for l in lines if not l.startswith(f"| {args.strategy} |")]
            text = "\n".join(lines) + "\n"
        text += (f"| {summary['strategy']} | {summary['n']} | "
                 f"{summary['faithfulness']:.3f} | {summary['answer_relevancy']:.3f} | "
                 f"{summary['context_precision']:.3f} | {summary['context_recall']:.3f} | "
                 f"{summary['avg_latency_ms']:.0f} |\n")
        md.write_text(text, encoding="utf-8")
    else:
        render_markdown([summary])
