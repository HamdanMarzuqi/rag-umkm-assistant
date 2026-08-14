"""Bangun draft golden dataset untuk evaluasi RAG.

Untuk sejumlah chunk sampel, minta LLM membuat (pertanyaan, jawaban ideal)
yang HANYA bisa dijawab dari chunk itu. Output = draft JSON yang WAJIB
diverifikasi manual (buang entri jelek) sebelum dipakai eval.

Jalankan:
    .venv/Scripts/python.exe -m eval.build_golden --n 40
Output: eval/golden_draft.json  (lalu review -> simpan sbagai golden_dataset.json)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, llm  # noqa: E402

GEN_SYSTEM = (
    "Anda pembuat soal evaluasi. Dari KUTIPAN regulasi UMKM, buat SATU pasangan "
    "tanya-jawab yang jawabannya ADA di kutipan. Pertanyaan natural (seperti "
    "pelaku UMKM bertanya), jawaban ringkas & faktual. Balas HANYA JSON: "
    '{"question": "...", "answer": "..."}. Jika kutipan tidak layak dijadikan '
    'soal (mis. daftar isi, tanda tangan), balas {"skip": true}.'
)

OUT = Path(__file__).resolve().parent / "golden_draft.json"


def _load_chunks_from_qdrant(n):
    client, mode = config.get_qdrant_client()
    total = client.count(config.COLLECTION).count
    print(f"   Qdrant mode={mode} total_chunk={total}")
    # ambil semua payload lalu sampel; scroll batче
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            config.COLLECTION, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    # hanya chunk cukup panjang (lebih layak jadi soal)
    good = [p.payload for p in points if len(p.payload.get("text", "")) > 300]
    random.seed(42)
    random.shuffle(good)
    return good[:n]


def build(n):
    print("1) Ambil sampel chunk ...")
    chunks = _load_chunks_from_qdrant(n)
    print(f"   {len(chunks)} chunk terpilih (>300 char)")

    dataset = []
    for i, ch in enumerate(chunks, 1):
        prompt = f"KUTIPAN (Sumber: {ch['source']}, hal.{ch['page']}):\n{ch['text']}"
        try:
            raw, _ = llm.generate(prompt, system=GEN_SYSTEM)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            obj = json.loads(raw)
        except Exception as e:
            print(f"   [{i}] gagal parse: {e}")
            continue
        if obj.get("skip") or not obj.get("question"):
            continue
        dataset.append({
            "question": obj["question"],
            "ground_truth": obj["answer"],
            "source": ch["source"],
            "page": ch["page"],
            "verified": False,  # WAJIB set true manual setelah dicek
        })
        print(f"   [{i}] OK: {obj['question'][:70]}")

    OUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSELESAI. {len(dataset)} draft -> {OUT}")
    print("LANGKAH BERIKUT: review manual, set \"verified\": true untuk entri bagus,")
    print("buang yang jelek, simpan sebagai eval/golden_dataset.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="jumlah chunk sampel")
    build(ap.parse_args().n)
