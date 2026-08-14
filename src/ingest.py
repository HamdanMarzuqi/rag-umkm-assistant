"""Ingestion: PDF -> teks (pymupdf) -> chunk -> embed (e5) -> Qdrant.

Jalankan:
    cd rag-umkm-assistant
    .venv/Scripts/python.exe -m src.ingest

Idempoten: recreate collection tiap run. Chunk kosong (PDF hasil scan tanpa
teks) dilewati & dilaporkan supaya bisa di-OCR terpisah bila perlu.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402  (setelah path guard di config)

import fitz  # pymupdf
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


def extract_pages(pdf_path):
    """Return list teks per halaman (string)."""
    doc = fitz.open(pdf_path)
    pages = [p.get_text("text") for p in doc]
    doc.close()
    return pages


def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Chunk berbasis kata dengan overlap. size/overlap dihitung dalam kata
    (aproksimasi token; cukup untuk regulasi berbahasa Indonesia)."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(words):
        chunk = " ".join(words[start:start + size]).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def build_chunks():
    """Baca semua PDF -> list dict {id, text, payload}. Lapor PDF tanpa teks."""
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Tidak ada PDF di {config.PDF_DIR}")

    records, empty_pdfs, pid = [], [], 0
    for pdf in pdfs:
        pages = extract_pages(pdf)
        doc_text_len = sum(len(p.strip()) for p in pages)
        if doc_text_len < 50:                      # nyaris tanpa teks -> scan
            empty_pdfs.append(pdf.name)
            continue
        for pno, ptext in enumerate(pages, 1):
            for cno, chunk in enumerate(chunk_text(ptext)):
                records.append({
                    "id": pid,
                    "text": chunk,
                    "payload": {
                        "source": pdf.name,
                        "page": pno,
                        "chunk": cno,
                        "text": chunk,
                    },
                })
                pid += 1
    return records, empty_pdfs, len(pdfs)


def main():
    print("1) Membaca & chunking PDF ...")
    records, empty_pdfs, n_pdf = build_chunks()
    print(f"   PDF: {n_pdf} | chunk: {len(records)} | tanpa-teks: {len(empty_pdfs)}")
    if empty_pdfs:
        print("   PDF tanpa teks (perlu OCR, dilewati):")
        for n in empty_pdfs:
            print("     -", n)

    print(f"2) Load embedding model: {config.EMBED_MODEL} (CPU) ...")
    model = SentenceTransformer(config.EMBED_MODEL, device="cpu")

    # e5 minta prefix "passage: " untuk dokumen
    texts = ["passage: " + r["text"] for r in records]
    print(f"3) Meng-embed {len(texts)} chunk ...")
    vectors = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )

    print("4) Menyiapkan Qdrant ...")
    client, mode = config.get_qdrant_client()
    print(f"   mode: {mode}")
    client.recreate_collection(
        config.COLLECTION,
        vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
    )

    print("5) Upsert ke Qdrant ...")
    points = [
        PointStruct(id=r["id"], vector=vectors[i].tolist(), payload=r["payload"])
        for i, r in enumerate(records)
    ]
    for i in range(0, len(points), 128):
        client.upsert(config.COLLECTION, points[i:i + 128])

    cnt = client.count(config.COLLECTION).count
    print(f"SELESAI. {cnt} vektor tersimpan di collection '{config.COLLECTION}'.")


if __name__ == "__main__":
    main()
