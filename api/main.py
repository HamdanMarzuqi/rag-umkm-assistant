"""FastAPI: endpoint /query untuk RAG UMKM.

Jalankan:
    cd rag-umkm-assistant
    .venv/Scripts/python.exe -m uvicorn api.main:app --port 8000
Dokumen interaktif: http://localhost:8000/docs

ponytail: tanpa auth (demo lokal/portfolio). Tambah API key header bila deploy publik.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src import rag, config  # noqa: E402

app = FastAPI(title="RAG UMKM Assistant", version="0.1.0")


class Query(BaseModel):
    question: str
    strategy: str = "naive"
    top_k: int = config.TOP_K


@app.get("/health")
def health():
    return {"status": "ok", "collection": config.COLLECTION}


@app.post("/query")
def query(q: Query):
    return rag.answer(q.question, strategy=q.strategy, top_k=q.top_k)
