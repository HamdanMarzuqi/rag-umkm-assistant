"""Konfigurasi terpusat: env, path, konstanta, factory Qdrant client.

Catatan PYTHONPATH: di environment ini shell menyuntik PYTHONPATH ke
site-packages Hermes, yang bisa menimpa paket venv. Guard di bawah membuang
path Hermes dari sys.path saat runtime supaya paket venv yang dipakai.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Guard PYTHONPATH: buang path Hermes agar venv project yang menang ──
sys.path[:] = [p for p in sys.path if "hermes-agent" not in p.replace("\\", "/")]

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

def get_secret(key: str, default: str = "") -> str:
    """Ambil value dari os.environ atau st.secrets secara dinamis."""
    val = os.environ.get(key)
    if val:
        return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default

# ── Sumber & storage ──
PDF_DIR = Path(os.getenv("PDF_DIR", r"E:\Dokumenku_2025\Regulasi_Panduan UMKM_For_RAG"))
QDRANT_PATH = ROOT / "qdrant_storage"          # embedded on-disk store
COLLECTION = os.getenv("COLLECTION_NAME", "umkm_regulasi")

# ── Model (CPU-friendly) ──
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = 384                                 # multilingual-e5-small
OPENCODE_MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4-flash-free")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Chunking ──
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ── Retrieval ──
TOP_K = 5

# ── API keys & endpoints ──
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# ── Evaluator (RAGAS) via 9Router OpenAI-compatible ──
EVAL_BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:20128/v1")
EVAL_API_KEY = os.getenv("EVAL_API_KEY", "")
EVAL_MODEL = os.getenv("EVAL_MODEL", "local-claude")


def get_qdrant_client():
    """Kembalikan Qdrant client. Pakai server Docker bila hidup, jika tidak
    fallback ke embedded on-disk. Return (client, mode_str)."""
    from qdrant_client import QdrantClient
    try:
        c = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2.0)
        c.get_collections()  # ping
        return c, f"server({QDRANT_HOST}:{QDRANT_PORT})"
    except Exception:
        QDRANT_PATH.mkdir(exist_ok=True)
        return QdrantClient(path=str(QDRANT_PATH)), "embedded"
