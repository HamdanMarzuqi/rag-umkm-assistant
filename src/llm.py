"""Wrapper LLM dengan 4-tier fallback: Opencode(Zen) -> Groq -> Gemini -> 9Router(local-claude).

Urutan:
1. Opencode (Zen / DeepSeek-V4-Flash)
2. Groq (Llama 3.3 70B)
3. Gemini (Gemini 2.0 Flash)
4. 9Router (local-claude) sebagai jaring pengaman terakhir
"""
import time

from src import config


def _opencode_zen(prompt, system, max_retries=2):
    """Opencode Zen OpenAI-compatible endpoint."""
    from openai import OpenAI
    client = OpenAI(base_url=config.OPENCODE_BASE_URL, api_key=config.OPENCODE_API_KEY)
    last = None
    for i in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=config.OPENCODE_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return r.choices[0].message.content, "opencode_zen"
        except Exception as e:
            last = e
            time.sleep(2 ** i)
    raise last


def _groq(prompt, system, max_retries=3):
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    last = None
    for i in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return r.choices[0].message.content, "groq"
        except Exception as e:
            last = e
            time.sleep(2 ** i)
    raise last


def _gemini(prompt, system):
    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL, system_instruction=system)
    r = model.generate_content(prompt)
    return r.text, "gemini"


def _router_local(prompt, system):
    """9Router OpenAI-compatible endpoint."""
    from openai import OpenAI
    client = OpenAI(base_url=config.EVAL_BASE_URL, api_key=config.EVAL_API_KEY)
    r = client.chat.completions.create(
        model=config.EVAL_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return r.choices[0].message.content, "router_local"


_CHAIN = [
    ("opencode_zen", lambda p, s: _opencode_zen(p, s), bool(config.OPENCODE_API_KEY)),
    ("groq", lambda p, s: _groq(p, s), bool(config.GROQ_API_KEY)),
    ("gemini", lambda p, s: _gemini(p, s), bool(config.GEMINI_API_KEY)),
    ("router_local", lambda p, s: _router_local(p, s), bool(config.EVAL_API_KEY)),
]


def generate(prompt, system="You are a helpful assistant."):
    """Return (text, provider). Coba tiap provider sesuai urutan; jika gagal
    lanjut ke berikutnya. Return error string + 'error' provider jika SEMUA
    gagal, supaya caller (eval/RAG) tidak crash."""
    errors = []
    for name, fn, available in _CHAIN:
        if not available:
            errors.append(f"{name}: no api key")
            continue
        try:
            return fn(prompt, system)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:100]}")
            continue
    return (
        "ERROR: semua provider gagal. " + " | ".join(errors),
        "error",
    )


if __name__ == "__main__":
    txt, prov = generate("Sebut 1 kalimat: apa itu UMKM?")
    print(f"[{prov}] {txt}")
