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
    api_key = config.get_secret("OPENCODE_API_KEY", config.OPENCODE_API_KEY)
    base_url = config.get_secret("OPENCODE_BASE_URL", config.OPENCODE_BASE_URL)
    model = config.get_secret("OPENCODE_MODEL", config.OPENCODE_MODEL)
    client = OpenAI(base_url=base_url, api_key=api_key)
    last = None
    for i in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
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


def _groq(prompt, system, max_retries=2):
    from groq import Groq
    api_key = config.get_secret("GROQ_API_KEY", config.GROQ_API_KEY)
    primary_model = config.get_secret("GROQ_MODEL", config.GROQ_MODEL)
    models_to_try = [primary_model]
    if "8b" not in primary_model.lower():
        models_to_try.append("llama-3.1-8b-instant")
    client = Groq(api_key=api_key)
    last = None
    for m in models_to_try:
        for i in range(max_retries):
            try:
                r = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                return r.choices[0].message.content, f"groq"
            except Exception as e:
                last = e
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    break
                time.sleep(1)
    raise last


def _gemini(prompt, system):
    import google.generativeai as genai
    api_key = config.get_secret("GEMINI_API_KEY", config.GEMINI_API_KEY)
    model_name = config.get_secret("GEMINI_MODEL", config.GEMINI_MODEL)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, system_instruction=system)
    r = model.generate_content(prompt)
    return r.text, "gemini"


def _ollama(prompt, system, max_retries=2):
    """Ollama API / Cloud endpoint via OpenAI SDK."""
    from openai import OpenAI
    api_key = config.get_secret("OLLAMA_API_KEY", getattr(config, "OLLAMA_API_KEY", ""))
    base_url = config.get_secret("OLLAMA_BASE_URL", getattr(config, "OLLAMA_BASE_URL", "https://api.ollama.com/v1"))
    model = config.get_secret("OLLAMA_MODEL", getattr(config, "OLLAMA_MODEL", "llama3.2"))
    client = OpenAI(base_url=base_url, api_key=api_key)
    last = None
    for i in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return r.choices[0].message.content, "ollama"
        except Exception as e:
            last = e
            time.sleep(2 ** i)
    raise last


def _router_local(prompt, system):
    """9Router OpenAI-compatible endpoint."""
    from openai import OpenAI
    api_key = config.get_secret("EVAL_API_KEY", config.EVAL_API_KEY)
    base_url = config.get_secret("EVAL_BASE_URL", config.EVAL_BASE_URL)
    model = config.get_secret("EVAL_MODEL", config.EVAL_MODEL)
    client = OpenAI(base_url=base_url, api_key=api_key)
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return r.choices[0].message.content, "router_local"


def _has_key(provider_name):
    """Runtime check: apakah API key tersedia? Baca dinamis dari env & secrets."""
    return {
        "opencode_zen": bool(config.get_secret("OPENCODE_API_KEY", config.OPENCODE_API_KEY)),
        "groq": bool(config.get_secret("GROQ_API_KEY", config.GROQ_API_KEY)),
        "gemini": bool(config.get_secret("GEMINI_API_KEY", config.GEMINI_API_KEY)),
        "ollama": bool(config.get_secret("OLLAMA_API_KEY", getattr(config, "OLLAMA_API_KEY", ""))),
        "router_local": bool(config.get_secret("EVAL_API_KEY", config.EVAL_API_KEY)),
    }.get(provider_name, False)


def generate(prompt, system="You are a helpful assistant."):
    """Return (text, provider). Cek key runtime (per-call, bukan module-load)
    agar Secrets di Streamlit Cloud yang berubah saat app jalan langsung kena.
    Return error string + 'error' provider jika SEMUA gagal."""
    errors = []
    chain = [
        ("opencode_zen", _opencode_zen),
        ("groq", _groq),
        ("gemini", _gemini),
        ("ollama", _ollama),
        ("router_local", _router_local),
    ]
    for name, fn in chain:
        if not _has_key(name):
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
