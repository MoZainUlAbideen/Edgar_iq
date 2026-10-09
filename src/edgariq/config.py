"""Central place for environment-driven config.

Everything here is loaded from a local .env file (never committed — see
.gitignore) via python-dotenv. Copy .env.example to .env and fill in your
own values before running anything.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # SEC requires a real name + email in the User-Agent header.
    SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "")

    # Groq API key for the reasoning/generation LLM calls (later milestones).
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_MIN_REQUEST_INTERVAL_SECONDS: float = float(
        os.getenv("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "2.5")
    )

    # Which embedder to use: "ollama" (local dev) or "gemini" (hosted -- needed
    # for the deployed API, which can't run Ollama).
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # --- Live API (src/edgariq/api) ---
    # Comma-separated browser origins allowed to call the API (your Vercel site).
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500")
    API_TICKER: str = os.getenv("API_TICKER", "NVDA")
    API_INDEX_PATH: str = os.getenv("API_INDEX_PATH", "deploy/indexes/NVDA.json.gz")
    API_RATE_LIMIT_PER_HOUR: int = int(os.getenv("API_RATE_LIMIT_PER_HOUR", "6"))
    API_DAILY_CAP: int = int(os.getenv("API_DAILY_CAP", "150"))
    API_MAX_QUEUE: int = int(os.getenv("API_MAX_QUEUE", "4"))

    # Ollama runs locally; default assumes the standard local install.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


settings = Settings()
