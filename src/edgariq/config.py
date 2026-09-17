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

    # Ollama runs locally; default assumes the standard local install.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


settings = Settings()
