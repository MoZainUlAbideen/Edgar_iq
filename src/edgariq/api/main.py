"""Production entrypoint:  uvicorn edgariq.api.main:app"""

from edgariq.api.factory import build_app

app = build_app()
