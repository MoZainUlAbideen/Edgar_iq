# Deploying the live EdgarIQ API (Render) + frontend (Vercel)

Architecture: browser (Vercel static site) -> FastAPI on Render -> Groq (LLM) + Gemini API (query embeddings),
searching a gzip index committed in `deploy/indexes/`.

Why Gemini embeddings instead of Ollama: Render can't run Ollama. Vectors from Ollama and Gemini are **not
interchangeable**, so the index must be rebuilt with Gemini. The index records which embedder built it, and the
server refuses to start on a mismatch rather than silently retrieving garbage. Your original
`data/indexes/NVDA.json` (Ollama) is never touched.

All commands are PowerShell, run from the repo root (`D:\VS-Code-Projects\Edgar_iq`).

## 1. Install dependencies and run tests
```powershell
uv add fastapi uvicorn
uv add --dev httpx
uv run pytest -v
```

## 2. Get a free Gemini key
https://aistudio.google.com/apikey (works with a personal Google account). Add to `.env`:
```
GEMINI_API_KEY=your_key_here
```

## 3. Build the Gemini index and check quality
```powershell
uv run python scripts/build_index.py NVDA --provider gemini --out data/indexes/NVDA_gemini.json
uv run python scripts/run_eval.py NVDA --index data/indexes/NVDA_gemini.json
```
Only ship it if the golden set still passes (the Ollama index scored 7/7).

The first line printed must say `Embedding with gemini:gemini-embedding-001:768`. A rate-limited run can be
re-run as-is: finished embeddings are cached in `data/embedding_cache/`. The eval script picks the matching
embedder from the index itself.

## 4. Package the index for deployment
```powershell
uv run python scripts/package_index.py data/indexes/NVDA_gemini.json
```

## 5. Try the API locally
```powershell
$env:EMBEDDING_PROVIDER="gemini"
uv run uvicorn edgariq.api.main:app --port 8001
```
In a second terminal:
```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
$job = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/ask -ContentType "application/json" -Body '{"question":"What was data center revenue in the 10-Q filed 2026-05-20?"}'
Start-Sleep 30
Invoke-RestMethod "http://127.0.0.1:8001/ask/$($job.job_id)"
```

## 6. Push the backend repo
```powershell
git status
git add src scripts tests deploy render.yaml .env.example DEPLOY.md README.md pyproject.toml uv.lock
git commit -m "Add live FastAPI backend, hosted Gemini embeddings and Render config"
git push
```

## 7. Render
Dashboard -> New -> Blueprint -> select this repo (uses `render.yaml`). Set the secret env vars when prompted:
`GROQ_API_KEY`, `GEMINI_API_KEY`, `ALLOWED_ORIGINS` (your Vercel URL, no trailing slash).
Check `https://<service>.onrender.com/health`.

## 8. Vercel frontend
Edit `web/config.js`: `window.EDGARIQ_API_URL = "https://<service>.onrender.com";` then:
```powershell
cd web
git add .
git commit -m "Wire chat widget to live backend"
git push
cd ..
```
Vercel redeploys on push; the chat input switches on when `/health` answers.

## Gemini free-tier quota (learned the hard way)
The free tier allows about **1,000 embedding requests per day**, and every text counts as one request.
Building the NVDA index (1,144 chunks) uses a full day's quota, which is why `build_index.py` caches finished
embeddings and resumes. At serve time each question costs about 3-4 query embeddings (identical queries are
cached in memory), so the default `API_DAILY_CAP=150` questions/day stays under the limit. When the quota is
gone, the API tells visitors so, and the widget's recorded answers keep working. The quota resets daily.

## Behaviour to know about
- Free Render instances sleep after ~15 min idle; the first request takes ~30-60 s. The widget shows a
  "connecting" message and falls back to recorded answers if the backend is unreachable.
- Abuse protection: 6 questions/hour per IP and 150/day overall (env-configurable), one worker with a queue of 4,
  500-char questions. Limits live in memory and reset on restart.
- Scope is NVDA only, as indexed. Answers keep the grounding warnings (unverified numbers, critic notes).
