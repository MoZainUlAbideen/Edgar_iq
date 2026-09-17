# EdgarIQ

A grounded research assistant over SEC filings (10-K / 10-Q / 8-K). Ask a
question like "how did NVIDIA's data center revenue grow YoY, and what risk
factors did they flag around export controls?" and get an answer with every
number and claim traced back to the exact filing it came from — verified by
a calculator agent, not eyeballed by an LLM.

## Why this exists

Standard RAG-over-PDF breaks on financial filings: tables lose their
structure when flattened to text, charts are invisible to text-only
pipelines, and a hallucinated number here isn't a quirky mistake — it's the
kind of error that leads to a bad decision. EdgarIQ is built around
grounding and verification as first-class concerns, not an afterthought.

## Architecture

```
SEC EDGAR filings -> Doc parsing (OCR, tables, charts) -> Hybrid RAG index
                                                                 |
                                                                 v
                                            Agent orchestrator (plan, retrieve, verify)
                                                                 |
                                                                 v
                                              Grounded answer, cited to source page
```

- **Backend**: Ollama (local embeddings) + Groq API (reasoning/generation)
- **Ingestion**: direct SEC EDGAR REST API, no scraping
- **v1 scope**: text + table RAG, multi-agent orchestration, eval harness.
  Vision-based page retrieval (for charts/tables OCR mangles) is a planned v2.

## Setup

```bash
uv sync
cp .env.example .env
# then edit .env: set SEC_USER_AGENT to "Your Name your@email.com" (SEC requires this)
```

## Running the smoke test

This hits the real SEC API, so it needs to run somewhere with actual
internet access (not a sandboxed container):

```bash
uv run python scripts/smoke_test.py
```

## Running tests

```bash
uv run pytest -v
```

These are all mocked (no real network calls), so they run anywhere,
including CI.

## Progress log

- [x] Project scaffold (uv, package layout, git)
- [x] SEC EDGAR ingestion client — ticker resolution, filing history, document download
- [x] Unit tests for ingestion (mocked HTTP)
- [ ] Document parsing (tables, chart captions)
- [ ] Hybrid RAG index (Ollama embeddings)
- [ ] Multi-agent orchestrator (planner, retriever, calculator, critic)
- [ ] Eval harness (golden set, grounding checks, regression tracking)
- [ ] Frontend
