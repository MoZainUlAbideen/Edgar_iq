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

<img width="1329" height="632" alt="image" src="https://github.com/user-attachments/assets/46176026-c518-45e3-95e2-8c9ae0595334" />


## Progress log

- [x] Project scaffold (uv, package layout, git)
- [x] SEC EDGAR ingestion client — ticker resolution, filing history, document download
- [x] Unit tests for ingestion (mocked HTTP)
- [x] Document parsing — clean text + data-table filtering, validated and bug-fixed against a real live filing (14 tests passing)
- [x] Chunking + Ollama embeddings + local vector search (30 tests passing)
- [x] Multi-filing indexing pipeline with persistence — validated end-to-end on 8 real NVDA filings (1,144 chunks)
- [x] Multi-agent orchestrator — planner, retriever, drafter, numeric grounding checker, critic (58 tests passing, 4 real bugs found and fixed via testing + one live end-to-end run)
- [x] Eval harness — golden set (7 cases, real verified facts), deterministic + LLM-judge grading, regression tracking, HTML report (89 tests passing)
- [x] Diagnosed and fixed real retrieval failures found by the eval harness: date-aware metadata filtering + retrieval-depth tuning took the golden set from 29% → 86% → 100% pass rate
- [ ] Frontend
