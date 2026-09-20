"""Runs the golden-set eval against a ticker's index, saves a timestamped
JSON result + a viewable HTML report, and checks for regressions against
the most recent previous run for that ticker.

Usage:
    uv run python scripts/run_eval.py NVDA

Requires an index already built (build_index.py) and cases for that
ticker in src/edgariq/evaluation/golden_set.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgariq.config import settings
from edgariq.evaluation import (
    GOLDEN_SET,
    EvalReport,
    compare_reports,
    generate_html_report,
    run_eval,
)
from edgariq.indexing import OllamaEmbedder, VectorStore
from edgariq.llm import GroqClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
RESULTS_DIR = PROJECT_ROOT / "eval_results"


def _atomic_write_text(path: Path, content: str) -> None:
    """Write to a sibling .tmp file, then rename into place. A crash during
    the write leaves the incomplete .tmp file, never a corrupt version of
    the real file — the exact scenario that caused the previous crash to
    leave behind an empty .json that a later run then failed to parse."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    index_path = INDEX_DIR / f"{ticker}.json"
    if not index_path.exists():
        raise SystemExit(
            f"No index found at {index_path}. Build one first with:\n"
            f"  uv run python scripts/build_index.py {ticker}"
        )

    cases = [c for c in GOLDEN_SET if c.ticker == ticker]
    if not cases:
        raise SystemExit(
            f"No golden-set cases defined for {ticker}. Add some to "
            f"src/edgariq/evaluation/golden_set.py."
        )

    store = VectorStore.load(index_path)
    embedder = OllamaEmbedder(
        base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBEDDING_MODEL
    )
    llm = GroqClient(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        min_request_interval=settings.GROQ_MIN_REQUEST_INTERVAL_SECONDS,
    )

    print(f"Running {len(cases)} golden-set cases against {ticker}...\n"
          f"(each case runs the full planner/retriever/drafter/critic pipeline, "
          f"plus a judge call for qualitative cases — this will take a while)")
    report = run_eval(cases, ticker, store, embedder, llm)

    ticker_dir = RESULTS_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Load the most recent PRIOR run (if any) before writing this one, so we
    # have something to compare against. Skip (rather than crash on) a
    # corrupt/empty file — that's exactly what a previous run that crashed
    # mid-write leaves behind, and one bad old file shouldn't block a new run.
    previous_files = sorted(ticker_dir.glob("*.json"))
    previous_report = None
    if previous_files:
        try:
            previous_report = EvalReport.model_validate_json(
                previous_files[-1].read_text(encoding="utf-8")
            )
        except Exception as e:
            print(
                f"Warning: couldn't read previous result file "
                f"{previous_files[-1].name} ({e}); skipping regression comparison "
                f"for this run. (You can delete that file — it's likely a leftover "
                f"from an earlier crashed run.)"
            )

    timestamp = report.run_at.replace(":", "-")
    json_path = ticker_dir / f"{timestamp}.json"
    html_path = ticker_dir / "latest.html"
    # Write to a temp file and rename into place, rather than writing the
    # real path directly — a crash mid-write (like the encoding bug that
    # created the corrupt file above) then never leaves a partial/corrupt
    # file at the real path for a future run to trip over.
    _atomic_write_text(json_path, report.model_dump_json(indent=2))
    _atomic_write_text(html_path, generate_html_report(report))

    n_passed = sum(r.grade.passed for r in report.results)
    print(f"\n{'=' * 60}\nRESULTS: {ticker}\n{'=' * 60}")
    print(f"Pass rate:    {report.pass_rate:.0%} ({n_passed}/{len(report.results)})")
    print(f"Avg latency:  {report.avg_latency_seconds:.1f}s per question")
    print(f"Warning rate: {report.warning_rate:.0%}\n")

    for r in report.results:
        status = "PASS" if r.grade.passed else "FAIL"
        print(f"  [{status}] {r.case.id} (score={r.grade.score:.2f}, {r.latency_seconds:.1f}s)")
        if not r.grade.passed:
            print(f"         {r.grade.detail}")

    if previous_report:
        print(f"\n{'=' * 60}\nREGRESSION CHECK (vs previous run)\n{'=' * 60}")
        messages = compare_reports(previous_report, report)
        if messages:
            for m in messages:
                print(f"  {m}")
        else:
            print("  No regressions or improvements detected.")

    print(f"\nSaved: {json_path}")
    print(f"HTML report: {html_path}  (open this in a browser)")


if __name__ == "__main__":
    main()