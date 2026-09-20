"""Generates a self-contained, styled HTML report from an EvalReport —
polished enough to screenshot for a portfolio/LinkedIn post, and useful on
its own as a quick pass/fail dashboard after every run.
"""

from __future__ import annotations

import html

from edgariq.evaluation.models import EvalReport

_STYLE = """
:root {
  --bg: #0f1117; --card: #171a23; --border: #262a36; --text: #e6e8ee; --muted: #9aa1b1;
  --pass: #3ddc84; --fail: #ff5c6c; --accent: #6c8cff;
}
body { background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 2rem; }
h1 { margin-bottom: 0.25rem; }
.meta { color: var(--muted); margin-bottom: 2rem; }
.summary { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.stat { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.5rem; min-width: 160px; }
.stat .label { color: var(--muted); font-size: 0.85rem; }
.stat .value { font-size: 1.8rem; font-weight: 700; margin-top: 0.25rem; }
.case { background: var(--card); border: 1px solid var(--border); border-left: 4px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem; }
.case.pass { border-left-color: var(--pass); }
.case.fail { border-left-color: var(--fail); }
.case-header { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.5rem; font-size: 0.85rem; color: var(--muted); }
.badge { font-weight: 700; padding: 0.1rem 0.6rem; border-radius: 999px; font-size: 0.75rem; }
.badge.pass { background: rgba(61,220,132,0.15); color: var(--pass); }
.badge.fail { background: rgba(255,92,108,0.15); color: var(--fail); }
.case-id { font-family: monospace; color: var(--accent); }
.question { font-weight: 600; margin-bottom: 0.4rem; }
.detail { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.4rem; }
details summary { cursor: pointer; color: var(--accent); font-size: 0.85rem; }
pre { white-space: pre-wrap; background: #0d0f15; border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; margin-top: 0.5rem; font-size: 0.85rem; }
"""


def _render_case(result) -> str:
    status = "PASS" if result.grade.passed else "FAIL"
    status_class = "pass" if result.grade.passed else "fail"
    return f"""
<div class="case {status_class}">
  <div class="case-header">
    <span class="badge {status_class}">{status}</span>
    <span class="case-id">{html.escape(result.case.id)}</span>
    <span>{html.escape(result.case.category)}</span>
    <span>score: {result.grade.score:.2f}</span>
    <span>{result.latency_seconds:.2f}s</span>
  </div>
  <div class="question">{html.escape(result.case.question)}</div>
  <div class="detail">{html.escape(result.grade.detail)}</div>
  <details><summary>Full answer</summary><pre>{html.escape(result.answer.answer)}</pre></details>
</div>
"""


def generate_html_report(report: EvalReport) -> str:
    cases_html = "".join(_render_case(r) for r in report.results)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EdgarIQ Eval Report — {html.escape(report.ticker)}</title>
<style>{_STYLE}</style>
</head>
<body>
  <h1>EdgarIQ Eval Report</h1>
  <div class="meta">{html.escape(report.ticker)} — run at {html.escape(report.run_at)}</div>
  <div class="summary">
    <div class="stat"><div class="label">Pass rate</div><div class="value">{report.pass_rate:.0%}</div></div>
    <div class="stat"><div class="label">Avg latency</div><div class="value">{report.avg_latency_seconds:.1f}s</div></div>
    <div class="stat"><div class="label">Warning rate</div><div class="value">{report.warning_rate:.0%}</div></div>
    <div class="stat"><div class="label">Cases</div><div class="value">{len(report.results)}</div></div>
  </div>
  {cases_html}
</body>
</html>"""
