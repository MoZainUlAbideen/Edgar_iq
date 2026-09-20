"""The golden set: real questions with real, human-verified expected facts.

Every expected_facts/expected_topics value here was confirmed against an
actual EdgarIQ run against real NVDA filings earlier in this project — not
invented. That matters for an eval harness: if the ground truth itself is
guessed rather than verified, a "pass" doesn't actually tell you anything.

Add more cases as you verify more facts by hand. This is intentionally a
living file, not a one-time deliverable — an eval harness is only as good
as the golden set behind it, and a golden set grows with the product.
"""

from __future__ import annotations

from edgariq.evaluation.models import EvalCase

GOLDEN_SET: list[EvalCase] = [
    EvalCase(
        id="q1_datacenter_revenue_2026q1",
        ticker="NVDA",
        question=(
            "What was NVIDIA's data center revenue in the 10-Q filed "
            "2026-05-20, and what was the year-over-year growth?"
        ),
        category="quantitative",
        expected_facts=["$75.2 billion", "92%"],
    ),
    EvalCase(
        id="q2_datacenter_revenue_2025q2",
        ticker="NVDA",
        question=(
            "What was NVIDIA's data center revenue growth in the 10-Q "
            "filed 2025-05-28?"
        ),
        category="quantitative",
        expected_facts=["$39.1 billion", "73%"],
    ),
    EvalCase(
        id="q3_datacenter_revenue_2024q3",
        ticker="NVDA",
        question=(
            "What was NVIDIA's data center revenue year-over-year growth "
            "in the 10-Q filed 2024-11-20?"
        ),
        category="quantitative",
        expected_facts=["112%"],
    ),
    EvalCase(
        id="q4_amortization_total",
        ticker="NVDA",
        question=(
            "What is NVIDIA's total estimated future amortization expense "
            "for intangible assets, according to the 10-Q filed 2026-08-26?"
        ),
        category="quantitative",
        expected_facts=["$2,998"],
    ),
    EvalCase(
        id="q5_amortization_fy2028",
        ticker="NVDA",
        question=(
            "What is NVIDIA's estimated amortization expense for fiscal "
            "year 2028, according to the 10-Q filed 2026-08-26?"
        ),
        category="quantitative",
        expected_facts=["795"],
    ),
    EvalCase(
        id="q6_export_control_china",
        ticker="NVDA",
        question="What export control risks has NVIDIA flagged related to China?",
        category="qualitative",
        expected_topics=[
            "NVIDIA is effectively foreclosed from competing in China's data center market",
            "export controls are complex, based on technical parameters like processing performance",
            "the U.S. government may impose additional or worldwide export controls in the future",
        ],
    ),
    EvalCase(
        id="q7_export_control_reputational_risk",
        ticker="NVDA",
        question="What reputational or compliance risks has NVIDIA disclosed related to export control diversion?",
        category="qualitative",
        expected_topics=[
            "allegations of diversion of controlled products to restricted destinations",
            "compliance obligations imposed on partners, suppliers, and customers are complex and burdensome",
            "these issues could harm NVIDIA's business relationships and reputation",
        ],
    ),
]
