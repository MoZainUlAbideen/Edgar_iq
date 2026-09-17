"""Real, live smoke test against SEC EDGAR.

Run this from YOUR machine (not the sandbox) once you've filled in .env:

    uv run python scripts/smoke_test.py

It should print NVIDIA's CIK, its 3 most recent 10-Q filings, and the
first 300 characters of the most recent one's HTML — proving the full
ingestion chain works end to end against the real API.
"""

from edgariq.config import settings
from edgariq.ingestion import EdgarClient, FilingType
from edgariq.parsing import parse_filing_html


def main() -> None:
    if not settings.SEC_USER_AGENT:
        raise SystemExit(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and fill it in "
            "with your real name + email, e.g. 'Zain Ahmed zain@example.com'."
        )

    client = EdgarClient(user_agent=settings.SEC_USER_AGENT)

    print("Resolving ticker NVDA...")
    profile = client.resolve_ticker("NVDA")
    print(f"  -> {profile.name} (CIK {profile.cik})\n")

    print("Fetching recent 10-Q filings...")
    filings = client.get_filing_history(
        cik=profile.cik, form_types=[FilingType.QUARTERLY_REPORT], limit=3
    )
    for f in filings:
        print(f"  - {f.form_type.value} filed {f.filing_date} (report period {f.report_date})")

    print("\nDownloading the most recent one...")
    doc = client.download_filing(filings[0])
    print(f"  -> {len(doc.raw_html):,} characters from {doc.source_url}")

    print("\nParsing the filing (this may take a few seconds on a 1MB+ document)...")
    parsed = parse_filing_html(doc.raw_html)
    print(f"  -> {len(parsed.tables)} data tables found (after filtering out layout tables)")
    print(f"  -> {len(parsed.text_sections)} text sections extracted")
    if parsed.tables:
        sample = parsed.tables[0]
        print(f"\n  Sample table (context: {sample.context!r}):")
        print("  " + sample.to_markdown().replace("\n", "\n  "))


if __name__ == "__main__":
    main()