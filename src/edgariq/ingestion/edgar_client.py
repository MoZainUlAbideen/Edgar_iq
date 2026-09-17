"""Client for the SEC EDGAR REST API.

SEC EDGAR is free and requires no API key, but it DOES require:
  1. A descriptive User-Agent header identifying you (name + email) — requests
     without one are rejected.
  2. Respecting their rate limit (max ~10 requests/second, but we're
     conservative here since this is a portfolio project, not a scraper).

Docs: https://www.sec.gov/os/accessing-edgar-data

NOTE FOR ZAIN: this client can't be tested from the sandboxed dev
environment (sec.gov isn't reachable there) — run it from your own machine
where you have real internet access. See tests/test_edgar_client.py for
mocked tests that verify the logic without hitting the network, and the
README for a real end-to-end smoke test you can run locally.
"""

from __future__ import annotations

import time
from datetime import datetime

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from edgariq.ingestion.models import (
    CompanyProfile,
    FilingDocument,
    FilingMetadata,
    FilingType,
)

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"

# SEC's company ticker -> CIK mapping. This file is a few hundred KB and
# rarely changes, so callers should cache it rather than re-fetching per
# lookup (this client caches it in-memory for the life of the instance).
TICKER_LOOKUP_URL = f"{SEC_BASE}/files/company_tickers.json"


class EdgarClient:
    def __init__(self, user_agent: str, min_request_interval: float = 0.15):
        """
        Args:
            user_agent: REQUIRED by SEC. Format: "Your Name your@email.com".
                Requests without a real identifying User-Agent get blocked.
            min_request_interval: seconds to wait between requests, to stay
                well under SEC's rate limit.
        """
        if "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with a real contact email, e.g. "
                "'Zain Ahmed zain@example.com'. See "
                "https://www.sec.gov/os/accessing-edgar-data"
            )
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._min_interval = min_request_interval
        self._last_request_time = 0.0
        self._ticker_cache: dict[str, dict] | None = None

    def _throttled_get(self, url: str, **kwargs) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        response = self._request_with_retry(url, **kwargs)
        self._last_request_time = time.monotonic()
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _request_with_retry(self, url: str, **kwargs) -> requests.Response:
        response = self._session.get(url, timeout=15, **kwargs)
        response.raise_for_status()
        return response

    def resolve_ticker(self, ticker: str) -> CompanyProfile:
        """Look up a company's CIK from its stock ticker.

        Raises:
            KeyError: if the ticker isn't found in SEC's mapping.
        """
        if self._ticker_cache is None:
            resp = self._throttled_get(TICKER_LOOKUP_URL)
            # This endpoint returns {"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...}
            self._ticker_cache = {
                entry["ticker"].upper(): entry for entry in resp.json().values()
            }

        entry = self._ticker_cache.get(ticker.upper())
        if entry is None:
            raise KeyError(f"Ticker '{ticker}' not found in SEC's company list")

        return CompanyProfile(
            ticker=entry["ticker"],
            cik=str(entry["cik_str"]).zfill(10),
            name=entry["title"],
        )

    def get_filing_history(
        self, cik: str, form_types: list[FilingType] | None = None, limit: int = 10
    ) -> list[FilingMetadata]:
        """Fetch a company's recent filing history, optionally filtered by form type."""
        url = f"{SEC_DATA_BASE}/submissions/CIK{cik}.json"
        resp = self._throttled_get(url)
        data = resp.json()

        recent = data["filings"]["recent"]
        wanted_forms = {ft.value for ft in form_types} if form_types else None

        results: list[FilingMetadata] = []
        for i, form in enumerate(recent["form"]):
            if wanted_forms and form not in wanted_forms:
                continue
            try:
                form_type = FilingType(form)
            except ValueError:
                continue  # skip forms outside our v1 scope

            report_date_str = recent["reportDate"][i] or None
            results.append(
                FilingMetadata(
                    cik=cik,
                    accession_number=recent["accessionNumber"][i],
                    form_type=form_type,
                    filing_date=datetime.strptime(recent["filingDate"][i], "%Y-%m-%d").date(),
                    report_date=(
                        datetime.strptime(report_date_str, "%Y-%m-%d").date()
                        if report_date_str
                        else None
                    ),
                    primary_document=recent["primaryDocument"][i],
                )
            )
            if len(results) >= limit:
                break

        return results

    def download_filing(self, metadata: FilingMetadata) -> FilingDocument:
        """Download the primary document of a filing (the actual 10-K/10-Q HTML)."""
        cik_no_padding = str(int(metadata.cik))
        url = (
            f"{SEC_BASE}/Archives/edgar/data/{cik_no_padding}/"
            f"{metadata.accession_number_no_dashes}/{metadata.primary_document}"
        )
        resp = self._throttled_get(url)
        return FilingDocument(metadata=metadata, raw_html=resp.text, source_url=url)
