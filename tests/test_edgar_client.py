"""Tests for EdgarClient using `responses` to mock SEC's HTTP endpoints.

These verify our parsing/logic is correct WITHOUT hitting the real network —
useful in CI and in sandboxed environments. Run the real smoke test in
scripts/smoke_test.py to confirm actual network behavior against sec.gov.
"""

import pytest
import responses

from edgariq.ingestion.edgar_client import EdgarClient
from edgariq.ingestion.models import FilingType

VALID_USER_AGENT = "Test Runner test@example.com"

FAKE_TICKER_RESPONSE = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

FAKE_SUBMISSIONS_RESPONSE = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K", "10-Q"],
            "accessionNumber": [
                "0001045810-24-000029",
                "0001045810-24-000100",
                "0001045810-24-000150",
                "0001045810-23-000200",
            ],
            "filingDate": ["2024-02-21", "2024-05-22", "2024-06-01", "2023-11-21"],
            "reportDate": ["2024-01-28", "2024-04-28", "", "2023-10-29"],
            "primaryDocument": ["nvda-20240128.htm", "nvda-20240428.htm", "form8k.htm", "nvda-20231029.htm"],
        }
    }
}


def test_rejects_user_agent_without_email():
    with pytest.raises(ValueError, match="User-Agent"):
        EdgarClient(user_agent="Zain Ahmed")


@responses.activate
def test_resolve_ticker_finds_known_company():
    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json=FAKE_TICKER_RESPONSE,
        status=200,
    )
    client = EdgarClient(user_agent=VALID_USER_AGENT, min_request_interval=0)

    profile = client.resolve_ticker("nvda")  # lowercase on purpose

    assert profile.ticker == "NVDA"
    assert profile.cik == "0001045810"
    assert profile.name == "NVIDIA CORP"


@responses.activate
def test_resolve_ticker_caches_lookup_across_calls():
    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json=FAKE_TICKER_RESPONSE,
        status=200,
    )
    client = EdgarClient(user_agent=VALID_USER_AGENT, min_request_interval=0)

    client.resolve_ticker("NVDA")
    client.resolve_ticker("AAPL")

    assert len(responses.calls) == 1  # second call used the cache, not the network


@responses.activate
def test_resolve_ticker_raises_for_unknown_ticker():
    responses.add(
        responses.GET,
        "https://www.sec.gov/files/company_tickers.json",
        json=FAKE_TICKER_RESPONSE,
        status=200,
    )
    client = EdgarClient(user_agent=VALID_USER_AGENT, min_request_interval=0)

    with pytest.raises(KeyError):
        client.resolve_ticker("NOTAREALTICKER")


@responses.activate
def test_get_filing_history_filters_by_form_type():
    responses.add(
        responses.GET,
        "https://data.sec.gov/submissions/CIK0001045810.json",
        json=FAKE_SUBMISSIONS_RESPONSE,
        status=200,
    )
    client = EdgarClient(user_agent=VALID_USER_AGENT, min_request_interval=0)

    filings = client.get_filing_history(
        cik="0001045810", form_types=[FilingType.QUARTERLY_REPORT]
    )

    assert len(filings) == 2
    assert all(f.form_type == FilingType.QUARTERLY_REPORT for f in filings)
    assert filings[0].accession_number == "0001045810-24-000100"


@responses.activate
def test_get_filing_history_respects_limit():
    responses.add(
        responses.GET,
        "https://data.sec.gov/submissions/CIK0001045810.json",
        json=FAKE_SUBMISSIONS_RESPONSE,
        status=200,
    )
    client = EdgarClient(user_agent=VALID_USER_AGENT, min_request_interval=0)

    filings = client.get_filing_history(cik="0001045810", limit=1)

    assert len(filings) == 1


@responses.activate
def test_download_filing_builds_correct_url_and_returns_html():
    from edgariq.ingestion.models import FilingMetadata

    metadata = FilingMetadata(
        cik="0001045810",
        accession_number="0001045810-24-000029",
        form_type=FilingType.ANNUAL_REPORT,
        filing_date="2024-02-21",
        primary_document="nvda-20240128.htm",
    )
    expected_url = (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581024000029/nvda-20240128.htm"
    )
    responses.add(responses.GET, expected_url, body="<html>fake filing</html>", status=200)

    client = EdgarClient(user_agent=VALID_USER_AGENT, min_request_interval=0)
    document = client.download_filing(metadata)

    assert document.raw_html == "<html>fake filing</html>"
    assert document.source_url == expected_url
