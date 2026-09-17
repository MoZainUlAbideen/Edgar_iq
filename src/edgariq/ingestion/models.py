"""Data models for SEC EDGAR filings.

These are the shapes that flow through the rest of the pipeline (parsing,
indexing, retrieval). Keeping them as pydantic models — rather than raw
dicts — means every downstream module gets validation and autocomplete
for free, and it's obvious at a glance what a "filing" actually is.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class FilingType(str, Enum):
    """The filing forms we care about for v1.

    SEC uses dozens of form types; we deliberately scope to the ones an
    analyst actually reads for fundamentals and risk factors.
    """

    ANNUAL_REPORT = "10-K"
    QUARTERLY_REPORT = "10-Q"
    CURRENT_REPORT = "8-K"


class CompanyProfile(BaseModel):
    """Result of resolving a ticker (e.g. "NVDA") to SEC identifiers."""

    ticker: str
    cik: str = Field(description="10-digit zero-padded CIK, e.g. '0001045810'")
    name: str


class FilingMetadata(BaseModel):
    """One entry from a company's filing history."""

    cik: str
    accession_number: str = Field(description="e.g. '0001045810-24-000029'")
    form_type: FilingType
    filing_date: date
    report_date: date | None = Field(
        default=None, description="Period the filing covers, if provided"
    )
    primary_document: str = Field(
        description="Filename of the main document within the filing, e.g. 'nvda-20240128.htm'"
    )

    @property
    def accession_number_no_dashes(self) -> str:
        """SEC document URLs need the accession number without dashes."""
        return self.accession_number.replace("-", "")


class FilingDocument(BaseModel):
    """A downloaded filing, ready for the parsing stage."""

    metadata: FilingMetadata
    raw_html: str
    source_url: str
