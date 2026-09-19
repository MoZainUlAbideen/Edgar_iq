from edgariq.indexing.chunking import chunk_filing
from edgariq.ingestion.models import FilingMetadata, FilingType
from edgariq.parsing.models import ParsedFiling, ParsedTable

METADATA = FilingMetadata(
    cik="0001045810",
    accession_number="0001045810-24-000029",
    form_type=FilingType.ANNUAL_REPORT,
    filing_date="2024-02-21",
    primary_document="nvda-20240128.htm",
)
SOURCE_URL = "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm"


def test_short_sections_get_grouped_into_fewer_chunks():
    parsed = ParsedFiling(
        text_sections=["Short sentence one.", "Short sentence two.", "Short sentence three."],
        tables=[],
    )
    chunks = chunk_filing(parsed, METADATA, SOURCE_URL)

    assert len(chunks) == 1  # small sections should merge into a single chunk
    assert chunks[0].chunk_type == "text"
    assert "Short sentence one." in chunks[0].text
    assert "Short sentence three." in chunks[0].text


def test_large_sections_split_into_multiple_chunks():
    long_section = "A" * 900
    parsed = ParsedFiling(text_sections=[long_section, long_section, long_section], tables=[])

    chunks = chunk_filing(parsed, METADATA, SOURCE_URL)

    assert len(chunks) >= 2  # shouldn't all cram into one oversized chunk


def test_tables_become_their_own_chunk_never_merged_with_text():
    parsed = ParsedFiling(
        text_sections=["Some prose about the company's performance this quarter."],
        tables=[ParsedTable(headers=["Segment", "Revenue"], rows=[["Gaming", "$2,880"]], context="Revenue table")],
    )
    chunks = chunk_filing(parsed, METADATA, SOURCE_URL)

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "Revenue table" in table_chunks[0].text
    assert "Gaming" in table_chunks[0].text
    # the table's content shouldn't leak into a text chunk
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert all("Gaming" not in c.text for c in text_chunks)


def test_chunks_carry_provenance_metadata_for_citations():
    parsed = ParsedFiling(text_sections=["Some prose here that is long enough to count."], tables=[])
    chunks = chunk_filing(parsed, METADATA, SOURCE_URL)

    meta = chunks[0].metadata
    assert meta["cik"] == "0001045810"
    assert meta["form_type"] == "10-K"
    assert meta["accession_number"] == "0001045810-24-000029"
    assert meta["source_url"] == SOURCE_URL


def test_empty_filing_produces_no_chunks():
    parsed = ParsedFiling(text_sections=[], tables=[])
    assert chunk_filing(parsed, METADATA, SOURCE_URL) == []
