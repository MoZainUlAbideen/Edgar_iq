"""Tests for parse_filing_html.

The synthetic HTML below deliberately mixes layout-only tables (the kind
SEC filing software generates by the hundreds for spacing/alignment) with
one genuine financial table, to prove the numeric-ratio filter actually
discriminates between them rather than just extracting everything.
"""

from edgariq.parsing.html_parser import parse_filing_html

SAMPLE_FILING_HTML = """
<html>
<body>
  <!-- Pure layout table: single empty cell, used for spacing -->
  <table><tr><td>&nbsp;</td></tr></table>

  <!-- Another layout table: non-numeric label wrapper -->
  <table>
    <tr><td>NVIDIA Corporation</td></tr>
    <tr><td>Form 10-Q</td></tr>
  </table>

  <h2>Revenue by Segment</h2>
  <!-- A real data table -->
  <table>
    <tr><th>Segment</th><th>Q2 FY26</th><th>Q2 FY25</th></tr>
    <tr><td>Data Center</td><td>$26,272</td><td>$10,323</td></tr>
    <tr><td>Gaming</td><td>$2,880</td><td>$2,486</td></tr>
    <tr><td>Professional Visualization</td><td>$454</td><td>$379</td></tr>
  </table>

  <p>This is the risk factors section discussing export control regulations
  and their potential impact on our data center revenue going forward.</p>

  <p>Short.</p>
</body>
</html>
"""


def test_filters_out_layout_tables():
    result = parse_filing_html(SAMPLE_FILING_HTML)
    assert len(result.tables) == 1


def test_extracts_correct_headers_and_rows():
    result = parse_filing_html(SAMPLE_FILING_HTML)
    table = result.tables[0]

    assert table.headers == ["Segment", "Q2 FY26", "Q2 FY25"]
    assert ["Data Center", "$26,272", "$10,323"] in table.rows
    assert len(table.rows) == 3


def test_captures_nearby_heading_as_context():
    result = parse_filing_html(SAMPLE_FILING_HTML)
    assert "Revenue by Segment" in result.tables[0].context


def test_table_renders_to_valid_markdown():
    result = parse_filing_html(SAMPLE_FILING_HTML)
    md = result.tables[0].to_markdown()

    assert md.startswith("| Segment | Q2 FY26 | Q2 FY25 |")
    assert "Data Center" in md


def test_text_extraction_keeps_real_prose_drops_short_fragments():
    result = parse_filing_html(SAMPLE_FILING_HTML)

    assert any("export control" in s for s in result.text_sections)
    assert "Short." not in result.text_sections  # below min paragraph length


def test_empty_document_produces_no_tables_or_sections():
    result = parse_filing_html("<html><body></body></html>")
    assert result.tables == []
    assert result.text_sections == []


# Regression test for a real bug found against a live NVIDIA 10-Q: SEC
# filing markup sometimes puts a currency symbol in its own <td>, separate
# from the value, which was silently truncating the real number off rows
# that had one.
SPLIT_CURRENCY_HTML = """
<html><body>
  <h2>Future Amortization Expense</h2>
  <table>
    <tr><th>Fiscal Year</th><th>Amount</th></tr>
    <tr><td>2027</td><td>$</td><td>120</td></tr>
    <tr><td>2028</td><td>795</td></tr>
    <tr><td>Total</td><td>$</td><td>915</td></tr>
  </table>
</body></html>
"""


def test_merges_split_currency_symbol_instead_of_dropping_the_value():
    result = parse_filing_html(SPLIT_CURRENCY_HTML)
    table = result.tables[0]

    assert ["2027", "$120"] in table.rows
    assert ["2028", "795"] in table.rows
    assert ["Total", "$915"] in table.rows