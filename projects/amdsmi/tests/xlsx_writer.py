#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Minimal .xlsx writer using only the standard library.

Just enough of SpreadsheetML for the KPI coverage workbook: several sheets,
text and integer cells, bold text (whole cell or runs inside a cell), wrapped
header cells, merged ranges and column widths. Kept dependency-free because
``api_coverage_report.py`` runs on test hosts where openpyxl/xlsxwriter are not
installed.
"""

from __future__ import annotations

import re
import zipfile
from xml.sax.saxutils import escape

# Style ids, matching the <cellXfs> order written by _STYLES below.
STYLE_NORMAL = 0
STYLE_BOLD = 1
STYLE_HEADER = 2  # bold, wrapped, vertically centred

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2">
<font><sz val="11"/><name val="Calibri"/></font>
<font><b/><sz val="11"/><name val="Calibri"/></font>
</fonts>
<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>
</cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>
"""


class Cell:
    """One cell: a value, a style id, or ``runs`` of (text, bold) for mixed bold."""

    def __init__(self, value="", style=STYLE_NORMAL, runs=None):
        self.value = value
        self.style = style
        self.runs = runs


def bold(value):
    return Cell(value, STYLE_BOLD)


def header(value):
    return Cell(value, STYLE_HEADER)


class Sheet:
    def __init__(self, name):
        # Excel limits: 31 chars, none of : \ / ? * [ ]
        self.name = re.sub(r"[:\\/?*\[\]]", "-", name)[:31]
        self.rows = []  # list of (list[Cell | None], height or None)
        self.merges = []  # "A1:E1"
        self.widths = {}

    def add_row(self, cells=(), height=None):
        self.rows.append(([_as_cell(c) for c in cells], height))
        return len(self.rows)  # 1-based row number

    def add_title(self, text, width):
        """Add a bold title row merged across the first ``width`` columns."""
        row = self.add_row([bold(text)])
        if width > 1:
            self.merges.append(f"A{row}:{_col_letter(width)}{row}")

    def autosize(self):
        """Fit column widths to content, ignoring single-cell title rows."""
        for cells, _ in self.rows:
            filled = [c for c in cells if c is not None and c.value not in ("", None)]
            if len(filled) <= 1:
                continue
            for index, cell in enumerate(cells, start=1):
                if cell is None or cell.value in ("", None):
                    continue
                longest = max(len(line) for line in str(cell.value).split("\n"))
                width = min(60, longest * 1.15 + 2)
                self.widths[index] = max(self.widths.get(index, 8), width)


def _as_cell(value):
    if isinstance(value, Cell):
        return value
    return Cell("" if value is None else value)


def _col_letter(index):
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _text(value):
    return escape(_CONTROL_CHARS.sub("", str(value)))


def _cell_xml(ref, cell):
    style = f' s="{cell.style}"' if cell.style else ""
    if cell.runs:
        parts = []
        for text, bold in cell.runs:
            props = (
                "<rPr>" + ("<b/>" if bold else "") + '<sz val="11"/><rFont val="Calibri"/></rPr>'
            )
            parts.append(f'<r>{props}<t xml:space="preserve">{_text(text)}</t></r>')
        return f'<c r="{ref}"{style} t="inlineStr"><is>{"".join(parts)}</is></c>'
    value = cell.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if value in ("", None):
            return f'<c r="{ref}"{style}/>' if cell.style else ""
        return f'<c r="{ref}"{style} t="inlineStr"><is><t xml:space="preserve">{_text(value)}</t></is></c>'
    return f'<c r="{ref}"{style}><v>{value}</v></c>'


def _sheet_xml(sheet):
    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    ]
    if sheet.widths:
        out.append("<cols>")
        for index in sorted(sheet.widths):
            out.append(
                f'<col min="{index}" max="{index}" width="{sheet.widths[index]:.1f}" customWidth="1"/>'
            )
        out.append("</cols>")
    out.append("<sheetData>")
    for row_no, (cells, height) in enumerate(sheet.rows, start=1):
        body = "".join(
            _cell_xml(f"{_col_letter(col)}{row_no}", cell)
            for col, cell in enumerate(cells, start=1)
            if cell is not None
        )
        attrs = f' ht="{height}" customHeight="1"' if height else ""
        if body or attrs:
            out.append(f'<row r="{row_no}"{attrs}>{body}</row>')
    out.append("</sheetData>")
    if sheet.merges:
        out.append(f'<mergeCells count="{len(sheet.merges)}">')
        out.extend(f'<mergeCell ref="{ref}"/>' for ref in sheet.merges)
        out.append("</mergeCells>")
    out.append("</worksheet>")
    return "".join(out)


_XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_SHEETML_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml."


def _content_types_xml(sheet_count):
    sheet_parts = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="{_SHEETML_TYPE}worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        _XML_DECL + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/xl/workbook.xml" ContentType="{_SHEETML_TYPE}sheet.main+xml"/>'
        f'<Override PartName="/xl/styles.xml" ContentType="{_SHEETML_TYPE}styles+xml"/>'
        + sheet_parts
        + "</Types>"
    )


def _root_rels_xml():
    return (
        _XML_DECL + f'<Relationships xmlns="{_NS_PKG_REL}">'
        f'<Relationship Id="rId1" Type="{_NS_REL}/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _workbook_xml(sheets):
    entries = "".join(
        f'<sheet name="{escape(s.name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, s in enumerate(sheets, start=1)
    )
    return (
        _XML_DECL
        + f'<workbook xmlns="{_NS_MAIN}" xmlns:r="{_NS_REL}"><sheets>{entries}</sheets></workbook>'
    )


def _workbook_rels_xml(sheet_count):
    """Sheets are rId1..rIdN; the stylesheet takes the next id."""
    sheet_rels = "".join(
        f'<Relationship Id="rId{i}" Type="{_NS_REL}/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, sheet_count + 1)
    )
    styles_rel = (
        f'<Relationship Id="rId{sheet_count + 1}" Type="{_NS_REL}/styles" Target="styles.xml"/>'
    )
    return (
        _XML_DECL + f'<Relationships xmlns="{_NS_PKG_REL}">{sheet_rels}{styles_rel}</Relationships>'
    )


def write_workbook(path, sheets):
    """Write ``sheets`` (a list of Sheet) to ``path`` as an .xlsx file."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml(len(sheets)))
        archive.writestr("_rels/.rels", _root_rels_xml())
        archive.writestr("xl/workbook.xml", _workbook_xml(sheets))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(sheets)))
        archive.writestr("xl/styles.xml", _STYLES)
        for index, sheet in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(sheet))
