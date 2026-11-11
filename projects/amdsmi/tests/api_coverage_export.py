#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Write the KPI coverage tables as an .xlsx workbook.

``api_coverage_report.py`` builds each table once, as a ``Table``; this module
turns that list into ``_api_coverage.xlsx``: a raw ``Data`` sheet holding every
table, plus one formatted sheet per table that names a ``sheet``.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

import xlsx_writer

ARROW = " → "


class Table(NamedTuple):
    """One report table. ``rows`` has the header row first and may hold markdown."""

    title: str
    rows: List[list]  # [] for a heading-only block
    # The fields below only affect the table's own XLSX sheet.
    sheet: Optional[str] = None  # sheet name; None means the table appears in Data only
    sheet_title: Optional[str] = None  # title on the sheet, defaults to ``title``
    hide_trend: bool = False  # leave the last (Trend) column off the sheet
    header_height: Optional[float] = None  # room for two-line headers
    bold_today: bool = False  # in the **Total** row, bold the middle of "S → Today → Goal"


def plain(cell):
    """A cell as plain text: strip the markdown the .md report uses."""
    text = str(cell)
    # Order matters: strip markdown bold first (adjacent '**'), then unescape
    # the literal footnote markers ('\*'), which are not adjacent stars.
    for old, new in (("<br>", " "), ("**", ""), ("\\*", "*"), ("`", ""), ("\t", " ")):
        text = text.replace(old, new)
    return text


def _sheet_value(cell):
    """Integers stay numbers in a sheet; everything else becomes plain text."""
    return cell if isinstance(cell, int) else plain(cell)


def _data_sheet(tables):
    sheet = xlsx_writer.Sheet("Data")
    for table in tables:
        sheet.add_row([table.title])
        for row in table.rows:
            sheet.add_row([_sheet_value(cell) for cell in row])
        sheet.add_row()
    sheet.autosize()
    return sheet


def _body_cells(row, bold_today):
    """Cells for one data row. A **bold** first cell (the Total row) bolds the row."""
    is_total = str(row[0]).startswith("**")
    cells = [xlsx_writer.bold(_sheet_value(row[0]))]
    for cell in row[1:]:
        text = _sheet_value(cell)
        parts = text.split(ARROW) if isinstance(text, str) else []
        if is_total and bold_today and len(parts) == 3:
            start, today, goal = parts
            runs = [(start + ARROW, False), (today, True), (ARROW + goal, False)]
            cells.append(xlsx_writer.Cell(text, runs=runs))
        elif is_total:
            cells.append(xlsx_writer.bold(text))
        else:
            cells.append(xlsx_writer.Cell(text))
    return cells


def _table_sheet(table):
    """A formatted sheet: merged bold title, bold wrapped header, bold first column."""
    width = len(table.rows[0]) - (1 if table.hide_trend else 0)
    sheet = xlsx_writer.Sheet(table.sheet)
    sheet.add_title(table.sheet_title or table.title, width)
    header = [str(cell).replace("<br>", "\n") for cell in table.rows[0][:width]]
    sheet.add_row([xlsx_writer.header(_sheet_value(cell)) for cell in header], table.header_height)
    for row in table.rows[1:]:
        sheet.add_row(_body_cells(row[:width], table.bold_today))
    sheet.autosize()
    return sheet


def write_workbook(out_dir, tables, stem="_api_coverage"):
    """Write ``<stem>.xlsx`` for ``tables`` into ``out_dir``."""
    sheets = [_data_sheet(tables)]
    sheets += [_table_sheet(table) for table in tables if table.sheet and table.rows]
    xlsx_writer.write_workbook(out_dir / f"{stem}.xlsx", sheets)
