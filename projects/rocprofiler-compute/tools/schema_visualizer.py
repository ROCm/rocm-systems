#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""
Analysis database schema visualizer for rocprofiler-compute.

Renders the two diagrams shown in ``docs/how-to/analyze/cli.rst`` from the
definitions in ``src/utils/analysis_orm.py``, so neither can drift from the
code. Re-run it whenever a table, column, foreign key, or view changes, and
commit the regenerated PNGs.

Setup and workflow are documented in CONTRIBUTING.md.

Usage (from the ``rocprofiler-compute`` project root):
    ./tools/schema_visualizer.py [--output-dir DIR]
"""

import argparse
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

# This file lives under tools/; add src/ to path for rocprof_compute_* imports.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Import from the ORM module so the diagrams stay a view onto the real schema.
from sqlalchemy import MetaData, Table  # noqa: E402

from utils.analysis_orm import PREFIX, Base, Database  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "docs" / "data" / "analyze"
)
SCHEMA_DIAGRAM_NAME = "analysis_data_dump_schema"
VIEWS_DIAGRAM_NAME = "analysis_data_dump_views"

TABLE_FILL = "#1f4e79"
VIEW_FILL = "#7b3f00"
KEY_ROW_FILL = "#dce6f1"
ROW_FILL = "#ffffff"
BADGE_COLOR = "#b05a00"
TYPE_COLOR = "#666666"
RENDER_DPI = "110"

GRAPH_ATTRS = "\n".join([
    # splines=spline routes edges around nodes; ortho draws them through.
    "  graph [rankdir=LR, splines=spline, overlap=false, nodesep=0.55,",
    "         ranksep=1.9, pad=0.4, bgcolor=white, fontname=Helvetica];",
    "  node [shape=plaintext, fontname=Helvetica, fontsize=11];",
    '  edge [color="#555555", penwidth=1.1];',
])

# One row of a rendered node: column name, declared type, and PK/FK badge.
NodeColumn = tuple[str, str, str]


def render_node(name: str, columns: Iterable[NodeColumn], header_fill: str) -> str:
    """Build a Graphviz HTML-label node listing a table or view's columns.

    Each column cell is given a port named after the column so edges attach to
    the exact row rather than to the box.
    """
    column_list = list(columns)
    # Views carry no keys, so drop the badge column rather than leave it blank.
    span = 3 if any(badge for _, _, badge in column_list) else 2
    rows = [
        f'<TR><TD COLSPAN="{span}" BGCOLOR="{header_fill}">'
        f'<FONT COLOR="white" POINT-SIZE="13"><B>{name}</B></FONT></TD></TR>'
    ]
    for column_name, column_type, badge in column_list:
        fill = KEY_ROW_FILL if badge == "PK" else ROW_FILL
        label = f"<B>{column_name}</B>" if badge == "PK" else column_name
        # SQLite declares no type for computed view columns.
        type_cell = (
            f'<FONT COLOR="{TYPE_COLOR}">{column_type}</FONT>' if column_type else " "
        )
        rows.append(
            f'<TR><TD BGCOLOR="{fill}" ALIGN="LEFT" PORT="{column_name}">{label}</TD>'
            f'<TD BGCOLOR="{fill}" ALIGN="LEFT">{type_cell}</TD>'
            f"{_badge_cell(badge, fill, span)}</TR>"
        )
    table = (
        '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
        f"{''.join(rows)}</TABLE>"
    )
    return f'  "{name}" [label=<{table}>];'


def build_schema_dot(metadata: MetaData) -> str:
    """Render every ORM table as a node and every foreign key as an edge."""
    lines = ["digraph schema {", GRAPH_ATTRS]
    lines.extend(
        render_node(table.name, _table_columns(table), TABLE_FILL)
        for table in metadata.sorted_tables
    )
    for table in metadata.sorted_tables:
        lines.extend(_foreign_key_edges(table))
    lines.append("}")
    return "\n".join(lines)


def build_views_dot(metadata: MetaData) -> str:
    """Render every analysis view as a node edged to the tables it reads."""
    view_columns, view_sources = _introspect_views(set(metadata.tables))

    lines = ["digraph views {", GRAPH_ATTRS]
    lines.extend(
        render_node(view_name, columns, VIEW_FILL)
        for view_name, columns in view_columns.items()
    )
    lines.extend(
        _source_table_node(source)
        for source in sorted({s for sources in view_sources.values() for s in sources})
    )
    lines.extend(
        f'  "{source}" -> "{view_name}" [arrowhead=vee];'
        for view_name, sources in view_sources.items()
        for source in sources
    )
    lines.append("}")
    return "\n".join(lines)


def write_png(dot_source: str, output_path: Path) -> None:
    """Render Graphviz source to a PNG."""
    dot_path = output_path.with_suffix(".dot")
    dot_path.write_text(dot_source, encoding="utf-8")
    try:
        subprocess.run(
            [
                "dot",
                "-Tpng",
                f"-Gdpi={RENDER_DPI}",
                str(dot_path),
                "-o",
                str(output_path),
            ],
            check=True,
        )
    finally:
        dot_path.unlink(missing_ok=True)
    print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the PNGs into (default: %(default)s)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagrams = {
        SCHEMA_DIAGRAM_NAME: build_schema_dot(Base.metadata),
        VIEWS_DIAGRAM_NAME: build_views_dot(Base.metadata),
    }
    try:
        for name, dot_source in diagrams.items():
            write_png(dot_source, args.output_dir / f"{name}.png")
    except FileNotFoundError:
        print(
            "Error: the Graphviz 'dot' binary was not found on PATH. "
            "See CONTRIBUTING.md for setup.",
            file=sys.stderr,
        )
        return 1
    return 0


def _badge_cell(badge: str, fill: str, span: int) -> str:
    """The trailing PK/FK cell, omitted entirely for badge-less nodes."""
    if badge:
        return (
            f'<TD BGCOLOR="{fill}" ALIGN="RIGHT">'
            f'<FONT COLOR="{BADGE_COLOR}"><B>{badge}</B></FONT></TD>'
        )
    return f'<TD BGCOLOR="{fill}"> </TD>' if span == 3 else ""


def _table_columns(table: Table) -> Iterator[NodeColumn]:
    """Columns of an ORM table in declaration order, with their key badges."""
    for column in table.columns:
        badge = "PK" if column.primary_key else ("FK" if column.foreign_keys else "")
        yield column.name, _type_name(str(column.type)), badge


def _foreign_key_edges(table: Table) -> Iterator[str]:
    """One parent-port to child-port edge per foreign key, dashed when nullable."""
    for column in table.columns:
        for foreign_key in column.foreign_keys:
            target = foreign_key.column
            style = "dashed" if column.nullable else "solid"
            yield (
                f'  "{target.table.name}":"{target.name}" -> '
                f'"{table.name}":"{column.name}" '
                f"[style={style}, arrowhead=crow, arrowtail=none, dir=both];"
            )


def _introspect_views(
    table_names: set[str],
) -> tuple[dict[str, list[NodeColumn]], dict[str, list[str]]]:
    """Materialize the analysis views and read back their columns and sources."""
    with tempfile.TemporaryDirectory() as scratch_dir:
        Database.init(str(Path(scratch_dir) / "schema_visualizer.db"))
        Database.create_views()
        connection = Database.get_session().connection().connection

        view_columns: dict[str, list[NodeColumn]] = {}
        view_sources: dict[str, list[str]] = {}
        for bare_name, view_sql in Database.get_view_sql().items():
            view_name = f"{PREFIX}{bare_name}_view"
            view_columns[view_name] = [
                (row[1], _type_name(row[2]), "")
                for row in connection.execute(f"PRAGMA table_info({view_name})")
            ]
            view_sources[view_name] = _view_source_tables(view_sql, table_names)
    return view_columns, view_sources


def _view_source_tables(view_sql: str, table_names: set[str]) -> list[str]:
    """Base tables a compiled view SELECT reads from."""
    return sorted({
        name for name in re.findall(rf"\b{PREFIX}\w+", view_sql) if name in table_names
    })


def _source_table_node(table_name: str) -> str:
    """A compact name-only node for a table feeding a view."""
    return (
        f'  "{table_name}" [label=<<TABLE BORDER="0" CELLBORDER="1" '
        f'CELLSPACING="0" CELLPADDING="6"><TR><TD BGCOLOR="{TABLE_FILL}">'
        f'<FONT COLOR="white"><B>{table_name}</B></FONT></TD></TR></TABLE>>];'
    )


def _type_name(declared_type: str) -> str:
    """Bare type name, dropping any length or precision arguments."""
    return declared_type.split("(")[0]


if __name__ == "__main__":
    sys.exit(main())
