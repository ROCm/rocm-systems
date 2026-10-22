# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Select and format the blocks and metrics that --list-metrics prints."""

import shutil
import textwrap
from typing import Any, Optional

from utils.logger import console_error
from utils.metrics.expression import update_normal_unit_string
from utils.utils_common import (
    INVALID_BLOCK_HINT,
    PC_SAMPLING_BLOCK_IDS,
    ROOFLINE_PANEL_ID,
    convert_filter_blocks_to_panel_ids,
    is_membw_panel_unprofiled,
    is_pc_sampling_collected,
    metric_passes_filter,
    resolve_block_token,
)

LIST_ALL_METRICS = "all"
BLOCK_HEADERS = ("Block_ID", "Alias", "Block")
METRIC_HEADERS = ("Metric_ID", "Metric", "Unit", "Description")
METRIC_NAME_WIDTH_MAX = 30
UNIT_WIDTH_MAX = 16
DESCRIPTION_WIDTH_MIN = 40
# Used when the output is not a terminal, e.g. piped to a file
TERMINAL_WIDTH_FALLBACK = 120


def keep_profiled_ids(
    metric_ids: list[str],
    profiling_config: dict[str, Any],
    has_roofline_data: bool,
    arch: str,
) -> list[str]:
    """Keep the ids of the blocks a workload profiled and has data for: the
    blocks its profiling config selects, Roofline only with roofline data, and
    PC Sampling only when it was collected."""
    filter_blocks = profiling_config.get("filter_blocks") or []
    profiled_panel_ids = convert_filter_blocks_to_panel_ids(filter_blocks, arch)
    skipped_block_ids = set()
    if not has_roofline_data:
        skipped_block_ids.add(str(ROOFLINE_PANEL_ID // 100))
    if not is_pc_sampling_collected(filter_blocks):
        skipped_block_ids.add(PC_SAMPLING_BLOCK_IDS[0])
    kept_block_ids = {
        block_id
        for block_id in metric_ids
        if "." not in block_id
        and block_id not in skipped_block_ids
        and _is_block_profiled(block_id, profiled_panel_ids, profiling_config)
    }
    return [
        metric_id
        for metric_id in metric_ids
        if metric_id.split(".")[0] in kept_block_ids
    ]


def select_listed_ids(
    metric_ids: list[str], selection: list[str], alias_to_block_id: dict[str, str]
) -> list[str]:
    """Return the ids to list: the blocks when nothing is selected, every id for
    "all", otherwise the selected ids with the ids above and below them."""
    if not selection:
        return [metric_id for metric_id in metric_ids if "." not in metric_id]
    if selection == [LIST_ALL_METRICS]:
        return metric_ids
    if LIST_ALL_METRICS in selection:
        console_error(
            f'"{LIST_ALL_METRICS}" lists every metric; pass it without other ids.'
        )

    selected_ids = []
    for token in selection:
        selected_id = resolve_block_token(token, alias_to_block_id)
        if selected_id not in metric_ids:
            console_error(f"Unknown block or metric id {token!r}.{INVALID_BLOCK_HINT}")
        selected_ids.append(selected_id)
    return [
        metric_id
        for metric_id in metric_ids
        if any(_is_same_id_branch(metric_id, selected) for selected in selected_ids)
    ]


def format_listing_header(
    arch: str, series: Optional[str], workload_dir: Optional[str]
) -> str:
    """Format the lines that say which workload and platform are listed."""
    lines = [f"Workload: {workload_dir}"] if workload_dir is not None else []
    lines.append(f"Platform: {arch} ({series})" if series else f"Platform: {arch}")
    return "\n".join(lines) + "\n"


def format_listing_body(
    metric_details: dict[str, tuple[str, str, str]],
    listed_ids: list[str],
    block_aliases: dict[str, str],
    per_channel_table_ids: set[str],
    show_metrics: bool,
) -> str:
    """Format the metric tables of the listed blocks, or, when no blocks or
    metrics were named, the block table with a hint on listing metrics."""
    if show_metrics:
        return format_metric_tables(
            metric_details, listed_ids, block_aliases, per_channel_table_ids
        )
    return (
        format_block_table(metric_details, listed_ids, block_aliases)
        + '\n\nAdd block ids or aliases, or "all", to --list-metrics to list '
        "their metrics."
    )


def format_block_table(
    metric_details: dict[str, tuple[str, str, str]],
    block_ids: list[str],
    block_aliases: dict[str, str],
) -> str:
    """Format the Block_ID, Alias and Block table."""
    rows = [
        (block_id, block_aliases.get(block_id, ""), metric_details[block_id][0])
        for block_id in block_ids
    ]
    widths = [
        max([len(header), *(len(row[column]) for row in rows)])
        for column, header in enumerate(BLOCK_HEADERS)
    ]
    return "\n".join(_format_grid(BLOCK_HEADERS, rows, widths, row_separators=False))


def format_metric_tables(
    metric_details: dict[str, tuple[str, str, str]],
    listed_ids: list[str],
    block_aliases: dict[str, str],
    per_channel_table_ids: set[str],
) -> str:
    """Format one grid per metric table, under block and table headings. All
    grids share column widths so they line up."""
    metric_rows = {
        metric_id: _metric_row(metric_id, metric_details[metric_id])
        for metric_id in listed_ids
        if metric_id.count(".") == 2
    }
    widths = _metric_column_widths(list(metric_rows.values()))

    lines: list[str] = []
    for block_id in [metric_id for metric_id in listed_ids if "." not in metric_id]:
        heading = f"{block_id}. {metric_details[block_id][0]}"
        if block_aliases.get(block_id):
            heading += f" (alias: {block_aliases[block_id]})"
        lines.append(heading)
        lines.extend(
            _format_block_tables(
                block_id,
                listed_ids,
                metric_details,
                metric_rows,
                widths,
                per_channel_table_ids,
            )
        )
    return "\n".join(lines)


def _is_block_profiled(
    block_id: str, profiled_panel_ids: set[int], profiling_config: dict[str, Any]
) -> bool:
    panel_id = int(block_id) * 100
    if is_membw_panel_unprofiled(panel_id, profiling_config):
        return False
    return metric_passes_filter(
        metric_id=block_id,
        panel_id=panel_id,
        data_source_idx=block_id,
        user_metric_filter=None,
        profile_panel_filter=profiled_panel_ids,
    )


def _is_same_id_branch(metric_id: str, selected_id: str) -> bool:
    """Return True if metric_id is selected_id, or above or below it."""
    return (
        metric_id == selected_id
        or metric_id.startswith(f"{selected_id}.")
        or selected_id.startswith(f"{metric_id}.")
    )


def _format_block_tables(
    block_id: str,
    listed_ids: list[str],
    metric_details: dict[str, tuple[str, str, str]],
    metric_rows: dict[str, tuple[str, str, str, str]],
    widths: list[int],
    per_channel_table_ids: set[str],
) -> list[str]:
    """Format the table headings and grids of one block. Per-channel tables
    collapse to one line; tables with no metrics on this arch are skipped."""
    lines: list[str] = []
    per_channel_tables: list[str] = []
    table_ids = [
        metric_id
        for metric_id in listed_ids
        if metric_id.count(".") == 1 and metric_id.split(".")[0] == block_id
    ]
    for table_id in table_ids:
        rows = [
            row
            for row_id, row in metric_rows.items()
            if row_id.startswith(f"{table_id}.")
        ]
        if rows:
            lines.append(f"{table_id} {metric_details[table_id][0]}")
            lines.extend(
                _format_grid(METRIC_HEADERS, rows, widths, row_separators=True)
            )
            lines.append("")
        elif table_id in per_channel_table_ids:
            per_channel_tables.append(table_id)

    if per_channel_tables:
        span = per_channel_tables[0]
        if len(per_channel_tables) > 1:
            span += f" to {per_channel_tables[-1]}"
        lines.extend([f"Per-channel metrics are not listed: {span}", ""])
    if not lines:
        lines.extend(["No metrics to list.", ""])
    return lines


def _metric_row(
    metric_id: str, details: tuple[str, str, str]
) -> tuple[str, str, str, str]:
    name, unit, description = details
    return metric_id, name, update_normal_unit_string(unit, "per_normUnit"), description


def _metric_column_widths(rows: list[tuple[str, str, str, str]]) -> list[int]:
    """Size Metric_ID to fit, cap Metric and Unit, and give Description the
    rest of the terminal width, but at least DESCRIPTION_WIDTH_MIN."""
    id_width, name_width, unit_width = (
        max([len(METRIC_HEADERS[column]), *(len(row[column]) for row in rows)])
        for column in range(3)
    )
    name_width = min(name_width, METRIC_NAME_WIDTH_MAX)
    unit_width = min(unit_width, UNIT_WIDTH_MAX)
    terminal_width = shutil.get_terminal_size((TERMINAL_WIDTH_FALLBACK, 24)).columns
    # Each of the 4 columns adds a border and a space on each side, plus 1 border
    border_width = 3 * len(METRIC_HEADERS) + 1
    description_width = max(
        DESCRIPTION_WIDTH_MIN,
        terminal_width - id_width - name_width - unit_width - border_width,
    )
    return [id_width, name_width, unit_width, description_width]


def _format_grid(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    widths: list[int],
    row_separators: bool,
) -> list[str]:
    """Format a box-drawn grid in the style of the analyze tables, wrapping
    each cell to its column width."""
    lines = [
        _grid_rule("╒", "╤", "╕", "═", widths),
        *_grid_row(headers, widths),
        _grid_rule("╞", "╪", "╡", "═", widths),
    ]
    for index, row in enumerate(rows):
        if row_separators and index > 0:
            lines.append(_grid_rule("├", "┼", "┤", "─", widths))
        lines.extend(_grid_row(row, widths))
    lines.append(_grid_rule("╘", "╧", "╛", "═", widths))
    return lines


def _grid_rule(left: str, middle: str, right: str, fill: str, widths: list[int]) -> str:
    return left + middle.join(fill * (width + 2) for width in widths) + right


def _grid_row(cells: tuple[str, ...], widths: list[int]) -> list[str]:
    """Return the text lines of one grid row, one line per wrapped cell line."""
    wrapped_cells = [
        textwrap.wrap(cell, width) or [""] for cell, width in zip(cells, widths)
    ]
    height = max(len(cell_lines) for cell_lines in wrapped_cells)
    padded_cells = [
        cell_lines + [""] * (height - len(cell_lines)) for cell_lines in wrapped_cells
    ]
    return [
        "│"
        + "│".join(
            f" {cell_lines[line]:<{width}} "
            for cell_lines, width in zip(padded_cells, widths)
        )
        + "│"
        for line in range(height)
    ]
