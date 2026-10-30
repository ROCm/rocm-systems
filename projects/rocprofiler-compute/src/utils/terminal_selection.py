# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Which blocks, tables and metrics analyze prints on the terminal.

The terminal always shows the panels up to speed-of-light. Analyze -b/--block
adds blocks, tables or single metrics to that. report.txt is not affected and
always holds every block.
"""

from dataclasses import dataclass
from typing import FrozenSet, List, Optional

from utils.utils_common import METRIC_ID_RE, convert_filter_blocks_to_panel_ids

# The last panel printed on the terminal without -b: System Speed-of-Light
TERMINAL_LAST_PANEL_ID = 200


def table_id_string(table_id: int) -> str:
    """Return the "block.table" id of a table, e.g. "7.1" for table 701."""
    return f"{table_id // 100}.{table_id % 100}"


@dataclass(frozen=True)
class TerminalSelection:
    """The terminal selection made with analyze -b.

    metric_ids holds the numeric -b tokens: block ("7"), table ("7.1") or
    metric ("7.1.5") ids. panel_ids holds the panels named by block aliases.
    """

    metric_ids: FrozenSet[str] = frozenset()
    panel_ids: FrozenSet[int] = frozenset()

    def shows_whole_panel(self, panel_id: int) -> bool:
        """Return True if every table and metric of the panel is printed."""
        return (
            panel_id <= TERMINAL_LAST_PANEL_ID
            or panel_id in self.panel_ids
            or str(panel_id // 100) in self.metric_ids
        )

    def shows_panel(self, panel_id: int) -> bool:
        """Return True if any part of the panel is printed."""
        block_prefix = f"{panel_id // 100}."
        return self.shows_whole_panel(panel_id) or any(
            metric_id.startswith(block_prefix) for metric_id in self.metric_ids
        )

    def shows_table(self, panel_id: int, table_id: int) -> bool:
        """Return True if any row of the table is printed."""
        table_token = table_id_string(table_id)
        return (
            self.shows_whole_panel(panel_id)
            or table_token in self.metric_ids
            or any(
                metric_id.startswith(f"{table_token}.") for metric_id in self.metric_ids
            )
        )

    def shows_metric(self, panel_id: int, table_id: int, metric_id: str) -> bool:
        """Return True if the metric row ("7.1.5") of the table is printed."""
        return (
            self.shows_whole_panel(panel_id)
            or table_id_string(table_id) in self.metric_ids
            or metric_id in self.metric_ids
        )


def build_terminal_selection(
    block_tokens: Optional[List[str]], arch: Optional[str]
) -> TerminalSelection:
    """Build the terminal selection from the analyze -b tokens.

    Block aliases are resolved for arch; an unknown alias is an error.
    """
    tokens = [str(token) for token in block_tokens or []]
    alias_tokens = [token for token in tokens if not METRIC_ID_RE.match(token)]
    return TerminalSelection(
        metric_ids=frozenset(token for token in tokens if METRIC_ID_RE.match(token)),
        panel_ids=frozenset(convert_filter_blocks_to_panel_ids(alias_tokens, arch)),
    )
