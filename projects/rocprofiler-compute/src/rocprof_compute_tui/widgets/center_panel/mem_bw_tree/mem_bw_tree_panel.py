##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

from __future__ import annotations

from typing import Any, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static

from rocprof_compute_tui.widgets.center_panel.mem_bw_tree.engine import DecisionEngine
from rocprof_compute_tui.widgets.center_panel.mem_bw_tree.metric_extract import (
    MetricExtractor,
)
from rocprof_compute_tui.widgets.center_panel.mem_bw_tree.tree_spec import TREE_DICT
from rocprof_compute_tui.widgets.center_panel.mem_bw_tree.ui_model import (
    DecisionResult,
    EvalStatus,
)
from rocprof_compute_tui.widgets.mem_bw_tree_core import (
    NodeSelected,
    TreeCanvas,
    TreeNode,
)


class MemBwTreePanel(Static):
    """
    Center tab panel hosting the Mem BW decision tree (BFS engine).

    Features:
      - BFS traversal of the decision tree
      - Every reached node is highlighted
      - Node status: TRUE / FALSE / ERROR / NO_RULE
      - Click-to-explain predicate details
    """

    DEFAULT_CSS = """
    MemBwTreePanel {
        layout: vertical;
    }

    #mem-bw-tree-scroll {
        height: 1fr;
        width: 1fr;
    }

    #mem-bw-details-scroll {
        height: 12;
        border: solid $border;
        padding: 0;
        margin-top: 1;
    }

    #mem-bw-details {
        padding: 1 1;
    }

    .details-title {
        text-style: bold;
    }

    .details-pass {
        color: $success;
    }

    .details-fail {
        color: $error;
    }

    .details-warn {
        color: $warning;
    }
    """

    def __init__(self, id: str = "mem-bw-tree-panel") -> None:
        super().__init__(id=id)
        self._root: Optional[TreeNode] = None
        self._current_kernel: Optional[str] = None
        self._kernel_data: Optional[dict[str, Any]] = None

        self._engine = DecisionEngine()
        self._last_decision: Optional[DecisionResult] = None
        self._last_snapshot = None
        self._details: Optional[Static] = None

    def compose(self) -> ComposeResult:
        self._root = TreeNode.from_dict(TREE_DICT)

        with ScrollableContainer(id="mem-bw-tree-scroll"):
            yield TreeCanvas(self._root)

        with ScrollableContainer(id="mem-bw-details-scroll"):
            self._details = Static("", id="mem-bw-details")
            yield self._details

    def set_kernel(
        self,
        kernel_name: str,
        kernel_data: Optional[dict[str, Any]] = None,
    ) -> None:
        self._current_kernel = kernel_name
        self._kernel_data = kernel_data or {}

        # 1) Extract required metrics
        required = self._engine.required_metric_ids()
        snap = MetricExtractor.extract(self._kernel_data, required)

        # 2) BFS evaluate the tree
        assert self._root is not None
        decision = self._engine.evaluate(snap, self._root)

        self._last_decision = decision
        self._last_snapshot = snap

        # 3) Apply decision results to tree nodes
        self._apply_decision_to_tree(self._root, decision)

        # 4) Update details panel for current selection
        self._update_details_for_selected()

        # 5) Redraw
        self.refresh()

    def get_current_kernel(self) -> Optional[str]:
        return self._current_kernel

    def _apply_decision_to_tree(
        self,
        root: TreeNode,
        decision: Optional[DecisionResult],
    ) -> None:
        """
        Apply EvalStatus-based tags to each reached node.
        """
        # Clear all tags first
        for n in root.walk():
            n.tags.clear()
            n.metadata = ""

        if decision is None:
            return

        for node_id, ev in decision.node_eval.items():
            node = root.find_by_id(node_id)
            if node is None:
                continue

            if ev.status == EvalStatus.TRUE:
                node.tags.add("eval_true")
            elif ev.status == EvalStatus.FALSE:
                node.tags.add("eval_false")
            elif ev.status == EvalStatus.ERROR:
                node.tags.add("eval_error")
            elif ev.status == EvalStatus.NO_RULE:
                node.tags.add("eval_no_rule")

    @on(NodeSelected)
    def _on_node_selected(self, event: NodeSelected) -> None:
        self._update_details(event.node_id)

    def _update_details_for_selected(self) -> None:
        canvas = self.query_one("#mem-bw-tree-canvas", TreeCanvas)
        if canvas and canvas.selected:
            self._update_details(canvas.selected.node_id)

    def _update_details(self, node_id: str) -> None:
        if self._details is None:
            return

        title = (
            f"Mem BW Tree  |  Kernel: {self._current_kernel or 'N/A'}\n"
            f"Selected Node: {node_id}\n\n"
        )

        if self._last_decision is None or self._last_snapshot is None:
            self._details.update(
                title + "No decision data yet. Run analysis to evaluate the tree."
            )
            return

        decision = self._last_decision
        snap = self._last_snapshot

        ev = decision.node_eval.get(node_id)
        if ev is None:
            self._details.update(title + "Node was not reached during BFS evaluation.")
            return

        # Status header
        if ev.status == EvalStatus.TRUE:
            status = "[details-pass]PASS[/details-pass]"
        elif ev.status == EvalStatus.FALSE:
            status = "[details-fail]FAIL[/details-fail]"
        elif ev.status == EvalStatus.ERROR:
            status = "[details-warn]ERROR[/details-warn]"
        else:  # NO_RULE
            status = "REACHED (no rule)"

        lines = [title, f"{status}\n\n"]

        # Error details
        if ev.status == EvalStatus.ERROR and ev.error:
            lines.append(f"[details-warn]{ev.error}[/details-warn]\n")

        # Predicate details
        for pr in ev.predicate_results:
            lines.append(f"Expression: {pr.expression}\n")
            lines.append(f"{pr.details}\n\n")

            lines.append("Inputs:\n")
            for mid, val in pr.inputs.items():
                mm = snap.meta.get(mid)
                if mm:
                    unit = f" {mm.unit}" if mm.unit else ""
                    lines.append(
                        f"  - {mid}: {val:.6g}{unit}\n"
                        f"      {mm.panel} → {mm.table}\n"
                        f"      {mm.metric_name}\n"
                    )
                else:
                    lines.append(f"  - {mid}: {val:.6g}\n")
            lines.append("\n")

        # Missing metrics
        if snap.missing:
            lines.append(
                "[details-warn]Missing metrics "
                "(not found in kernel data):[/details-warn]\n"
            )
            for mid in snap.missing:
                lines.append(f"  - {mid}\n")

        self._details.update("".join(lines))
