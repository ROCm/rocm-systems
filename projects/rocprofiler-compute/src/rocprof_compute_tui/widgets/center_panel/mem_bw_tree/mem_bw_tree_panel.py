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
from rocprof_compute_tui.widgets.center_panel.mem_bw_tree.tree_spec import (
    PATHS,
    TREE_DICT,
)
from rocprof_compute_tui.widgets.center_panel.mem_bw_tree.ui_model import DecisionResult
from rocprof_compute_tui.widgets.mem_bw_tree_core import (
    NodeSelected,
    TreeCanvas,
    TreeNode,
)


class MemBwTreePanel(Static):
    """
    Center tab panel hosting the Mem BW decision tree (beta v1).

    Beta v1 features:
      - Evaluate a metric-driven rule set (ID-based)
      - Highlight an active path when the rule passes
      - Provide a click-to-explain details panel for the selected node
    """

    DEFAULT_CSS = """
    MemBwTreePanel {
        layout: vertical;
    }

    #mem-bw-tree-scroll {
        height: 1fr;
        width: 1fr;
    }

    #mem-bw-details {
        height: 12;
        border: solid $border;
        padding: 1 1;
        margin-top: 1;
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
        self._last_decision: Optional[DecisionResult] = None  # DecisionResult
        self._details: Optional[Static] = None

    def compose(self) -> ComposeResult:
        self._root = TreeNode.from_dict(TREE_DICT)

        with ScrollableContainer(id="mem-bw-tree-scroll"):
            yield TreeCanvas(self._root)

        self._details = Static("", id="mem-bw-details")
        yield self._details

    # ------------------------------------------------------------------
    # Kernel-selection API for MainView / KernelView
    # ------------------------------------------------------------------
    def set_kernel(
        self,
        kernel_name: str,
        kernel_data: Optional[dict[str, Any]] = None,
    ) -> None:
        self._current_kernel = kernel_name
        self._kernel_data = kernel_data or {}

        # 1) Extract required metrics by ID
        required = self._engine.required_metric_ids()
        snap = MetricExtractor.extract(self._kernel_data, required)

        # 2) Evaluate rules -> decision result (active path + explanations)
        decision = self._engine.evaluate(snap, PATHS)
        self._last_decision = (decision, snap)

        # 3) Apply decision to tree node tags + metadata
        if self._root is not None:
            self._apply_decision_to_tree(self._root, decision)

        # 4) Update details to match current selected node
        self._update_details_for_selected()

        # 5) Redraw
        self.refresh()

    def get_current_kernel(self) -> Optional[str]:
        return self._current_kernel

    # ------------------------------------------------------------------
    # Decision -> UI mapping
    # ------------------------------------------------------------------
    def _apply_decision_to_tree(
        self, root: TreeNode, decision: Optional[DecisionResult] = None
    ) -> None:
        active_ids = set(decision.active_path or [])

        # If we have an active path, dim everything else for clarity.
        has_active = bool(active_ids)

        for n in root.walk():
            n.tags.clear()
            n.metadata = (
                ""  # beta: keep boxes clean; details panel shows full explanation
            )

            if has_active:
                if n.node_id in active_ids:
                    n.tags.add("active")
                else:
                    n.tags.add("dim")

        # If a rule exists for a node and it failed due to missing metrics, mark warn
        for node_id, ev in decision.node_eval.items():
            if not ev.passed:
                # If missing metrics, show warn; otherwise keep dim/default.
                # We'll infer missing from details text for now (beta).
                if root.find_by_id(node_id):
                    # Tag warn even if dim, so it stands out when selected.
                    root.find_by_id(node_id).tags.add("warn")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Node selection -> details
    # ------------------------------------------------------------------
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

        title = f"[b]Mem BW Tree[/b]  |  Kernel: {self._current_kernel or 'N/A'}\n"
        title += f"[b]Selected Node:[/b] {node_id}\n\n"

        if self._last_decision is None:
            self._details.update(
                title + "No decision data yet. Run analysis / open Mem BW Tree."
            )
            return

        decision, snap = self._last_decision

        ev = decision.node_eval.get(node_id)
        if ev is None:
            # No rule attached to this node in beta
            self._details.update(
                title
                + "No rule attached to this node (beta).\n"
                + "As we scale, we’ll attach rule logic to more nodes."
            )
            return

        status = "PASS" if ev.passed else "FAIL"
        status_class = "details-pass" if ev.passed else "details-fail"

        lines = [title, f"[{status_class}]{status}[/{status_class}]\n"]

        for pr in ev.predicate_results:
            lines.append(f"[b]Expression[/b]: {pr.expression}\n")
            lines.append(f"{pr.details}\n")

            # Inputs with provenance (panel/table)
            lines.append("[b]Inputs[/b]:\n")
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

        if snap.missing:
            lines.append(
                "\n[details-warn]Missing metrics "
                "(not found in kernel data):[/details-warn]\n"
            )
            for mid in snap.missing:
                lines.append(f"  - {mid}\n")

        self._details.update("".join(lines))
