from __future__ import annotations

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static

# Adjust this import based on where mem_bw_tree.py currently lives.
from rocprof_compute_tui.widgets.mem_bw_tree_core import TreeCanvas, TreeNode


class MemBwTreePanel(Static):
    """
    Center tab panel hosting the Mem BW decision tree.

    PoC:
      - Loads static example data from mem_bw_tree.py
      - Tracks current kernel (no metric-based logic yet)
    """

    def __init__(self, id: str = "mem-bw-tree-panel") -> None:
        super().__init__(id=id)
        self._root: Optional[TreeNode] = None
        self._current_kernel: Optional[str] = None

    def compose(self) -> ComposeResult:
        mem_bw_decision_tree_dict = {
            "label": "Workload BW Measurement",
            "children": [
                {
                    "label": "High EA -> L2 Backpressure",
                    "children": [
                        {"label": "SoC Tunning"},
                        {
                            "label": "High TCR -> TCP Backpressure",
                            "children": [
                                {
                                    "label": "vL1d Hit Low",
                                    "children": [
                                        {
                                            "label": "Non-Temporal set",
                                            "children": [
                                                {"label": "Try non-temporal (NT)"},
                                                {
                                                    "label": "High TC -> TA Backpressure",
                                                    "children": [
                                                        {
                                                            "label": "High TA -> VMEM Backpressure",
                                                            "children": [
                                                                {
                                                                    "label": "SW optimization"
                                                                },
                                                                {
                                                                    "label": "Bottleneck: Workload"
                                                                },
                                                            ],
                                                        },
                                                        {
                                                            "label": "High vL1d set full stall",
                                                            "children": [
                                                                {
                                                                    "label": "Tagram hotspotting",
                                                                    "children": [
                                                                        {
                                                                            "label": "Bottleneck: TCP capacity"
                                                                        },
                                                                        {
                                                                            "label": "SW optimization"
                                                                        },
                                                                        {
                                                                            "label": "UTCL1 Stall",
                                                                            "children": [
                                                                                {
                                                                                    "label": "UTCL1 Latency",
                                                                                    "children": [
                                                                                        {
                                                                                            "label": "Bottleneck: UTCL2"
                                                                                        }
                                                                                    ],
                                                                                }
                                                                            ],
                                                                        },
                                                                    ],
                                                                }
                                                            ],
                                                        },
                                                    ],
                                                },
                                            ],
                                        }
                                    ],
                                },
                                {
                                    "label": "L2 Channel hotspotting",
                                    "children": [
                                        {"label": "SW optimization"},
                                        {
                                            "label": "L2 Hit Low",
                                            "children": [
                                                {"label": "Try non-temporal (NT)"},
                                                {
                                                    "label": "Write Traffic",
                                                    "children": [
                                                        {"label": "Bottleneck: L2 Arb"},
                                                        {"label": "Unlikely"},
                                                    ],
                                                },
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        }

        self._root = TreeNode.from_dict(mem_bw_decision_tree_dict)

        with ScrollableContainer(id="mem-bw-tree-scroll"):
            yield TreeCanvas(self._root)

    # ------------------------------------------------------------------
    # Kernel-selection API for MainView / KernelView
    # ------------------------------------------------------------------
    def set_kernel(
        self,
        kernel_name: str,
        kernel_data: dict[str, Any] | None = None,
    ) -> None:
        """Record current kernel selection for future logic."""
        self._current_kernel = kernel_name
        # (Later: modify TreeNode metadata to reflect metrics from kernel_data)
        self.refresh()

    def get_current_kernel(self) -> Optional[str]:
        return self._current_kernel
