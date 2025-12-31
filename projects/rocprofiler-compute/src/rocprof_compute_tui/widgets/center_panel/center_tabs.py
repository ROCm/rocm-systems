##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import TabbedContent, TabPane

from rocprof_compute_tui.views.kernel_view import KernelView
from rocprof_compute_tui.widgets.center_panel.mem_bw_tree.mem_bw_tree_panel import (
    MemBwTreePanel,
)


class CenterTabs(Container):
    """
    Tabbed center panel containing:
      - KernelView
      - Mem BW Tree View
    """

    def __init__(self) -> None:
        super().__init__(id="center-panel")
        self._kernel_view: Optional[KernelView] = None
        self._mem_tree: Optional[MemBwTreePanel] = None

    def compose(self) -> ComposeResult:
        with TabbedContent(id="center-tabbed"):
            with TabPane("Kernel View", id="tab-kernel-view"):
                kv = KernelView()
                self._kernel_view = kv
                yield kv

            with TabPane("Mem BW Tree", id="tab-mem-tree"):
                mt = MemBwTreePanel()
                self._mem_tree = mt
                yield mt

    # ------------------------------------------------------------------
    # Accessors used by MainView
    # ------------------------------------------------------------------
    def get_kernel_view(self) -> Optional[KernelView]:
        return self._kernel_view

    def get_mem_tree(self) -> Optional[MemBwTreePanel]:
        return self._mem_tree

    def show_kernel_view(self) -> None:
        tabbed = self.query_one("#center-tabbed", TabbedContent)
        tabbed.active = "tab-kernel-view"

    def show_mem_tree_for_kernel(
        self,
        kernel_name: str,
        kernel_data: Optional[dict[str, Any]],
    ) -> None:
        """Activate Mem BW Tree tab and update contents."""
        if self._mem_tree:
            self._mem_tree.set_kernel(kernel_name, kernel_data)

        tabbed = self.query_one("#center-tabbed", TabbedContent)
        tabbed.active = "tab-mem-tree"

    def on_mount(self) -> None:
        self.add_class("section")
