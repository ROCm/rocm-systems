# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import TabPane

from rocprof_compute_tui.views.kernel_view import KernelView
from rocprof_compute_tui.widgets.tabbed_content import TabsTabbedContent


class CenterPanel(Vertical):
    """
    The response area.
    """

    COMPONENT_CLASSES = {
        "border-title-status",
    }

    def __init__(self) -> None:
        super().__init__()

        self.default_tab = "center-analyze"
        self.kernel_view = KernelView()

    def compose(self) -> ComposeResult:
        with TabsTabbedContent(initial="tab-kernel"):
            with TabPane("Basic View", id="tab-kernel"):
                yield self.kernel_view

    def on_mount(self) -> None:
        self.add_class("section")
