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

"""
KernelView
----------
Center panel showing kernel analysis results.
"""

from __future__ import annotations

from typing import Any, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import (
    Container,
    Horizontal,
    ScrollableContainer,
)
from textual.message import Message
from textual.widgets import Label, RadioButton, RadioSet

from config import rocprof_compute_home
from rocprof_compute_tui.widgets.collapsibles import build_all_sections
from rocprof_compute_tui.widgets.instant_button import InstantButton


class OpenMemTree(Message):
    """Message emitted when the MemTree button is pressed."""

    def __init__(self, sender: KernelView, kernel_name: str) -> None:
        super().__init__()
        self.sender = sender
        self.kernel_name = kernel_name


class KernelView(Container):
    """Center-panel kernel analysis UI."""

    DEFAULT_CSS = """
    KernelView {
        layout: vertical;
    }

    #top-container {
        height: 1fr;
        border: none;
        margin-top: 1;
    }

    #bottom-container {
        height: 4fr;
        border: none;
        margin-top: 2;
    }

    .kernel-table-header {
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 1;
        offset: 5 0;
        margin-top: 1;
    }

    .kernel-row {
        padding: 0 1;
        border-bottom: solid $border;
    }

    RadioSet {
        border: solid $border;
    }
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
    ) -> None:
        super().__init__(id="kernel-view")

        # Data injected by analysis
        self.kernel_to_df_dict: dict[str, dict[str, Any]] = {}
        self.top_kernel_to_df_list: list[dict[str, Any]] = []
        self.current_selection: Optional[str] = None

        # Status label
        self.status_label: Optional[Label] = None

        # Config path
        if config_path is not None:
            self.config_path = config_path
        elif rocprof_compute_home:
            self.config_path = str(
                rocprof_compute_home
                / "rocprof_compute_tui"
                / "utils"
                / "kernel_view_config.yaml"
            )
        else:
            self.config_path = None

    # ------------------------------------------------------------------
    # Compose: ONLY static layout
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="top-container")
        yield ScrollableContainer(id="bottom-container")

    # ------------------------------------------------------------------
    # Status message (short-lived)
    # ------------------------------------------------------------------
    def update_view(self, message: str, log_level: str) -> None:
        if self.status_label is None:
            self.status_label = Label(message, classes=log_level)
            self.mount(self.status_label)
        else:
            self.status_label.update(message)
            self.status_label.set_classes(log_level)

    # ------------------------------------------------------------------
    # Update results after analysis completes
    # ------------------------------------------------------------------
    def update_results(
        self,
        kernel_to_df_dict: dict[str, dict[str, Any]],
        top_kernel_to_df_list: list[dict[str, Any]],
    ) -> None:
        self.kernel_to_df_dict = kernel_to_df_dict
        self.top_kernel_to_df_list = top_kernel_to_df_list

        top_container = self.query_one("#top-container", ScrollableContainer)
        top_container.remove_children()

        if not self.top_kernel_to_df_list:
            top_container.mount(Label("No kernels available", classes="placeholder"))
            return

        # --------------------------------------------------------------
        # Header row: summary + Mem BW Tree button
        # --------------------------------------------------------------
        summary = f"{len(self.top_kernel_to_df_list)} kernels profiled"

        header_row = Horizontal(
            Label(summary, classes="kernel-table-header"),
            InstantButton("Mem BW Tree", id="btn-open-mem-tree"),
            id="kernel-header-row",
        )
        top_container.mount(header_row)

        # --------------------------------------------------------------
        # Table header
        # --------------------------------------------------------------
        keys = list(self.top_kernel_to_df_list[0].keys())
        header_text = " | ".join(f"{key:20}" for key in keys)
        top_container.mount(Label(header_text, classes="kernel-table-header"))

        # --------------------------------------------------------------
        # Kernel selector radios
        # --------------------------------------------------------------
        radio_buttons: list[RadioButton] = []
        for idx, kernel in enumerate(self.top_kernel_to_df_list):
            row_text = " | ".join(
                f"{str(kernel.get(key, 'N/A'))[:18]:20}" for key in keys
            )
            rb = RadioButton(row_text, id=f"kernel-{idx}")
            rb.kernel_data = kernel  # attach raw data
            radio_buttons.append(rb)

        if radio_buttons:
            radio_set = RadioSet(*radio_buttons)
            top_container.mount(radio_set)
            first_kernel = radio_buttons[0].kernel_data
            self.current_selection = first_kernel.get("Kernel_Name")
        else:
            self.current_selection = None

        self.update_bottom_content()

    # ------------------------------------------------------------------
    # Radio button selection
    # ------------------------------------------------------------------
    @on(RadioSet.Changed)
    def on_radio_changed(self, event: RadioSet.Changed) -> None:
        if not event.pressed:
            return

        kernel_data = getattr(event.pressed, "kernel_data", None)
        if not kernel_data:
            return

        self.current_selection = kernel_data.get("Kernel_Name")
        self.update_bottom_content()

    # ------------------------------------------------------------------
    # Mem BW Tree button
    # ------------------------------------------------------------------
    def on_instant_button_instant_pressed(
        self,
        event: InstantButton.InstantPressed,
    ) -> None:
        if event.button.id != "btn-open-mem-tree":
            return

        if not self.current_selection:
            try:
                self.app.notify("No kernel selected", severity="warning")
            except Exception:
                pass
            return

        self.post_message(OpenMemTree(self, self.current_selection))

    # ------------------------------------------------------------------
    # Bottom collapsible view
    # ------------------------------------------------------------------
    def update_bottom_content(self) -> None:
        bottom = self.query_one("#bottom-container", ScrollableContainer)
        bottom.remove_children()

        bottom.mount(Label("Toggle kernel selection to view analysis."))

        if (
            not self.current_selection
            or self.current_selection not in self.kernel_to_df_dict
        ):
            bottom.mount(
                Label(
                    f"No data for kernel selection: {self.current_selection}",
                    classes="error",
                )
            )
            return

        bottom.mount(Label(f"Current kernel selection: {self.current_selection}"))

        try:
            sections = build_all_sections(
                self.kernel_to_df_dict[self.current_selection],
                self.config_path,
            )
            for section in sections:
                bottom.mount(section)
        except Exception as e:
            bottom.mount(Label(f"Error displaying results: {e}", classes="error"))
