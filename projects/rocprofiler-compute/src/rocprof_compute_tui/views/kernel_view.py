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
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from typing import Any, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Label, RadioButton, RadioSet

from config import rocprof_compute_home
from rocprof_compute_tui.widgets.collapsibles import build_all_sections


class KernelView(Container):
    """Center panel with analysis results split into two scrollable sections."""

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

    def __init__(self, config_path: Optional[str] = None) -> None:
        super().__init__(id="kernel-view")
        self.kernel_to_df_dict: dict[str, dict[str, Any]] = {}
        self.top_kernel_to_df_list: list[dict[str, Any]] = []
        self.current_selection: Optional[str] = None
        self.status_label: Optional[Label] = None

        self.config_path = config_path or str(
            rocprof_compute_home
            / "rocprof_compute_tui"
            / "utils"
            / "kernel_view_config.yaml"
            if rocprof_compute_home
            else None
        )

    def compose(self) -> ComposeResult:
        """
        Compose the split panel layout with two scrollable containers.
        """
        with VerticalScroll(id="top-container"):
            yield Label(
                (
                    "Open a workload directory to run analysis and view individual "
                    "kernel analysis results."
                ),
                classes="placeholder",
            )

        with VerticalScroll(id="bottom-container"):
            # empty on init
            pass

    def update_results(
        self,
        kernel_to_df_dict: dict[str, dict[str, Any]],
        top_kernel_to_df_list: list[dict[str, Any]],
    ) -> None:
        self.kernel_to_df_dict = kernel_to_df_dict
        self.top_kernel_to_df_list = top_kernel_to_df_list

        top_container = self.query_one("#top-container", VerticalScroll)
        top_container.remove_children()

        if not self.top_kernel_to_df_list:
            top_container.mount(Label("No kernels available", classes="placeholder"))
            return

        # Build and mount components
        self.new_perf_metric()
        # build header section
        keys = self.top_kernel_to_df_list[0].keys()
        header_text = " | ".join(f"{key:25}" for key in keys)
        top_container.mount(Label(header_text, classes="kernel-table-header"))

        # build selector section
        radio_buttons = []
        for i, kernel in enumerate(self.top_kernel_to_df_list):
            row_text = " | ".join(
                f"{str(kernel.get(key, 'N/A'))[:18]:25}" for key in keys
            )
            button = RadioButton(row_text, id=f"kernel-{i}")
            button.kernel_data = kernel
            radio_buttons.append(button)
        top_container.mount(RadioSet(*radio_buttons))

        # build analysis section
        self.current_selection = self.top_kernel_to_df_list[0]["Kernel_Name"]
        self.update_bottom_content()

    def update_view(self, message: str, log_level: str) -> None:
        if not hasattr(self, "status_label") or self.status_label is None:
            self.status_label = Label(message, classes=log_level)
            self.mount(self.status_label)
        else:
            self.status_label.update(message)
            self.status_label.set_classes(log_level)

    def new_perf_metric(self) -> None:
        new_metrics = ["VGPRs", "Grid Size", "Workgroup Size"]
        for new_metric in new_metrics:
            for i, kernel in enumerate(self.top_kernel_to_df_list):
                df_path = self.kernel_to_df_dict[kernel["Kernel_Name"]]["7. Wavefront"][
                    "7.1 Wavefront Launch Stats"
                ]["df"]
                metric_avg = df_path[df_path["Metric"] == new_metric]["Avg"].iloc[0]
                self.top_kernel_to_df_list[i][new_metric] = metric_avg

    @on(RadioSet.Changed)
    def on_radio_changed(self, event: RadioSet.Changed) -> None:
        if not event.pressed:
            return

        kernel_data = getattr(event.pressed, "kernel_data", None)
        if kernel_data and "Kernel_Name" in kernel_data:
            self.current_selection = kernel_data["Kernel_Name"]
            self.update_bottom_content()

    def update_bottom_content(self) -> None:
        bottom_container = self.query_one("#bottom-container", VerticalScroll)
        bottom_container.remove_children()

        bottom_container.mount(
            Label("Toggle kernel selection to view detailed analysis.")
        )

        if not (
            self.current_selection and self.current_selection in self.kernel_to_df_dict
        ):
            bottom_container.mount(
                Label(
                    f"No data available for kernel: {self.current_selection}",
                    classes="error",
                )
            )
            return

        bottom_container.mount(
            Label(f"Current kernel selection: {self.current_selection}")
        )

        try:
            sections = build_all_sections(
                self.kernel_to_df_dict[self.current_selection], self.config_path
            )
            for section in sections:
                bottom_container.mount(section)
        except Exception as e:
            bottom_container.mount(
                Label(f"Error displaying results: {str(e)}", classes="error")
            )
