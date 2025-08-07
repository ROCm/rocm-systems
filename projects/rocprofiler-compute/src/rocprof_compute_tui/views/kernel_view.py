"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from typing import Optional

from textual import on
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

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(id="kernel-view")
        self.dfs = {}
        self.top_kernels = {}
        self.current_selection = None

        self.config_path = config_path or (
            rocprof_compute_home
            / "rocprof_compute_tui"
            / "utils"
            / "kernel_view_config.yaml"
            if rocprof_compute_home
            else None
        )

    def compose(self):
        """
        Compose the split panel layout with two scrollable containers.
        """
        with VerticalScroll(id="top-container"):
            yield Label(
                "Open a workload directory to run analysis and view individual kernel analysis results.",
                classes="placeholder",
            )

        with VerticalScroll(id="bottom-container"):
            # empty on init
            pass

    def update_results(self, per_kernel_dfs, top_kernels) -> None:
        self.dfs = per_kernel_dfs
        self.top_kernels = top_kernels

        top_container = self.query_one("#top-container", VerticalScroll)
        top_container.remove_children()

        if not self.top_kernels:
            top_container.mount(Label("No kernels available", classes="placeholder"))
            return

        # Build and mount components
        self.new_perf_metric()
        # build header section
        keys = self.top_kernels[0].keys()
        header_text = " | ".join(f"{key:25}" for key in keys)
        top_container.mount(Label(header_text, classes="kernel-table-header"))

        # build selector section
        radio_buttons = []
        for i, kernel in enumerate(self.top_kernels):
            row_text = " | ".join(
                f"{str(kernel.get(key, 'N/A'))[:18]:25}" for key in keys
            )
            button = RadioButton(row_text, id=f"kernel-{i}")
            button.kernel_data = kernel
            radio_buttons.append(button)
        top_container.mount(RadioSet(*radio_buttons))

        # build analysis section
        self.current_selection = self.top_kernels[0]["Kernel_Name"]
        self.update_bottom_content()

    def update_view(self, message: str, log_level: str) -> None:
        if not hasattr(self, "status_label") or self.status_label is None:
            self.status_label = Label(message, classes=log_level)
            self.mount(self.status_label)
        else:
            self.status_label.update(message)
            self.status_label.set_classes(log_level)

    def new_perf_metric(self):
        new_metrics = ["VGPRs", "Grid Size", "Workgroup Size"]
        for new_metric in new_metrics:
            for i, kernel in enumerate(self.top_kernels):
                df_path = self.dfs[kernel["Kernel_Name"]]["7. Wavefront"][
                    "7.1 Wavefront Launch Stats"
                ]["df"]
                metric_avg = (
                    df_path[df_path["Metric"] == new_metric]["Avg"].iloc[0].item()
                )
                self.top_kernels[i][new_metric] = metric_avg

        """
        header_order = [
            "Dispatch_ID",
            "Kernel_Name",
            "Mean(ns)",
            "Median(ns)",
            "Sun(ns)",
            "Compute Throughput",
            "Memory Throughput",
            "VGPRs",
            "Grid Size",
            "Work Group Size",
        ]
        """

    @on(RadioSet.Changed)
    def on_radio_changed(self, event: RadioSet.Changed) -> None:
        if not event.pressed:
            return

        kernel_data = getattr(event.pressed, "kernel_data", None)
        if kernel_data and "Kernel_Name" in kernel_data:
            self.current_selection = kernel_data["Kernel_Name"]
            self.update_bottom_content()

    def update_bottom_content(self):
        bottom_container = self.query_one("#bottom-container", VerticalScroll)
        bottom_container.remove_children()

        bottom_container.mount(
            Label("Toggle kernel selection to view detailed analysis.")
        )

        if not (self.current_selection and self.current_selection in self.dfs):
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
                self.dfs[self.current_selection], self.config_path
            )
            for section in sections:
                bottom_container.mount(section)
        except Exception as e:
            bottom_container.mount(
                Label(f"Error displaying results: {str(e)}", classes="error")
            )
