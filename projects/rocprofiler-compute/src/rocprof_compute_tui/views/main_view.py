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
Main View Module
---------------
Contains the main view layout and organization for the application.
"""

import threading
import traceback
from pathlib import Path
from typing import Any, Optional

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable

from rocprof_compute_tui.analysis_tui import tui_analysis
from rocprof_compute_tui.utils.tui_utils import Logger, LogLevel
from rocprof_compute_tui.widgets.center_panel.center_area import CenterPanel
from rocprof_compute_tui.widgets.menu_bar.menu_bar import MenuBar
from rocprof_compute_tui.widgets.right_panel.right import RightPanel
from rocprof_compute_tui.widgets.tabs.tabs_area import TabsArea
from utils import file_io


class MainView(Horizontal):
    """Main view layout for the application."""

    selected_path: reactive[Optional[Path]] = reactive(None)
    kernel_to_df_dict: reactive[dict[str, dict[str, Any]]] = reactive({})
    top_kernel_to_df_list: reactive[list[dict[str, Any]]] = reactive([])

    def __init__(self) -> None:
        super().__init__(id="main-container")
        self.logger = Logger()
        self.logger.info("MainView initialized", update_ui=False)

    def flush(self) -> None:
        """Required for stdout compatibility."""
        pass

    def compose(self) -> ComposeResult:
        self.logger.info("Composing main view layout", update_ui=False)
        yield MenuBar()

        # Center Container - Holds both analysis results and output tabs
        with Horizontal(id="center-container"):
            with Vertical(id="activity-container"):
                # Center Panel - Analysis results display
                yield CenterPanel()

                # Bottom Panel - Output, terminal, and metric description
                tabs = TabsArea()
                yield tabs

                # Store references to text areas
                self.metric_description = tabs.description_area
                self.output = tabs.output_area

                self.logger.set_output_area(self.output)
                self.logger.info("Main view layout composed")

            # Right Panel - Additional tools/features
            yield RightPanel()

    @on(DataTable.CellSelected)
    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        table = event.data_table
        row_idx = event.coordinate.row

        visible_data = table.get_row_at(row_idx)
        description = self._get_row_description(table, row_idx)

        if self.metric_description is not None:
            self.metric_description.text = (
                f"Selected Metric ID: {visible_data[0]}\n"
                f"Selected Metric: {visible_data[1]}\n"
                f"Description: {description}"
            )

    def _get_row_description(self, table: DataTable, row_idx: int) -> str:
        """Get description for a table row with safe attribute access."""
        try:
            if hasattr(table, "_df") and table._df is not None:
                return str(table._df.iloc[row_idx].get("Description", "No description"))
        except (IndexError, AttributeError, KeyError):
            pass
        return "N/A"

    @work(thread=True)
    def run_analysis(self) -> None:
        """
        Run analysis in a background worker thread.
        All UI updates are marshalled back onto the main thread.
        """

        # Capture selected path at the beginning to avoid races
        selected = self.selected_path

        # -----------------------------
        # 1. No directory selected
        # -----------------------------
        if not selected:

            def ui_no_directory() -> None:
                self.app.notify(
                    "No directory selected for analysis", severity="warning"
                )
                self._update_kernel_view(
                    "No directory selected for analysis", LogLevel.ERROR
                )

            self.app.call_from_thread(ui_no_directory)
            return

        # Reset analysis results on the UI thread before starting
        def ui_reset_before_analysis() -> None:
            self.kernel_to_df_dict = {}
            self.top_kernel_to_df_list = []
            self.logger.info(f"Starting analysis on: {selected}")
            self.logger.info("Loading...")
            self.app.notify(f"Running analysis on: {selected}", severity="information")
            self._update_kernel_view(
                f"Running analysis on: {selected}", LogLevel.SUCCESS
            )

        self.app.call_from_thread(ui_reset_before_analysis)

        try:
            # ------------------------------------
            # 2. Initialize analyzer
            # ------------------------------------
            analyzer = tui_analysis(
                self.app.args, self.app.supported_archs, str(selected)
            )
            analyzer.sanitize()

            # ------------------------------------
            # 3. Load sysinfo
            # ------------------------------------
            sysinfo_path = selected / "sysinfo.csv"
            if not sysinfo_path.exists():
                # Let the UI thread handle the error and reset state
                error = FileNotFoundError(f"sysinfo.csv not found at {sysinfo_path}")
                tb = traceback.format_exc()

                def ui_missing_sysinfo() -> None:
                    error_msg = f"Analysis failed: {error}"
                    self.logger.error(f"{error_msg}\n{tb}")
                    self.app.notify(
                        f"sysinfo.csv not found at: {sysinfo_path}", severity="error"
                    )
                    self.kernel_to_df_dict = {}
                    self.top_kernel_to_df_list = []
                    self._update_kernel_view(error_msg, LogLevel.ERROR)

                self.app.call_from_thread(ui_missing_sysinfo)
                return

            sys_info = file_io.load_sys_info(str(sysinfo_path)).iloc[0].to_dict()
            self.app.load_soc_specs(sys_info)
            analyzer.set_soc(self.app.soc)

            # ------------------------------------
            # 4. Run preprocessing
            # ------------------------------------
            analyzer.pre_processing()

            def ui_after_preprocessing() -> None:
                self.app.notify("Profiling data loaded", severity="information")

            self.app.call_from_thread(ui_after_preprocessing)

            # ------------------------------------
            # 5. Kernel analysis (heavy work)
            # ------------------------------------
            kernel_to_df_dict = analyzer.run_kernel_analysis()
            top_kernel_to_df_list = analyzer.run_top_kernel()

            # ------------------------------------
            # 6. Pass results to UI thread
            # ------------------------------------
            self.app.call_from_thread(
                self._analysis_success,
                kernel_to_df_dict,
                top_kernel_to_df_list,
            )

        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            error_msg = f"Analysis failed: {str(e)}"

            def ui_error_handler() -> None:
                # Clear in-memory results
                self.kernel_to_df_dict = {}
                self.top_kernel_to_df_list = []

                # Log, notify, and update kernel view safely
                self.logger.error(f"{error_msg}\n{tb}")
                self.app.notify(error_msg, severity="error")
                self._update_kernel_view(error_msg, LogLevel.ERROR)

            self.app.call_from_thread(ui_error_handler)

    def _analysis_success(
        self,
        kernel_to_df_dict: dict[str, dict[str, Any]],
        top_kernel_to_df_list: list[dict[str, Any]],
    ) -> None:
        self.kernel_to_df_dict = kernel_to_df_dict or {}
        self.top_kernel_to_df_list = top_kernel_to_df_list or []

        if not self.kernel_to_df_dict or not self.top_kernel_to_df_list:
            self.app.notify(
                "Analysis completed but not all data was produced",
                severity="warning",
            )
            self._update_kernel_view(
                "Analysis completed but not all data was returned", LogLevel.WARNING
            )
        else:
            self.refresh_results()
            self.logger.info("Kernel Analysis completed successfully")
            self.app.notify("Kernel analysis completed", severity="information")

    def _update_kernel_view(self, message: str, log_level: LogLevel) -> None:
        app = self.app

        # detect thread
        in_ui_thread = threading.get_ident() == app._thread_id

        def apply() -> None:
            view = self.query_one("#kernel-view")
            if view:
                view.update_view(message, log_level)

        if in_ui_thread:
            apply()
        else:
            app.call_from_thread(apply)

    def refresh_results(self) -> None:
        kernel_view = self.query_one("#kernel-view")
        if kernel_view:
            kernel_view.update_results(
                self.kernel_to_df_dict, self.top_kernel_to_df_list
            )
            self.logger.success("Results displayed successfully.")
        else:
            self.logger.error("Kernel view not found or no data available")

    def refresh_view(self) -> None:
        if self.kernel_to_df_dict and self.top_kernel_to_df_list:
            self.refresh_results()
        else:
            self.logger.warning("No data available for refresh")
