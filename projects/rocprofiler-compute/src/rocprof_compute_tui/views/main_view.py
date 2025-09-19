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
        self.kernel_to_df_dict = {}
        self.top_kernel_to_df_list = []

        if not self.selected_path:
            self._update_kernel_view(
                "No directory selected for analysis", LogLevel.ERROR
            )
            return

        try:
            self.logger.info(f"Starting analysis on: {self.selected_path}")
            self.logger.info("Loading...")

            self._update_kernel_view(
                f"Running analysis on: {self.selected_path}", LogLevel.SUCCESS
            )

            # 1. Create and TUI analyzer
            analyzer = tui_analysis(
                self.app.args, self.app.supported_archs, str(self.selected_path)
            )
            analyzer.sanitize()

            # 2. Load and process system info and Configure SoC
            sysinfo_path = self.selected_path / "sysinfo.csv"
            if not sysinfo_path.exists():
                raise FileNotFoundError(f"sysinfo.csv not found at {sysinfo_path}")

            sys_info = file_io.load_sys_info(str(sysinfo_path)).iloc[0].to_dict()
            self.app.load_soc_specs(sys_info)
            analyzer.set_soc(self.app.soc)

            # 3. run analysis
            analyzer.pre_processing()
            self.kernel_to_df_dict = analyzer.run_kernel_analysis()
            self.top_kernel_to_df_list = analyzer.run_top_kernel()

            if not self.kernel_to_df_dict or not self.top_kernel_to_df_list:
                self._update_kernel_view(
                    "Analysis completed but not all data was returned", LogLevel.WARNING
                )
            else:
                self.app.call_from_thread(self.refresh_results)
                self.logger.info("Kernel Analysis completed successfully")

        except Exception as e:
            import traceback

            error_msg = f"Analysis failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self._update_kernel_view(error_msg, LogLevel.ERROR)

    def _update_kernel_view(self, message: str, log_level: LogLevel) -> None:
        self.app.call_from_thread(
            lambda: self.query_one("#kernel-view").update_view(message, log_level)
        )

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
