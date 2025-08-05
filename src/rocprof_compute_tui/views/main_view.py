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

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable

from rocprof_compute_tui.analysis_tui import tui_analysis
from rocprof_compute_tui.config import DEFAULT_START_PATH
from rocprof_compute_tui.utils.tui_utils import Logger, LogLevel
from rocprof_compute_tui.widgets.center_panel.center_area import CenterPanel
from rocprof_compute_tui.widgets.menu_bar.menu_bar import MenuBar
from rocprof_compute_tui.widgets.right_panel.right import RightPanel
from rocprof_compute_tui.widgets.tabs.tabs_area import TabsArea
from utils import file_io


class MainView(Horizontal):
    """Main view layout for the application."""

    selected_path = reactive(None)
    per_kernel_dfs = reactive({})
    top_kernels = reactive([])

    def __init__(self):
        super().__init__(id="main-container")
        self.start_path = (
            # NOTE: is cwd the best choice?
            Path.cwd() if DEFAULT_START_PATH is None else Path(DEFAULT_START_PATH))

        self.logger = Logger()
        self.logger.info("MainView initialized", update_ui=False)

    def flush(self):
        """Required for stdout compatibility."""
        pass

    def compose(self) -> ComposeResult:
        self.logger.info("Composing main view layout", update_ui=False)
        yield MenuBar()

        # Center Container - Holds both analysis results and output tabs
        with Horizontal(id="center-container"):
            with Vertical(id="activity-container"):
                # Center Panel - Analysis results display
                center_panel = CenterPanel()
                yield center_panel
                self.center = center_panel

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

        self.logger.info(f"Cell selected at row {row_idx}")

        try:
            row_data = table.get_row_at(row_idx)
            content = f"Selected Metric ID: {row_data[0]}\n"
            content += f"Selected Metric: {row_data[1]}\n"
            # content += f"Metric Description:\n\t{row_data[-1]}"

            self.metric_description.text = content
            self.logger.info(f"Row {row_idx} data displayed in metric_description")

        except Exception as e:
            error_msg = f"Error displaying row {row_idx}: {str(e)}"
            table.add_column("Error")
            table.add_row(str(e))
            self.metric_description.text = error_msg
            self.logger.error(error_msg)

    @work(thread=True)
    def run_analysis(self) -> None:
        self.per_kernel_dfs = {}
        self.top_kernels = []

        if not self.selected_path:
            error_msg = "No directory selected for analysis"
            self._update_view(error_msg, LogLevel.ERROR)
            self.logger.error(error_msg)
            return

        try:
            self.logger.info(f"Starting analysis on: {self.selected_path}")
            self._update_view(f"Running analysis on: {self.selected_path}",
                              LogLevel.SUCCESS)

            # Step 1: Create analyzer
            try:
                self.logger.info("Step 1: Creating analyzer")
                self.logger.info(f"Step 1: args {self.app.args}")
                self.logger.info(f"Step 1: arch {self.app.supported_archs}")
                self.logger.info("Step 1: Creating analyzer")
                analyzer = tui_analysis(self.app.args, self.app.supported_archs,
                                        self.selected_path)
                self.logger.info("Step 1: Analyzer created successfully")
            except Exception as e:
                self.logger.error(f"Step 1 failed - Error creating analyzer: {str(e)}")
                raise

            # Step 2: Sanitize analyzer
            try:
                self.logger.info("Step 2: Sanitizing analyzer")
                analyzer.sanitize()
                self.logger.info("Step 2: Analyzer sanitized successfully")
            except Exception as e:
                self.logger.error(
                    f"Step 2 failed - Error sanitizing analyzer: {str(e)}")
                raise

            # Step 3: Load sys_info
            try:
                self.logger.info("Step 3: Loading sys_info")
                sysinfo_path = Path(self.selected_path).joinpath("sysinfo.csv")
                self.logger.info(f"Step 3: sysinfo_path = {sysinfo_path}")

                if not sysinfo_path.exists():
                    raise FileNotFoundError(f"sysinfo.csv not found at {sysinfo_path}")

                sys_info_df = file_io.load_sys_info(sysinfo_path)
                self.logger.info(f"Step 3: sys_info_df type = {type(sys_info_df)}")
                self.logger.info(
                    f"Step 3: sys_info_df shape = {sys_info_df.shape if hasattr(sys_info_df, 'shape') else 'No shape attribute'}"
                )

            except Exception as e:
                self.logger.error(f"Step 3 failed - Error loading sys_info: {str(e)}")
                raise

            # Step 4: Convert sys_info to dict
            try:
                self.logger.info("Step 4: Converting sys_info to dict")

                # Check if it's actually a DataFrame
                if hasattr(sys_info_df, "iloc"):
                    sys_info = sys_info_df.iloc[0].to_dict()
                elif hasattr(sys_info_df, "to_dict"):
                    # If it's already a Series
                    sys_info = sys_info_df.to_dict()
                elif isinstance(sys_info_df, dict):
                    # If it's already a dict
                    sys_info = sys_info_df
                else:
                    raise TypeError(
                        f"Unexpected type for sys_info: {type(sys_info_df)}")

                self.logger.info(f"Step 4: sys_info converted = {sys_info}")

            except Exception as e:
                self.logger.error(
                    f"Step 4 failed - Error converting sys_info: {str(e)}")
                raise

            # Step 5: Load SoC specs
            try:
                self.logger.info("Step 5: Loading SoC specs")
                self.app.load_soc_specs(sys_info)
                self.logger.info(f"Step 5: SoC loaded = {self.app.soc}")
            except Exception as e:
                self.logger.error(f"Step 5 failed - Error loading SoC specs: {str(e)}")
                raise

            # Step 6: Set SoC in analyzer
            try:
                self.logger.info("Step 6: Setting SoC in analyzer")
                analyzer.set_soc(self.app.soc)
                self.logger.info("Step 6: SoC set successfully")
            except Exception as e:
                self.logger.error(f"Step 6 failed - Error setting SoC: {str(e)}")
                raise

            # Step 7: Pre-processing
            try:
                self.logger.info("Step 7: Running pre-processing")
                analyzer.pre_processing()
                self.logger.info("Step 7: Pre-processing completed")
            except Exception as e:
                self.logger.error(f"Step 7 failed - Error in pre-processing: {str(e)}")
                raise
            # Step 8: Run analysis
            try:
                self.logger.info("Step 8: Running analysis")
                self.per_kernel_dfs = analyzer.run_kernel_analysis()
                self.top_kernels = analyzer.run_top_kernel()

                # TODO: add per kernel Roofline support when available

                if not self.per_kernel_dfs or not self.top_kernels:
                    warning_msg = "Step 8: Per Kernel Analysis completed but not all data was returned"
                    self._update_view(warning_msg, LogLevel.WARNING)
                    self.logger.warning(warning_msg)
                else:
                    self.app.call_from_thread(self.refresh_results)
                    self.logger.info("Step 8: Kernel Analysis completed successfully")
                    # self.logger.info(f"{self.per_kernel_dfs}")
            except Exception as e:
                self.logger.error(f"Step 8 failed - Error running analysis: {str(e)}")
                raise

        except Exception as e:
            import traceback

            error_msg = f"Unexpected error during analysis: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self._update_view(error_msg, LogLevel.ERROR)

    def _update_view(self, message: str, log_level: LogLevel) -> None:
        try:
            self.app.call_from_thread(self._safe_update_view, message, log_level)
        except Exception as e:
            self.logger.error(f"View update scheduling error: {str(e)}")

    def _safe_update_view(self, message: str, log_level: LogLevel) -> None:
        try:
            kernel_view = self.query_one("#kernel-view")
            if kernel_view:
                kernel_view.update_view(message, log_level)
            else:
                self.logger.warning("Analysis view not found when updating log")
        except Exception as e:
            self.logger.error(f"Log update error: {str(e)}")

    def refresh_results(self) -> None:
        try:
            self.logger.info("Refreshing kernel results")
            kernel_view = self.query_one("#kernel-view")
            if not kernel_view:
                self.logger.error("Kernel view not found")
                return

            if (not hasattr(self, "per_kernel_dfs") or self.per_kernel_dfs is None
                    or not hasattr(self, "top_kernels") or self.top_kernels is None):
                self.logger.error("No kernel analysis data available to display")
                return

            kernel_view.update_results(self.per_kernel_dfs, self.top_kernels)
            self.logger.success("Results displayed successfully.")
        except Exception as e:
            self.logger.error(f"Error refreshing results: {str(e)}")

    def refresh_view(self) -> None:
        self.logger.info("Refreshing view...")
        if self.top_kernels:
            self.refresh_results()
        else:
            self.logger.warning("No data available for refresh")
