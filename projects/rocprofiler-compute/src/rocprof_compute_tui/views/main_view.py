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
        self.start_path = Path(DEFAULT_START_PATH) if DEFAULT_START_PATH else Path.cwd()
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
        try:
            row_data = event.data_table.get_row_at(event.coordinate.row)
            self.metric_description.text = (
                f"Selected Metric ID: {row_data[0]}\nSelected Metric: {row_data[1]}\n"
            )
            self.logger.info(f"Row {event.coordinate.row} data displayed")
        except Exception as e:
            error_msg = f"Error displaying row {event.coordinate.row}: {str(e)}"
            self.metric_description.text = error_msg
            self.logger.error(error_msg)

    @work(thread=True)
    def run_analysis(self) -> None:
        self.per_kernel_dfs = {}
        self.top_kernels = []

        if not self.selected_path:
            try:
                self.app.call_from_thread(
                    lambda: self.query_one("#kernel-view").update_view(
                        "No directory selected for analysis", LogLevel.ERROR
                    )
                )
            except:
                pass
            return

        try:
            self.logger.info(f"Starting analysis on: {self.selected_path}")
            try:
                self.app.call_from_thread(
                    lambda: self.query_one("#kernel-view").update_view(
                        f"Running analysis on: {self.selected_path}", LogLevel.SUCCESS
                    )
                )
            except:
                pass

            # 1. Create and TUI analyzer
            analyzer = tui_analysis(
                self.app.args, self.app.supported_archs, self.selected_path
            )
            analyzer.sanitize()

            # 2. Load and process system info
            sysinfo_path = Path(self.selected_path) / "sysinfo.csv"
            if not sysinfo_path.exists():
                raise FileNotFoundError(f"sysinfo.csv not found at {sysinfo_path}")

            sys_info = file_io.load_sys_info(sysinfo_path).iloc[0].to_dict()

            # 3. Configure SoC and run analysis
            self.app.load_soc_specs(sys_info)
            analyzer.set_soc(self.app.soc)
            analyzer.pre_processing()
            self.per_kernel_dfs = analyzer.run_kernel_analysis()
            self.top_kernels = analyzer.run_top_kernel()

            if not self.per_kernel_dfs or not self.top_kernels:
                try:
                    self.app.call_from_thread(
                        lambda: self.query_one("#kernel-view").update_view(
                            "Analysis completed but not all data was returned",
                            LogLevel.WARNING,
                        )
                    )
                except:
                    pass
            else:
                self.app.call_from_thread(self.refresh_results)
                self.logger.info("Kernel Analysis completed successfully")

        except Exception as e:
            import traceback

            error_msg = f"Analysis failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            try:
                self.app.call_from_thread(
                    lambda: self.query_one("#kernel-view").update_view(
                        error_msg, LogLevel.ERROR
                    )
                )
            except:
                pass

    def refresh_results(self) -> None:
        try:
            kernel_view = self.query_one("#kernel-view")
            if kernel_view and self.per_kernel_dfs and self.top_kernels:
                kernel_view.update_results(self.per_kernel_dfs, self.top_kernels)
                self.logger.success("Results displayed successfully.")
            else:
                self.logger.error("Kernel view not found or no data available")
        except Exception as e:
            self.logger.error(f"Error refreshing results: {str(e)}")

    def refresh_view(self) -> None:
        if self.top_kernels:
            self.refresh_results()
        else:
            self.logger.warning("No data available for refresh")
