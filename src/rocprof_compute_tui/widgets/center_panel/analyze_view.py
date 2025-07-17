"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from importlib import resources
from typing import Any, Dict, Optional

from textual.containers import ScrollableContainer
from textual.widgets import Label

from rocprof_compute_tui.widgets.collapsibles import build_all_sections


class AnalyzeView(ScrollableContainer):
    """Center panel with analysis results."""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(id="analyze-view")
        self.dfs = {}

        if config_path is None:
            config_path = (
                resources.files("rocprof_compute_tui.utils") / "analyze_config.yaml"
            )

        self.config_path = str(config_path)

    def compose(self):
        """
        Compose the initial center panel state.
        """
        yield Label(
            "Open a workload directory to run analysis and view results",
            classes="placeholder",
        )

    def update_results(self, dfs: Dict[str, Any]) -> None:
        """
        Update the center panel with analysis results.
        """
        self.dfs = dfs
        self.remove_children()

        try:
            sections = build_all_sections(self.dfs, self.config_path)

            # Mount all sections
            for section in sections:
                self.mount(section)

        except Exception as e:
            self.mount(Label(f"Error displaying results: {str(e)}", classes="error"))

    def update_view(self, message: str, log_level: str) -> None:
        """
        Update the view with a status message.
        """
        self.remove_children()
        try:
            self.mount(Label(f"{message}", classes=log_level))
        except Exception as e:
            self.mount(Label(f"Error displaying results: {str(e)}", classes="error"))

    def reload_config(self, config_path: str = None) -> None:
        """
        Reload the configuration and update the view.
        """
        if config_path:
            self.config_path = config_path

        if self.dfs:
            self.update_results(self.dfs)
