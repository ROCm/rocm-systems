# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
ROCm Compute Profiler TUI - Main Application with Analysis Methods
----------------------------------------------------------------
"""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

import config
from rocprof_compute_tui.config import APP_TITLE
from rocprof_compute_tui.views.main_view import MainView
from utils.specs import generate_machine_specs
from utils.utils_common import get_version


class RocprofTUIApp(App):
    """Main application for the performance analysis tool."""

    VERSION = get_version(config.rocprof_compute_home)["version"]
    TITLE = f"{APP_TITLE} v{VERSION}"
    SUB_TITLE = "Workload Analysis Tool"

    CSS_PATH = str(Path(__file__).parent / "assets" / "style.css")
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="r", action="refresh", description="Refresh"),
        # TODO
        # Binding(key="a", action="analyze", description="Analyze"),
    ]

    def __init__(
        self,
        args: argparse.Namespace,
        supported_archs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.main_view = MainView()
        self.recent_dirs = self._load_recent_dirs()

        # Analysis attributes
        self.args = args
        self.supported_archs = supported_archs or {}
        self.soc: dict[str, Any] = {}
        self.mspec: Optional[Any] = None
        self.mouse = True

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.main_view
        yield Footer()

    def action_refresh(self) -> None:
        self.main_view.refresh_view()
        self.notify("View refreshed", severity="information")

    def load_soc_specs(self, sysinfo: Optional[dict] = None) -> None:
        try:
            self.mspec = generate_machine_specs(self.args, sysinfo)
            arch = self.mspec.gpu_arch

            soc_module = importlib.import_module(f"rocprof_compute_soc.soc_{arch}")
            soc_class = getattr(soc_module, f"{arch}_soc")
            self.soc[arch] = soc_class(self.args, self.mspec)

            self.notify(f"Loaded system specs for {arch}", severity="information")

        except Exception as e:
            self.notify(f"Failed to load system specs: {e}", severity="error")
            raise

    # -------------------------------------------------------------------------
    # Recent directories management
    # -------------------------------------------------------------------------
    def _load_recent_dirs(self) -> list[str]:
        recent_file = Path.home() / ".textual_browser_recent.json"
        if recent_file.exists():
            with open(recent_file) as f:
                return json.load(f)
        return []

    def _save_recent_dirs(self) -> None:
        recent_file = Path.home() / ".textual_browser_recent.json"
        with open(recent_file, "w") as f:
            json.dump(self.recent_dirs, f, indent=2)

    def add_recent_dir(self, directory: str) -> None:
        directory = str(Path(directory).resolve())

        # Remove if exists, add to front, keep max 5
        if directory in self.recent_dirs:
            self.recent_dirs.remove(directory)
        self.recent_dirs.insert(0, directory)
        self.recent_dirs = self.recent_dirs[:5]
        self._save_recent_dirs()

    def on_recent_selected(self, selected_dir: Optional[str]) -> None:
        if not selected_dir:
            self.notify("Directory selection cancelled", severity="information")
            return

        if Path(selected_dir) != self.main_view.selected_path:
            self.main_view.selected_path = Path(selected_dir)

        self.notify(f"Selected: {selected_dir}", severity="information")
        self.main_view.run_analysis()


def run_tui(
    args: argparse.Namespace, supported_archs: Optional[dict[str, Any]] = None
) -> None:
    """Run the TUI application."""
    app = RocprofTUIApp(args, supported_archs)
    app.run()
