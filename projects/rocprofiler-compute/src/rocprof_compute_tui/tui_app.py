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
ROCm Compute Profiler TUI - Main Application with Analysis Methods
----------------------------------------------------------------
"""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Footer, Header
from textual_fspicker import SelectDirectory

import config
from rocprof_compute_tui.config import APP_TITLE
from rocprof_compute_tui.views.main_view import MainView
from rocprof_compute_tui.widgets.menu_bar.menu_bar import DropdownMenu
from utils.specs import generate_machine_specs
from utils.utils import get_version


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

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.main_view
        yield Footer()

    def action_refresh(self) -> None:
        self.main_view.refresh_view()

    def load_soc_specs(self, sysinfo: Optional[dict] = None) -> None:
        self.mspec = generate_machine_specs(self.args, sysinfo)
        arch = self.mspec.gpu_arch
        soc_module = importlib.import_module(f"rocprof_compute_soc.soc_{arch}")
        soc_class = getattr(soc_module, f"{arch}_soc")
        self.soc[arch] = soc_class(self.args, self.mspec)

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

    @on(Button.Pressed, "#menu-open-workload")
    @work
    async def pick_directory(self) -> None:
        if opened := await self.push_screen_wait(SelectDirectory()):
            self.add_recent_dir(str(opened))
            self.main_view.selected_path = opened
            self.query_one("#file-dropdown", DropdownMenu).add_class("hidden")
            self.main_view.run_analysis()


def run_tui(
    args: argparse.Namespace, supported_archs: Optional[dict[str, Any]] = None
) -> None:
    """Run the TUI application."""
    app = RocprofTUIApp(args, supported_archs)
    app.run()
