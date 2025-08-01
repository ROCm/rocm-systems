##############################################################################bl
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

"""
ROCm Compute Profiler TUI - Main Application with Analysis Methods
----------------------------------------------------------------
"""

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Footer, Header
from textual_fspicker import SelectDirectory

import config
from rocprof_compute_tui.config import APP_TITLE
from rocprof_compute_tui.views.main_view import MainView
from rocprof_compute_tui.widgets.menu_bar.menu_bar import DropdownMenu
from utils.specs import MachineSpecs, generate_machine_specs
from utils.utils import get_version


class RocprofTUIApp(App):
    """Main application for the performance analysis tool."""

    VERSION = get_version(config.rocprof_compute_home)["version"]
    TITLE = f"{APP_TITLE} v{VERSION}"
    SUB_TITLE = "Workload Analysis Tool"

    CSS_PATH = "assets/style.css"
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="r", action="refresh", description="Refresh"),
        # TODO
        # Binding(key="a", action="analyze", description="Analyze"),
    ]

    def __init__(
        self, args: Optional[Any] = None, supported_archs: Optional[Dict] = None
    ) -> None:
        """
        Initialize the application.
        """
        super().__init__()
        self.main_view = MainView()

        self.recent_file = Path.home() / ".textual_browser_recent.json"
        self.recent_dirs: List[str] = []
        self.current_path = ""
        self.load_recent_directories()

        # Initialize analysis-related attributes
        self.args = args
        self.supported_archs = supported_archs or {}
        self.soc: Dict = {}
        self.mspec: Optional[MachineSpecs] = None

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        yield self.main_view
        yield Footer()

    def action_refresh(self) -> None:
        """Refresh the view."""
        try:
            self.main_view.refresh_view()
        except Exception as e:
            self.notify(f"Refresh failed: {str(e)}", severity="error")

    def load_soc_specs(self, sysinfo: dict = None) -> None:
        """
        Load OmniSoC instance for analysis.
        """
        self.mspec = generate_machine_specs(self.args, sysinfo)

        if self.args and self.args.specs:
            print(self.mspec)
            return

        arch = self.mspec.gpu_arch

        # Dynamically import and instantiate the SoC class
        soc_module = importlib.import_module("rocprof_compute_soc.soc_" + arch)
        soc_class = getattr(soc_module, arch + "_soc")
        self.soc[arch] = soc_class(self.args, self.mspec)

    def get_soc(self) -> Dict:
        """Get the SoC dictionary."""
        return self.soc

    def get_mspec(self) -> Optional[MachineSpecs]:
        """Get the machine specifications."""
        return self.mspec

    def load_recent_directories(self) -> None:
        """Load recent directories from file."""
        try:
            if self.recent_file.exists():
                with open(self.recent_file, "r") as f:
                    self.recent_dirs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.recent_dirs = []

    def save_recent_directories(self) -> None:
        """Save recent directories to file."""
        try:
            with open(self.recent_file, "w") as f:
                json.dump(self.recent_dirs, f, indent=2)
        except Exception as e:
            self.notify(f"Failed to save recent directories: {e}", severity="error")

    def add_to_recent(self, directory: str) -> None:
        """Add directory to recent list (FIFO, max 5 items)."""
        directory = os.path.abspath(directory)

        # Remove if already exists
        if directory in self.recent_dirs:
            self.recent_dirs.remove(directory)

        # Add to front
        # TODO: should we check to if workload dir can be successfully loaded?
        self.recent_dirs.insert(0, directory)

        # Keep only last 5
        self.recent_dirs = self.recent_dirs[:5]

        # Save to file
        self.save_recent_directories()

    def on_recent_selected(self, selected_dir: str) -> None:
        if selected_dir:
            self.main_view.selected_path = selected_dir
            self.main_view.run_analysis()

    @on(Button.Pressed, "#menu-open-workload")
    @work
    async def pick_a_directory(self) -> None:
        if opened := await self.push_screen_wait(SelectDirectory()):
            self.add_to_recent(str(opened))
            self.main_view.selected_path = opened
            dropdown = self.query_one(f"#file-dropdown", DropdownMenu)
            dropdown.add_class("hidden")
            self.main_view.run_analysis()


def run_tui(args: Optional[Any] = None, supported_archs: Optional[list] = None) -> None:
    """
    Run the TUI application.
    """
    try:
        app = RocprofTUIApp(args, supported_archs)
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to run TUI application: {str(e)}") from e
