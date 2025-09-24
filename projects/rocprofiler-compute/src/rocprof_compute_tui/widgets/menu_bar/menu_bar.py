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

from pathlib import Path
from typing import Any, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import Button

from rocprof_compute_tui.widgets.recent_directories import RecentDirectoriesScreen


class DropdownMenu(Container):
    def compose(self) -> ComposeResult:
        """Compose the dropdown menu with menu items."""
        yield Button("Open Workload", id="menu-open-workload", classes="menu-item")
        yield Button("Open Recent", id="menu-open-recent", classes="menu-item")
        # TODO:
        # yield Button("Attach", id="menu-attach", classes="menu-item")
        yield Button("Exit", id="menu-exit", classes="menu-item")

    def on_mount(self) -> None:
        self.add_class("hidden")


class MenuButton(Button):
    is_open = reactive(False)

    def __init__(self, label: str, menu_id: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(label, *args, **kwargs)
        self.menu_id = menu_id

    def on_click(self) -> None:
        self.is_open = not self.is_open
        dropdown = self.app.query_one(f"#{self.menu_id}", DropdownMenu)

        if self.is_open:
            dropdown.remove_class("hidden")
        else:
            dropdown.add_class("hidden")


class MenuBar(Container):
    """A menu bar that spans the width of the app."""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            MenuButton("File", "file-dropdown", id="menu-file"), id="menu-buttons"
        )

        with Container(id="dropdown-container"):
            yield DropdownMenu(id="file-dropdown")

    def on_mount(self) -> None:
        self.border_title = "MENU BAR"
        self.add_class("section")
        self.parent_main_view = self.screen.query_one("#main-container", Horizontal)

    @on(Button.Pressed, "#menu-open-recent")
    def show_recent(self) -> None:
        if not self.app.recent_dirs:
            self.notify("No recent directories found", severity="warning")
            return

        def on_recent_selected(selected_dir: Optional[str]) -> None:
            if selected_dir:
                path_obj = Path(selected_dir)
                if path_obj.exists():
                    self.parent_main_view.selected_path = path_obj
                    self.query_one("#file-dropdown", DropdownMenu).add_class("hidden")
                    self.parent_main_view.run_analysis()
                else:
                    # Remove non-existent path from recent dirs
                    if selected_dir in self.app.recent_dirs:
                        self.app.recent_dirs.remove(selected_dir)
                        self.app._save_recent_dirs()

        self.app.push_screen(
            RecentDirectoriesScreen(self.app.recent_dirs), on_recent_selected
        )

    @on(Button.Pressed, "#menu-exit")
    def exit_app(self) -> None:
        self.app.exit()
