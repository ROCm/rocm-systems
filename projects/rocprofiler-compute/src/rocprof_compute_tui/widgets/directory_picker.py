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
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import DirectoryTree, Label, Static

from rocprof_compute_tui.widgets.instant_button import InstantButton


class DirectoryPicker(ModalScreen[Optional[Path]]):
    DEFAULT_CSS = """
    DirectoryPicker {
        align: center middle;
    }

    #picker-container {
        width: 90;
        height: 35;
        background: $surface;
        border: thick $primary;
    }

    #picker-header {
        dock: top;
        width: 100%;
        height: auto;
        background: $primary;
        padding: 1;
    }

    #picker-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        color: $text;
    }

    #breadcrumb-container {
        dock: top;
        width: 100%;
        height: auto;
        padding: 1;
        background: $panel;
    }

    #breadcrumb {
        width: 100%;
        color: $text;
        content-align: left middle;
    }

    #picker-content {
        width: 100%;
        height: 1fr;
        padding: 1;
    }

    #dir-tree {
        width: 100%;
        height: 100%;
        border: round $primary-darken-2;
    }

    #picker-footer {
        dock: bottom;
        width: 100%;
        height: auto;
        padding: 1;
        background: $surface-darken-1;
    }

    #selection-info {
        dock: top;
        width: 100%;
        padding: 0 1;
        color: $success;
        text-style: italic;
    }

    #picker-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    #picker-buttons InstantButton {
        margin: 0 1;
        min-width: 16;
    }
    """

    def __init__(self, start_path: Optional[Path] = None) -> None:
        super().__init__()
        self.start_path = start_path or Path.cwd()
        self.selected_path: Optional[Path] = self.start_path

    def compose(self) -> ComposeResult:
        with Container(id="picker-container"):
            with Container(id="picker-header"):
                yield Label("📁 Select Directory", id="picker-title")

            with Container(id="breadcrumb-container"):
                yield Static(self._format_breadcrumb(self.start_path), id="breadcrumb")

            with Container(id="picker-content"):
                yield DirectoryTree(str(self.start_path), id="dir-tree")

            with Container(id="picker-footer"):
                yield Static("", id="selection-info")
                with Horizontal(id="picker-buttons"):
                    yield InstantButton("Select", id="select-dir", classes="primary")
                    yield InstantButton("Cancel", id="cancel-dir", classes="error")

    def on_mount(self) -> None:
        tree = self.query_one("#dir-tree", DirectoryTree)
        tree.show_root = True
        tree.show_guides = True
        tree.focus()
        self._update_selection_info()

    def _format_breadcrumb(self, path: Path) -> str:
        parts = list(path.parts)
        if len(parts) > 5:
            return f"{parts[0]} / ... / {' / '.join(parts[-4:])}"
        return str(path)

    def _update_selection_info(self) -> None:
        info = self.query_one("#selection-info", Static)
        if self.selected_path:
            info.update(f"Selected: {self.selected_path.name} ({self.selected_path})")
        else:
            info.update("No directory selected")

    @on(DirectoryTree.DirectorySelected)
    def on_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Update selection when a directory is selected in the tree."""
        self.selected_path = event.path
        breadcrumb = self.query_one("#breadcrumb", Static)
        breadcrumb.update(self._format_breadcrumb(event.path))
        self._update_selection_info()

    def on_instant_button_instant_pressed(
        self, event: InstantButton.InstantPressed
    ) -> None:
        """Handle instant button presses in this picker."""
        bid = event.button.id

        if bid == "select-dir":
            event.stop()
            if self.selected_path:
                self.dismiss(self.selected_path)
            else:
                # Nothing selected; keep modal open and notify user.
                self.notify("No directory selected", severity="warning")

        elif bid == "cancel-dir":
            event.stop()
            self.dismiss(None)

    def on_key(self, event) -> None:  # noqa: ANN001
        if event.key == "enter":
            button = self.query_one("#select-dir", InstantButton)
            self.on_instant_button_instant_pressed(InstantButton.InstantPressed(button))
        elif event.key == "escape":
            button = self.query_one("#cancel-dir", InstantButton)
            self.on_instant_button_instant_pressed(InstantButton.InstantPressed(button))
