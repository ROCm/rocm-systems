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

from typing import List

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListItem, ListView


class RecentDirectoriesScreen(ModalScreen):
    """Modal screen to display recent directories."""

    def __init__(self, recent_dirs: List[str]) -> None:
        super().__init__()
        self.recent_dirs = recent_dirs

    def compose(self) -> ComposeResult:
        with Container(id="recent-modal"):
            yield Label("Recent Directories", id="recent-title")
            if self.recent_dirs:
                with ListView(id="recent-list"):
                    for directory in self.recent_dirs:
                        yield ListItem(Label(directory))
            else:
                yield Label("No recent directories found", id="no-recent")
            with Horizontal(id="recent-buttons"):
                yield Button("Select", variant="primary", id="select-recent")
                yield Button("Close", variant="default", id="close-recent")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-recent":
            self.dismiss()
        elif event.button.id == "select-recent":
            list_view = self.query_one("#recent-list", ListView)
            if list_view.highlighted_child:
                selected_dir = self.recent_dirs[list_view.index or 0]
                self.dismiss(selected_dir)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        selected_dir = self.recent_dirs[event.list_view.index or 0]
        self.dismiss(selected_dir)
