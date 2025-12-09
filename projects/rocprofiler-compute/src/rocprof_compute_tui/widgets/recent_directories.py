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

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from rocprof_compute_tui.widgets.instant_button import InstantButton


class ClickableListItem(ListItem):
    """A ListItem that highlights on single click without selecting."""

    def on_mouse_down(self, event) -> None:  # noqa: ANN001
        event.stop()

        # Find parent ListView
        parent = self.parent
        list_view = None
        while parent is not None:
            if isinstance(parent, ListView):
                list_view = parent
                break
            parent = parent.parent

        if list_view is None:
            return

        # Determine index
        try:
            index = list_view.children.index(self)
        except ValueError:
            return

        # Update highlight immediately
        list_view.index = index
        list_view.refresh(layout=True)
        self.refresh(layout=True)

    def on_click(self, event) -> None:  # noqa: ANN001
        event.stop()

    def on_mouse_up(self, event) -> None:  # noqa: ANN001
        event.stop()


class RecentDirectoriesScreen(ModalScreen):
    """Modal screen to display recent directories as clickable list items."""

    def __init__(self, recent_dirs: list[str], current_dir: str | None = None) -> None:
        super().__init__()
        self.recent_dirs = recent_dirs
        self.current_dir = current_dir

    # -------------------------------------------------------------------------
    # Compose UI
    # -------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Container(id="recent-modal"):
            yield Label("Recent Directories", id="recent-title")

            if self.recent_dirs:
                with ListView(id="recent-list"):
                    for i, directory in enumerate(self.recent_dirs):
                        # Normal-looking list item (NO InstantButton inside)
                        yield ClickableListItem(
                            Label(directory),
                            id=f"recent-row-{i}",
                        )
            else:
                yield Label("No recent directories found", id="no-recent")

            with Horizontal(id="recent-buttons"):
                yield InstantButton("Select", variant="primary", id="select-recent")
                yield InstantButton("Close", variant="default", id="close-recent")

    # -------------------------------------------------------------------------
    # Initialize highlight when screen opens
    # -------------------------------------------------------------------------
    def on_mount(self) -> None:
        if not self.recent_dirs:
            return

        list_view = self.query_one("#recent-list", ListView)

        # Default to first item
        index = 0

        # If current_dir matches one of the recent dirs
        if self.current_dir:
            try:
                index = self.recent_dirs.index(self.current_dir)
            except ValueError:
                pass

        list_view.index = index
        list_view.focus()

    # -------------------------------------------------------------------------
    # Buttons
    # -------------------------------------------------------------------------
    def on_instant_button_instant_pressed(
        self, event: InstantButton.InstantPressed
    ) -> None:
        """Handle Select / Close / row-selection."""
        button = event.button
        bid = button.id

        # Close
        if bid == "close-recent":
            event.stop()
            self.dismiss(None)
            return

        # Select highlighted row
        if bid == "select-recent":
            event.stop()
            list_view = self.query_one("#recent-list", ListView)
            idx = list_view.index or 0
            self.dismiss(self.recent_dirs[idx])
            return

    # -------------------------------------------------------------------------
    # Keyboard activation (Enter or double-click)
    # -------------------------------------------------------------------------
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index or 0
        self.dismiss(self.recent_dirs[idx])
