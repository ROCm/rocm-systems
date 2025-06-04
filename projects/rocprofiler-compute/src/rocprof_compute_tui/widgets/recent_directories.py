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
