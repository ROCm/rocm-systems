from pathlib import Path
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Label, Static


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

    #picker-buttons Button {
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
                    yield Button("Select", variant="primary", id="select-dir")
                    yield Button("Cancel", variant="error", id="cancel-dir")

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
        self.selected_path = event.path
        breadcrumb = self.query_one("#breadcrumb", Static)
        breadcrumb.update(self._format_breadcrumb(event.path))
        self._update_selection_info()

    @on(Button.Pressed, "#select-dir")
    def select_directory(self) -> None:
        if self.selected_path:
            self.dismiss(self.selected_path)
        else:
            self.notify("No directory selected", severity="warning")

    @on(Button.Pressed, "#cancel-dir")
    def cancel_selection(self) -> None:
        self.dismiss(None)

    def on_key(self, event) -> None:  # noqa: ANN001
        if event.key == "enter":
            self.select_directory()
        elif event.key == "escape":
            self.cancel_selection()
