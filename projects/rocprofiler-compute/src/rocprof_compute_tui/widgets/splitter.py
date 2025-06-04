"""
Specialized Widget Modules
-------------------------
Contains custom widget implementations for the application.
"""

import pandas as pd
from textual.events import MouseDown, MouseMove, MouseUp
from textual.widgets import Static


class HorizontalSplitter(Static):
    """A draggable horizontal splitter between panels."""

    def __init__(self, **kwargs):
        """Initialize the horizontal splitter."""
        super().__init__("", id="splitter", **kwargs)
        self.dragging = False

    def on_mouse_down(self, event: MouseDown) -> None:
        """Handle mouse down events."""
        self.dragging = True
        self.capture_mouse()

    def on_mouse_up(self, event: MouseUp) -> None:
        """Handle mouse up events."""
        self.dragging = False
        self.release_mouse()

    def on_mouse_move(self, event: MouseMove) -> None:
        """Handle mouse move events for dragging."""
        if not self.dragging:
            return

        app = self.app
        top = app.query_one("#center-panel")
        bottom = app.query_one("#bottom-panel")

        new_top_height = event.screen_y
        total_height = app.size.height
        new_bottom_height = total_height - new_top_height - 1

        # Ensure minimum height
        if new_top_height < 3 or new_bottom_height < 3:
            return

        top.styles.height = new_top_height
        bottom.styles.height = new_bottom_height
