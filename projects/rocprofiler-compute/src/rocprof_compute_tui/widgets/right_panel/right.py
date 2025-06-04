"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from textual.containers import Vertical
from textual.widgets import Label


class RightPanel(Vertical):
    """Right panel for additional tools."""

    def __init__(self):
        """Initialize the right panel."""
        super().__init__()

    def compose(self):
        """Compose the right panel."""
        yield Label("🚧 Under Construction")

    def _on_mount(self):
        self.border_title = "🚧 UNDER CONSTRUCTION"
        self.add_class("section")
