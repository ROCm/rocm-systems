"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from textual.containers import Vertical
from textual.widgets import TabPane, TextArea

from rocprof_compute_tui.widgets.tabbed_content import TabsTabbedContent
from rocprof_compute_tui.widgets.tabs.tabs_terminal import Terimnal


class TabsArea(Vertical):
    """
    The response area.
    """

    COMPONENT_CLASSES = {
        "border-title-status",
    }

    def __init__(self):
        """Initialize the bottom panel."""

        super().__init__()

        # Create text areas as instance attributes
        self.description_area = TextArea(id="description-text", read_only=True)
        self.output_area = TextArea(id="output-text", read_only=True)

        # Set initial tab
        self.default_tab = "tab-output"

    def compose(self):
        with TabsTabbedContent(initial="tab-output"):
            with TabPane("METRIC DESCRIPTION", id="tab-description"):
                yield (self.description_area)

            with TabPane("OUTPUT", id="tab-output"):
                yield (self.output_area)

            with TabPane("TERMINAL", id="tab-terminal"):
                yield Terimnal()

    def on_mount(self) -> None:
        self.border_title = "TABS"
        self.add_class("section")
