"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from textual.containers import Vertical
from textual.widgets import Label, TabPane

from rocprof_compute_tui.widgets.center_panel.analyze_view import AnalyzeView
from rocprof_compute_tui.widgets.tabbed_content import TabsTabbedContent


class CenterPanel(Vertical):
    """
    The response area.
    """

    COMPONENT_CLASSES = {
        "border-title-status",
    }

    def __init__(self):
        super().__init__()

        self.default_tab = "center-analyze"
        self.analyze_view = AnalyzeView()

    def compose(self):
        with TabsTabbedContent(initial="tab-analyze"):
            with TabPane("Analyze Results", id="tab-analyze"):
                yield self.analyze_view
            # TODO:
            # with TabPane("placeholder (🚧)", id="tab-1"):
            #     yield Label("🚧 Under Construction")

    def on_mount(self) -> None:
        self.border_title = "CENTER TABS"
        self.add_class("section")
