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
        self.add_class("section")
