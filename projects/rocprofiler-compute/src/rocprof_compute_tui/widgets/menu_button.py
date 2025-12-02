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

from typing import Any

from textual.reactive import reactive

from rocprof_compute_tui.widgets.instant_button import InstantButton


class InstantMenuButton(InstantButton):
    """Menu bar button that toggles a dropdown using instant-click semantics."""

    is_open: reactive[bool] = reactive(False, init=False)

    def __init__(self, label: str, menu_id: str, **kwargs: Any) -> None:
        super().__init__(label, **kwargs)
        self.menu_id = menu_id  # id of the DropdownMenu widget

    def watch_is_open(self, value: bool) -> None:
        """Show/hide the associated dropdown and update visual state."""
        try:
            dropdown = self.app.query_one(f"#{self.menu_id}")
        except Exception:
            return

        if value:
            dropdown.styles.display = "block"
            dropdown.styles.visibility = "visible"
            dropdown.styles.pointer_events = "auto"
            dropdown.styles.opacity = 1
            dropdown.styles.height = "auto"
            dropdown.styles.width = "auto"

            self.add_class("-active")
        else:
            dropdown.styles.display = "none"
            dropdown.styles.visibility = "hidden"
            dropdown.styles.pointer_events = "none"
            dropdown.styles.opacity = 0
            dropdown.styles.height = 0
            dropdown.styles.width = 0

            self.remove_class("-active")

    def on_instant_button_instant_pressed(
        self, event: InstantButton.InstantPressed
    ) -> None:
        """Toggle menu when *this* menu button is pressed."""
        if event.button is not self:
            return
        event.stop()
        self.is_open = not self.is_open
