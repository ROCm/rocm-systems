# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from textual.message import Message
from textual.widgets import Button


class InstantButton(Button):
    """
    A button that posts a custom InstantPressed message for each button press.
    """

    class InstantPressed(Message):
        """Custom message fired once for each successful button press."""

        def __init__(self, button: "InstantButton") -> None:
            super().__init__()
            self.button = button  # the button that was pressed

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate Textual's Button.Pressed into InstantPressed."""
        if event.button is not self:
            return

        event.stop()
        self.post_message(self.InstantPressed(self))

    def trigger(self) -> None:
        """Programmatically trigger button (for keyboard shortcuts)."""
        self.post_message(self.InstantPressed(self))
