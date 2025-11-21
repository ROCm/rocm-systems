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

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button

from rocprof_compute_tui.widgets.recent_directories import RecentDirectoriesScreen


class DropdownMenu(Container):
    """Dropdown menu container with proper visibility handling."""

    BINDINGS = [
        Binding("escape", "close_menu", "Close", show=False),
    ]

    class Closed(Message):
        """Posted when dropdown is closed."""

        pass

    def compose(self) -> ComposeResult:
        yield Button("Open Workload", id="menu-open-workload", classes="menu-item")
        yield Button("Open Recent", id="menu-open-recent", classes="menu-item")
        yield Button("Exit", id="menu-exit", classes="menu-item")

    def on_mount(self) -> None:
        self.display = False  # Use display instead of CSS class for reliability

    def show(self) -> None:
        self.display = True
        self.focus()

    def hide(self) -> None:
        self.display = False
        self.post_message(self.Closed())

    def action_close_menu(self) -> None:
        self.hide()

    def on_blur(self) -> None:
        # Check if focus moved to a child or the parent menu button
        if self.display:
            # Use call_later to allow focus to settle first
            self.call_later(self._check_focus_and_close)

    def _check_focus_and_close(self) -> None:
        focused = self.app.focused
        # Don't close if focus is on a menu item or the menu button
        if focused is None:
            self.hide()
            return
        if not (
            self.is_ancestor_of(focused)
            or (hasattr(focused, "id") and focused.id == "menu-file")
        ):
            self.hide()

    def is_ancestor_of(self, widget) -> bool:  # noqa: ANN001
        current = widget
        while current is not None:
            if current is self:
                return True
            current = current.parent
        return False


class MenuButton(Button):
    """Menu button with reactive open state and proper sync."""

    is_open: reactive[bool] = reactive(False, init=False)

    def __init__(self, label: str, menu_id: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(label, *args, **kwargs)
        self.menu_id = menu_id
        self._dropdown: DropdownMenu | None = None

    def on_mount(self) -> None:
        self._dropdown = self.app.query_one(f"#{self.menu_id}", DropdownMenu)

    def watch_is_open(self, value: bool) -> None:
        if self._dropdown is None:
            return
        if value:
            self._dropdown.show()
            self.add_class("active")
        else:
            self._dropdown.hide()
            self.remove_class("active")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Toggle dropdown on press."""
        if event.button is not self:
            return
        event.stop()  # Prevent event bubbling
        self.is_open = not self.is_open

    @on(DropdownMenu.Closed)
    def on_dropdown_closed(self, event: DropdownMenu.Closed) -> None:
        self.is_open = False


class MenuBar(Container):
    """A menu bar that spans the width of the app."""

    BINDINGS = [
        Binding("escape", "close_all_menus", "Close menus", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Horizontal(
            MenuButton("File", "file-dropdown", id="menu-file"), id="menu-buttons"
        )
        with Container(id="dropdown-container"):
            yield DropdownMenu(id="file-dropdown")

    def on_mount(self) -> None:
        self.border_title = "MENU BAR"
        self.add_class("section")

    def action_close_all_menus(self) -> None:
        for menu_btn in self.query(MenuButton):
            menu_btn.is_open = False

    def close_dropdown(self) -> None:
        menu_button = self.query_one("#menu-file", MenuButton)
        menu_button.is_open = False

    @on(Button.Pressed, "#menu-open-recent")
    def show_recent(self) -> None:
        if not self.app.recent_dirs:
            self.notify("No recent directories found", severity="warning")
            return

        self.close_dropdown()
        self.app.push_screen(
            RecentDirectoriesScreen(self.app.recent_dirs), self.app.on_recent_selected
        )

    @on(Button.Pressed, "#menu-exit")
    def exit_app(self) -> None:
        self.app.exit()

    def on_click(self, event) -> None:  # noqa: ANN001
        """Close menus when clicking outside dropdown area."""
        # Check if click was outside menu system
        menu_btn = self.query_one("#menu-file", MenuButton)
        dropdown = self.query_one("#file-dropdown", DropdownMenu)

        if menu_btn.is_open:
            # Get click coordinates relative to widgets
            click_in_dropdown = dropdown.region.contains_point(event.screen_offset)
            click_in_button = menu_btn.region.contains_point(event.screen_offset)

            if not click_in_dropdown and not click_in_button:
                menu_btn.is_open = False
