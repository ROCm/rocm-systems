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

from __future__ import annotations

from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button

from rocprof_compute_tui.widgets.directory_picker import DirectoryPicker
from rocprof_compute_tui.widgets.recent_directories import RecentDirectoriesScreen


class DropdownMenu(Container):
    """Dropdown menu with reactive visibility."""

    is_visible: reactive[bool] = reactive(False)
    BINDINGS = [
        Binding("escape", "close_menu", "Close", show=False),
    ]

    class Closed(Message):
        """Posted when dropdown is closed."""

        pass

    def __init__(self, *args: Any, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._event_sequence = 0  # Single monotonic counter for all events
        self._blur_timer = None  # Track active blur timer for cancellation

    def compose(self) -> ComposeResult:
        yield Button("Open Workload", id="menu-open-workload", classes="menu-item")
        yield Button("Open Recent", id="menu-open-recent", classes="menu-item")
        yield Button("Exit", id="menu-exit", classes="menu-item")

    def on_mount(self) -> None:
        self.is_visible = False

    def watch_is_visible(self, value: bool) -> None:
        """React to visibility changes."""
        if value:
            self.display = True
            self.styles.pointer_events = "auto"
            self.styles.visibility = "visible"
            self.styles.opacity = 1.0
        else:
            self.styles.pointer_events = "none"
            self.styles.visibility = "hidden"
            self.styles.opacity = 0.0
            self.display = False
        # Force immediate screen repaint
        self.refresh(repaint=True, layout=False)

    # -------------------------------------------------------------------------
    # Public show/hide API
    # -------------------------------------------------------------------------
    def show(self) -> None:
        """Show the dropdown and make it focusable + hit-testable."""
        self.is_visible = True
        self.focus()

    def hide(self) -> None:
        """Hide the dropdown and remove it from hit-testing."""
        # Guard: only hide and post Closed if currently visible
        if not self.is_visible:
            return

        # Cancel any pending blur timer to prevent duplicate hide() calls
        if self._blur_timer is not None:
            self._blur_timer.stop()
            self._blur_timer = None

        self.is_visible = False
        self.post_message(self.Closed())

    def action_close_menu(self) -> None:
        self.hide()

    # -------------------------------------------------------------------------
    # Focus handling: close when focus leaves menu & menu button
    # -------------------------------------------------------------------------
    def on_blur(self) -> None:
        """Mark blur event but don't close immediately."""
        if not self.is_visible:
            return

        # Cancel any existing timer before starting a new one
        if self._blur_timer is not None:
            self._blur_timer.stop()

        self._event_sequence += 1
        blur_sequence = self._event_sequence

        # Use set_timer with fixed 200ms delay
        # This is more predictable than call_later()
        self._blur_timer = self.set_timer(
            0.2,  # 200ms - enough for focus to settle even over SSH
            lambda: self._check_focus_and_close(blur_sequence),
        )

    def on_focus(self) -> None:
        """Track focus events to cancel blur."""
        self._event_sequence += 1

    def _check_focus_and_close(self, blur_sequence: int) -> None:
        """Only close if no focus event occurred after the blur."""
        # Clear the timer reference since this callback is executing
        self._blur_timer = None

        # Check if focus was regained after this blur
        # Using a single monotonic counter: if current sequence is higher
        # than blur_sequence, a focus event occurred after the blur
        if self._event_sequence > blur_sequence:
            # Focus was regained, don't close
            return

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
    """Menu button with reactive open state and proper sync with DropdownMenu."""

    is_open: reactive[bool] = reactive(False, init=False)

    def __init__(self, label: str, menu_id: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(label, *args, **kwargs)
        self.menu_id = menu_id
        self._dropdown: DropdownMenu | None = None

    def on_mount(self) -> None:
        # IMPORTANT: delay lookup until after the full DOM is built
        def late_init() -> None:
            try:
                self._dropdown = self.app.query_one(f"#{self.menu_id}", DropdownMenu)
            except Exception:
                self._dropdown = None

        self.call_later(late_init)

    def watch_is_open(self, value: bool) -> None:
        """React to is_open changes by showing/hiding the dropdown."""
        if self._dropdown is None:
            return

        if value:
            self._dropdown.show()
            self.add_class("-active")
        else:
            self._dropdown.hide()
            self.remove_class("-active")
        # Force immediate button repaint
        self.refresh(repaint=True, layout=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Toggle dropdown on press."""
        if event.button is not self:
            return
        # Don't stop event - let it propagate for proper rendering
        self.is_open = not self.is_open

    @on(DropdownMenu.Closed)
    def on_dropdown_closed(self, event: DropdownMenu.Closed) -> None:  # noqa: ARG002
        self.is_open = False


class MenuBar(Container):
    """A menu bar that spans the width of the app."""

    BINDINGS = [
        Binding("escape", "close_all_menus", "Close menus", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Horizontal(
            MenuButton("File", "file-dropdown", id="menu-file"),
            id="menu-buttons",
        )
        with Container(id="dropdown-container"):
            # This ID is used everywhere; do not change lightly
            yield DropdownMenu(id="file-dropdown")

    def on_mount(self) -> None:
        self.border_title = "MENU BAR"
        self.add_class("section")

    def action_close_all_menus(self) -> None:
        """Close all open menus in the bar."""
        for menu_btn in self.query(MenuButton):
            menu_btn.is_open = False

    def close_dropdown(self) -> None:
        """Close the File menu dropdown."""
        menu_button = self.query_one("#menu-file", MenuButton)
        menu_button.is_open = False

    # -------------------------------------------------------------------------
    # Menu item actions
    # -------------------------------------------------------------------------
    @on(Button.Pressed, "#menu-open-workload")
    def open_workload(self, event: Button.Pressed) -> None:
        """Open the directory picker for workload selection."""
        self.close_dropdown()
        self._start_pick_directory()

    @on(Button.Pressed, "#menu-open-recent")
    def show_recent(self, event: Button.Pressed) -> None:  # noqa: ARG002
        """Open the Recent Directories screen."""
        if not self.app.recent_dirs:
            self.notify("No recent directories found", severity="warning")
            return

        # Close the dropdown when opening a modal
        self.close_dropdown()

        self.app.push_screen(
            RecentDirectoriesScreen(self.app.recent_dirs),
            self.app.on_recent_selected,
        )

    @on(Button.Pressed, "#menu-exit")
    def exit_app(self, event: Button.Pressed) -> None:  # noqa: ARG002
        """Exit the application."""
        self.app.exit()

    # -------------------------------------------------------------------------
    # Asynchronous directory picker workflow (moved from App into MenuBar)
    # -------------------------------------------------------------------------
    @work
    async def _start_pick_directory(self) -> None:
        """Open directory picker and handle selection."""
        app = self.app

        try:
            picker = DirectoryPicker()
            opened = await app.push_screen_wait(picker)
            if opened:
                app.log(f"Directory selected: {opened}")
                app.notify(f"Selected directory: {opened}", severity="information")

                app.add_recent_dir(str(opened))
                app.main_view.selected_path = opened

                app.notify("Running analysis…", severity="information")
                app.main_view.run_analysis()
            else:
                app.log("Directory selection cancelled")
                app.notify("Directory selection cancelled", severity="information")

        except Exception as e:  # noqa: BLE001
            app.log(f"Error in directory picker: {e}")
            app.notify(f"Error opening directory picker: {e}", severity="error")

    # -------------------------------------------------------------------------
    # Click outside to close menu
    # -------------------------------------------------------------------------
    def on_click(self, event) -> None:  # noqa: ANN001
        """Close menus when clicking outside dropdown area."""
        menu_btn = self.query_one("#menu-file", MenuButton)
        dropdown = self.query_one("#file-dropdown", DropdownMenu)

        if menu_btn.is_open and dropdown.is_visible:
            # Get click coordinates relative to widgets
            click_in_dropdown = dropdown.region.contains_point(event.screen_offset)
            click_in_button = menu_btn.region.contains_point(event.screen_offset)

            if not click_in_dropdown and not click_in_button:
                menu_btn.is_open = False
