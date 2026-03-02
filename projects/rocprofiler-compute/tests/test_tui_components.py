##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

"""Unit tests for TUI widgets."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from rocprof_compute_tui.widgets.instant_button import InstantButton
from rocprof_compute_tui.widgets.menu_bar.menu_bar import DropdownMenu, MenuButton


class TestInstantButton:
    """Test suite for InstantButton widget."""

    def test_instant_button_posts_message_exactly_once(self):
        """Test that InstantButton posts message exactly once per press."""
        button = InstantButton("Test Button")
        button.post_message = Mock()

        # Create a Button.Pressed event
        event = MagicMock()
        event.button = button

        # Press button once
        button.on_button_pressed(event)

        # Should have posted exactly one message
        assert button.post_message.call_count == 1
        posted_message = button.post_message.call_args[0][0]
        assert isinstance(posted_message, InstantButton.InstantPressed)
        assert posted_message.button is button

    def test_instant_button_ignores_other_button_events(self):
        """Test that InstantButton ignores events from other buttons."""
        button = InstantButton("Test Button")
        other_button = InstantButton("Other Button")
        button.post_message = Mock()

        # Create event from different button
        event = MagicMock()
        event.button = other_button

        button.on_button_pressed(event)

        # Should not have posted any message
        assert not button.post_message.called

    def test_trigger_posts_instant_pressed(self):
        """Test that trigger() method posts InstantPressed message."""
        button = InstantButton("Test Button")
        button.post_message = Mock()

        button.trigger()

        # Verify InstantPressed was posted
        assert button.post_message.called
        posted_message = button.post_message.call_args[0][0]
        assert isinstance(posted_message, InstantButton.InstantPressed)
        assert posted_message.button is button


class TestDropdownMenu:
    """Test suite for DropdownMenu widget."""

    def test_is_visible_false_sets_hidden_state(self):
        """Test that is_visible=False sets correct hidden styles."""
        menu = DropdownMenu()
        menu.styles = MagicMock()
        menu.refresh = Mock()

        # Trigger the watcher by setting is_visible
        menu.watch_is_visible(False)

        # Verify styles are set for hidden state
        assert menu.styles.pointer_events == "none"
        assert menu.styles.visibility == "hidden"
        assert menu.styles.opacity == 0.0
        assert menu.display is False
        menu.refresh.assert_called_with(repaint=True, layout=False)

    def test_is_visible_true_sets_visible_state(self):
        """Test that is_visible=True sets correct visible styles."""
        menu = DropdownMenu()
        menu.styles = MagicMock()
        menu.refresh = Mock()

        # Trigger the watcher by setting is_visible
        menu.watch_is_visible(True)

        # Verify styles are set for visible state
        assert menu.display is True
        assert menu.styles.pointer_events == "auto"
        assert menu.styles.visibility == "visible"
        assert menu.styles.opacity == 1.0
        menu.refresh.assert_called_with(repaint=True, layout=False)

    def test_check_focus_closes_when_sequence_matches(self):
        """Test that _check_focus_and_close closes menu when no focus after blur."""
        menu = DropdownMenu()
        menu.is_visible = True
        menu.hide = Mock()

        # Mock the app property using patch
        with patch.object(type(menu), "app", new_callable=lambda: MagicMock()):
            menu.app.focused = None

            # Set event sequence to 5 (no focus after blur)
            menu._event_sequence = 5

            # Call with blur sequence 5 (same as current, so no focus occurred)
            menu._check_focus_and_close(5)

            # Should have called hide
            menu.hide.assert_called_once()

    def test_check_focus_ignores_old_blur_events(self):
        """Test that _check_focus_and_close ignores old blur events."""
        menu = DropdownMenu()
        menu.is_visible = True
        menu.hide = Mock()

        # Current event sequence is newer (focus occurred after the blur)
        menu._event_sequence = 10

        # Call with old blur sequence
        menu._check_focus_and_close(5)

        # Should not have called hide (focus event at seq 10 > blur seq 5)
        assert not menu.hide.called

    def test_check_focus_stays_open_when_refocused(self):
        """Test that _check_focus_and_close stays open if focus was regained."""
        menu = DropdownMenu()
        menu.is_visible = True
        menu.hide = Mock()

        # Blur happened at sequence 5, then focus at sequence 6
        # Current event sequence is 6 (focus was regained)
        menu._event_sequence = 6

        # Call with blur sequence 5
        menu._check_focus_and_close(5)

        # Should not close because focus was regained (6 > 5)
        assert not menu.hide.called

    def test_show_sets_visible_and_focuses(self):
        """Test that show() sets is_visible=True and focuses menu."""
        menu = DropdownMenu()
        menu.focus = Mock()

        menu.show()

        assert menu.is_visible is True
        menu.focus.assert_called_once()

    def test_hide_sets_not_visible_and_posts_closed(self):
        """Test that hide() sets is_visible=False and posts Closed message."""
        menu = DropdownMenu()
        menu.is_visible = True  # Set visible first
        menu.post_message = Mock()

        menu.hide()

        assert menu.is_visible is False
        assert menu.post_message.called
        posted_message = menu.post_message.call_args[0][0]
        assert isinstance(posted_message, DropdownMenu.Closed)

    def test_hide_is_idempotent(self):
        """Test that hide() does nothing when menu is already hidden."""
        menu = DropdownMenu()
        menu.is_visible = False  # Already hidden
        menu.post_message = Mock()

        menu.hide()

        # Should not post Closed message when already hidden
        assert not menu.post_message.called
        assert menu.is_visible is False


class TestMenuButton:
    """Test suite for MenuButton widget."""

    def test_is_open_true_shows_dropdown(self):
        """Test that is_open=True calls dropdown.show()."""
        button = MenuButton("File", "test-dropdown")
        dropdown = MagicMock()
        button._dropdown = dropdown
        button.add_class = Mock()
        button.refresh = Mock()

        button.watch_is_open(True)

        dropdown.show.assert_called_once()
        button.add_class.assert_called_with("-active")

    def test_is_open_false_hides_dropdown(self):
        """Test that is_open=False calls dropdown.hide()."""
        button = MenuButton("File", "test-dropdown")
        dropdown = MagicMock()
        button._dropdown = dropdown
        button.remove_class = Mock()
        button.refresh = Mock()

        button.watch_is_open(False)

        dropdown.hide.assert_called_once()
        button.remove_class.assert_called_with("-active")

    def test_button_pressed_toggles_is_open(self):
        """Test that button press toggles is_open state."""
        button = MenuButton("File", "test-dropdown")
        button.is_open = False

        event = MagicMock()
        event.button = button

        button.on_button_pressed(event)

        assert button.is_open is True

        # Press again
        button.on_button_pressed(event)

        assert button.is_open is False

    def test_dropdown_closed_sets_is_open_false(self):
        """Test that dropdown closed event sets is_open to False."""
        button = MenuButton("File", "test-dropdown")
        button.is_open = True

        event = DropdownMenu.Closed()

        button.on_dropdown_closed(event)

        assert button.is_open is False


# Mark all tests in this file with the 'tui' marker for pytest
pytestmark = pytest.mark.tui
