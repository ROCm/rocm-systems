# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the DropdownMenu and MenuButton TUI widgets."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from rocprof_compute_tui.widgets.menu_bar.menu_bar import DropdownMenu, MenuButton

pytestmark = pytest.mark.tui

# =============================================================================
# Tests for DropdownMenu Widget
# =============================================================================


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


# =============================================================================
# Tests for MenuButton Widget
# =============================================================================


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
