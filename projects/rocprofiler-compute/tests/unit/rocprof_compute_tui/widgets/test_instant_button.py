# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the InstantButton TUI widget."""

from unittest.mock import MagicMock, Mock

import pytest

from rocprof_compute_tui.widgets.instant_button import InstantButton

pytestmark = pytest.mark.tui

# =============================================================================
# Tests for InstantButton Widget
# =============================================================================


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
