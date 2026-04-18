"""Tests for perfxpert.tools.plateau."""

import pytest

from perfxpert.tools import plateau
from perfxpert.tools._class import ToolClass


def test_no_plateau_when_improving():
    history = [
        {"total_runtime_ns": 1000000},
        {"total_runtime_ns": 900000},
        {"total_runtime_ns": 800000},
    ]
    r = plateau.check(history)
    assert r["plateau_detected"] is False


def test_plateau_when_within_threshold_for_min_iterations():
    """Spec §5: < 2% change for 2+ consecutive iterations = plateau."""
    history = [
        {"total_runtime_ns": 1000000},
        {"total_runtime_ns": 1005000},   # +0.5%
        {"total_runtime_ns": 998000},    # -0.7%
        {"total_runtime_ns": 1002000},   # +0.4% from prior
    ]
    r = plateau.check(history)
    assert r["plateau_detected"] is True


def test_single_iteration_never_plateau():
    r = plateau.check([{"total_runtime_ns": 1000000}])
    assert r["plateau_detected"] is False


def test_is_read_only_class():
    assert plateau.check.__tool_class__ == ToolClass.READ_ONLY
