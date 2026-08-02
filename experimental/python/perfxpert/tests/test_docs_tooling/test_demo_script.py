"""The bounded-creativity demo script must keep working.

A demo that silently rots is worse than no demo: it is discovered mid-
presentation. This runs the real script and asserts on the outcomes it
claims, so a change that breaks the walkthrough fails here first.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _PROJECT_ROOT / "scripts" / "demo_bounded_creativity.py"


@pytest.fixture(scope="module")
def demo_output() -> str:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"demo script exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
    )
    return result.stdout


def test_the_demo_reaches_every_step(demo_output: str) -> None:
    for n in range(1, 7):
        assert f"STEP {n}:" in demo_output, f"demo stopped before step {n}"


def test_the_hostile_model_is_rejected_at_every_boundary(demo_output: str) -> None:
    """Step 3 must show four rejections and no acceptance."""
    assert demo_output.count("REJECTED") == 4
    assert "*** ACCEPTED ***" not in demo_output, (
        "a hostile proposal got through a boundary the demo claims is closed"
    )


def test_the_demo_shows_parity_holding(demo_output: str) -> None:
    assert "identical?                   True" in demo_output


def test_the_demo_shows_the_injection_failing(demo_output: str) -> None:
    assert "none — injection did not land" in demo_output
