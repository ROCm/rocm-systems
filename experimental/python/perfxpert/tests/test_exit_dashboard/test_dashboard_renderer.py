"""Contract: --render flag emits a human-friendly rich-rendered summary."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "exit_dashboard.py"


@pytest.mark.dashboard
def test_render_mode_prints_table(tmp_path: Path) -> None:
    out = tmp_path / "dash.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--output", str(out), "--render", "--allow-partial"],
        capture_output=True, text=True,
    )
    # Look for table-ish output in stdout
    stdout = proc.stdout
    assert "Audit Gate" in stdout or "GO" in stdout or "NO-GO" in stdout or "PARTIAL" in stdout
    assert "parity" in stdout.lower()
    assert "red_team" in stdout or "red-team" in stdout.lower()
