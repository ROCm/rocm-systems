# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the absolute-path workload launcher."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from ml_api_tracing.recording import installer


def test_launch_parses_frameworks_and_runs_target(tmp_path, monkeypatch):

    target = tmp_path / "target.py"
    ok_path = tmp_path / "ok.txt"
    target.write_text(
        f"from pathlib import Path\nPath({str(ok_path)!r}).write_text('ok')\n",
        encoding="utf-8",
    )
    launch = Path(installer.__file__).resolve().parent / "launch.py"
    recorded = []
    monkeypatch.setattr(
        installer, "install_global_wraps", lambda names: recorded.append(list(names))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(launch), "--frameworks", "torch", "triton", "--", str(target)],
    )
    runpy.run_path(str(launch))
    assert recorded == [["torch", "triton"]]
    assert ok_path.read_text(encoding="utf-8") == "ok"


def test_launch_exits_2_without_target(monkeypatch, capsys):

    launch = Path(installer.__file__).resolve().parent / "launch.py"
    monkeypatch.setattr(sys, "argv", [str(launch), "--frameworks", "torch"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(launch))
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err


@pytest.mark.parametrize("exit_code", [0, 7])
def test_absolute_path_launch_bootstraps_clean_process(tmp_path, exit_code):
    target = tmp_path / "workload.py"
    target.write_text(
        "import json, sys\n"
        "from ml_api_tracing.recording import installer\n"
        "print(json.dumps([sys.argv, __name__, installer.__file__]))\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    launch = Path(installer.__file__).resolve().parent / "launch.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-S",
            str(launch),
            "--",
            str(target),
            "two words",
            "--flag",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == exit_code, completed.stderr
    assert json.loads(completed.stdout) == [
        [str(target), "two words", "--flag"],
        "__main__",
        str(Path(installer.__file__).resolve()),
    ]
