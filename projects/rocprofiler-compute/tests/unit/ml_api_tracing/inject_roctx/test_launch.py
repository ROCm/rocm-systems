# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the absolute-path workload launcher."""

import runpy
import sys
from pathlib import Path

import pytest

from ml_api_tracing.inject_roctx import installer, launch


@pytest.mark.parametrize(
    ("argv", "frameworks", "command"),
    [
        pytest.param(
            ["--frameworks", "torch", "triton", "--", "target.py", "--flag"],
            ["torch", "triton"],
            ["target.py", "--flag"],
            id="frameworks-and-separator",
        ),
        pytest.param(["--", "target.py"], [], ["target.py"], id="separator-only"),
        pytest.param(
            ["target.py", "two words"], [], ["target.py", "two words"], id="bare"
        ),
    ],
)
def test_parse_launch_arguments_splits_frameworks_and_command(
    argv, frameworks, command
):
    assert launch.parse_launch_arguments(argv) == (frameworks, command)


def test_parse_launch_arguments_exits_2_without_target(capsys):
    with pytest.raises(SystemExit) as excinfo:
        launch.parse_launch_arguments(["--frameworks", "torch"])

    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_launch_installs_frameworks_and_runs_target(tmp_path, monkeypatch):
    target = tmp_path / "target.py"
    ok_path = tmp_path / "ok.txt"
    target.write_text(
        f"from pathlib import Path\nPath({str(ok_path)!r}).write_text('ok')\n",
        encoding="utf-8",
    )
    launcher = Path(launch.__file__)
    recorded = []
    monkeypatch.setattr(
        installer, "install_global_wraps", lambda names: recorded.append(list(names))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(launcher), "--frameworks", "torch", "triton", "--", str(target)],
    )

    runpy.run_path(str(launcher), run_name="__main__")

    assert recorded == [["torch", "triton"]]
    assert ok_path.read_text(encoding="utf-8") == "ok"
