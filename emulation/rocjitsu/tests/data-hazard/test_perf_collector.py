# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Pytest: flamegraph assembly in ``mutate_and_test.benchmark``.

Run from ``tests/data-hazard`` (so ``mutate_and_test`` imports)::

    cd tests/data-hazard && pytest test_perf_collector.py -v

Every external command is faked, so neither perf nor the FlameGraph scripts
have to be installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from mutate_and_test.benchmark import PerfCollector  # noqa: E402
from mutate_and_test import benchmark  # noqa: E402

_SAMPLES = "kernel;plugin;track_lds_write 12\n"


def _collector(tmp_path: Path) -> PerfCollector:
    """A collector with one recorded profile, over commands that do not exist."""
    collector = PerfCollector(
        workdir=tmp_path,
        perf=Path("/usr/bin/perf"),
        stackcollapse=Path("/opt/flamegraph/stackcollapse-perf.pl"),
        flamegraph_pl=Path("/opt/flamegraph/flamegraph.pl"),
    )
    _, data_file = collector.record_args("demo")
    data_file.write_text("not a real perf.data, the commands reading it are faked")
    return collector


def _fake_commands(
    monkeypatch: pytest.MonkeyPatch, outcomes: Dict[str, subprocess.CompletedProcess]
) -> List[List[str]]:
    """Answer each step of the pipeline from *outcomes*, keyed by argv marker."""
    seen: List[List[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        for marker, result in outcomes.items():
            if any(marker in arg for arg in argv):
                return result
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    return seen


def _ok(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _failed(stderr: str, returncode: int = 1) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=stderr
    )


def test_writes_svg_when_every_step_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collector = _collector(tmp_path)
    _fake_commands(
        monkeypatch,
        {
            "perf": _ok("perf script output"),
            "stackcollapse-perf.pl": _ok(_SAMPLES),
            "flamegraph.pl": _ok("<svg>flames</svg>"),
        },
    )

    svg = tmp_path / "flamegraph.svg"
    collector.generate_flamegraph(svg)

    assert svg.read_text() == "<svg>flames</svg>"


def test_failed_flamegraph_leaves_no_svg_and_reports_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    collector = _collector(tmp_path)
    _fake_commands(
        monkeypatch,
        {
            "perf": _ok("perf script output"),
            "stackcollapse-perf.pl": _ok(_SAMPLES),
            "flamegraph.pl": _failed("ERROR: unknown option: --colors\n"),
        },
    )

    svg = tmp_path / "flamegraph.svg"
    collector.generate_flamegraph(svg)

    assert not svg.exists()
    err = capsys.readouterr().err
    assert "flamegraph.pl failed (exit 1)" in err
    assert "unknown option: --colors" in err


def test_empty_flamegraph_output_leaves_no_svg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    collector = _collector(tmp_path)
    _fake_commands(
        monkeypatch,
        {
            "perf": _ok("perf script output"),
            "stackcollapse-perf.pl": _ok(_SAMPLES),
            "flamegraph.pl": _ok("   \n"),
        },
    )

    svg = tmp_path / "flamegraph.svg"
    collector.generate_flamegraph(svg)

    assert not svg.exists()
    assert "produced no output" in capsys.readouterr().err


def test_failed_perf_script_skips_the_profile_and_reports_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    collector = _collector(tmp_path)
    seen = _fake_commands(
        monkeypatch,
        {
            "perf": _failed("failed to open perf.data: Permission denied\n"),
            "stackcollapse-perf.pl": _ok(_SAMPLES),
            "flamegraph.pl": _ok("<svg>flames</svg>"),
        },
    )

    svg = tmp_path / "flamegraph.svg"
    collector.generate_flamegraph(svg)

    assert not svg.exists()
    # The unusable profile is dropped before it can reach flamegraph.pl.
    assert all("flamegraph.pl" not in " ".join(argv) for argv in seen)
    err = capsys.readouterr().err
    assert "perf script on demo.perf.data failed (exit 1)" in err
    assert "Permission denied" in err
