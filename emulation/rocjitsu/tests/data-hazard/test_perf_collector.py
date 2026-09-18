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
from typing import Dict, List, NamedTuple

import pytest

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from mutate_and_test.benchmark import PerfCollector  # noqa: E402
from mutate_and_test import benchmark  # noqa: E402

_SAMPLES = "kernel;plugin;track_lds_write 12\n"


def _bare_collector(tmp_path: Path) -> PerfCollector:
    """A collector with nothing recorded, over commands that do not exist."""
    return PerfCollector(
        workdir=tmp_path,
        perf=Path("/usr/bin/perf"),
        stackcollapse=Path("/opt/flamegraph/stackcollapse-perf.pl"),
        flamegraph_pl=Path("/opt/flamegraph/flamegraph.pl"),
    )


def _collector(tmp_path: Path) -> PerfCollector:
    """A collector with one recorded profile."""
    collector = _bare_collector(tmp_path)
    _, data_file = collector.record_args("demo")
    data_file.write_text("not a real perf.data, the commands reading it are faked")
    return collector


def _step_name(argv: List[str]) -> str:
    """The program a step runs, named as the pipeline step it stands for.

    The FlameGraph steps run under an interpreter, so the script is the program
    and ``perl`` is not. Names are whole: ``stackcollapse-perf.pl`` is its own
    step rather than an instance of ``perf``.
    """
    program = Path(argv[0]).name
    if program == "perl" and len(argv) > 1:
        return Path(argv[1]).name
    return program


class _Call(NamedTuple):
    """One step of the pipeline as the fake ran it."""

    step: str
    argv: List[str]
    stdin: str


def _fake_commands(
    monkeypatch: pytest.MonkeyPatch, outcomes: Dict[str, subprocess.CompletedProcess]
) -> List[_Call]:
    """Answer each step of the pipeline from *outcomes*, keyed by program name."""
    seen: List[_Call] = []

    def fake_run(argv, **kwargs):
        argv = list(argv)
        step = _step_name(argv)
        seen.append(_Call(step, argv, kwargs.get("input", "")))
        if step not in outcomes:
            raise AssertionError(f"unexpected command: {argv}")
        return outcomes[step]

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
    seen = _fake_commands(
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
    # The profile is folded before it is drawn: perf's script output reaches
    # stackcollapse, and what stackcollapse folds is what flamegraph.pl draws.
    steps = {call.step: call for call in seen}
    assert steps["stackcollapse-perf.pl"].stdin == "perf script output"
    assert steps["flamegraph.pl"].stdin == _SAMPLES


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


def test_missing_profiles_leave_no_svg_and_report_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    collector = _bare_collector(tmp_path)

    svg = tmp_path / "flamegraph.svg"
    svg.write_text("<svg>an earlier run</svg>")
    collector.generate_flamegraph(svg)

    assert not svg.exists()
    assert "no perf.data files found" in capsys.readouterr().out


# Each way the pipeline ends without a flamegraph, by the step that ends it.
# The steps after the failing one are given outcomes that would have succeeded,
# so a case that stops early stops on its own account.
_FAILURES = {
    "perf script fails": {
        "perf": _failed("failed to open perf.data: Permission denied\n"),
        "stackcollapse-perf.pl": _ok(_SAMPLES),
        "flamegraph.pl": _ok("<svg>flames</svg>"),
    },
    "no stack samples collected": {
        "perf": _ok("perf script output"),
        "stackcollapse-perf.pl": _ok("   \n"),
        "flamegraph.pl": _ok("<svg>flames</svg>"),
    },
    "flamegraph.pl fails": {
        "perf": _ok("perf script output"),
        "stackcollapse-perf.pl": _ok(_SAMPLES),
        "flamegraph.pl": _failed("ERROR: unknown option: --colors\n"),
    },
    "flamegraph.pl output is empty": {
        "perf": _ok("perf script output"),
        "stackcollapse-perf.pl": _ok(_SAMPLES),
        "flamegraph.pl": _ok("   \n"),
    },
}


@pytest.mark.parametrize("outcomes", list(_FAILURES.values()), ids=list(_FAILURES))
def test_failed_generation_drops_the_svg_of_an_earlier_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcomes: Dict[str, subprocess.CompletedProcess],
) -> None:
    # The output directory is reused between runs, so a run that generates no
    # flamegraph must not leave the previous one behind to be read as its own.
    collector = _collector(tmp_path)
    _fake_commands(monkeypatch, outcomes)

    svg = tmp_path / "flamegraph.svg"
    svg.write_text("<svg>an earlier run</svg>")
    collector.generate_flamegraph(svg)

    assert not svg.exists()


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
    # The unusable profile is dropped before it can reach either step below.
    assert [call.step for call in seen] == ["perf"]
    err = capsys.readouterr().err
    assert "perf script on demo.perf.data failed (exit 1)" in err
    assert "Permission denied" in err
