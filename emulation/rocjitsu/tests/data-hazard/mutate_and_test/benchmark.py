# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Wall-clock benchmarking and perf-based flamegraph collection for the mutation testing pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------


class Benchmarker:
    """Accumulates wall-clock timings for named kernel runs.

    Usage::

        bm = Benchmarker()
        with bm.benchmark("my_shader", benchmark_enabled=True):
            run_executable(...)
        bm.elapsed("my_shader")   # float seconds
        bm.timings                # ordered dict label -> seconds
    """

    def __init__(self) -> None:
        self._timings: Dict[str, float] = {}

    @contextmanager
    def benchmark(
        self, label: str, *, benchmark_enabled: bool
    ) -> Generator[None, None, None]:
        """Time the enclosed block when enabled; no-op otherwise."""
        if not benchmark_enabled:
            yield
            return
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._timings[label] = time.monotonic() - t0

    def elapsed(self, label: str) -> float:
        """Return elapsed seconds for *label*. Raises ``KeyError`` if not recorded."""
        return self._timings[label]

    def has(self, label: str) -> bool:
        """Return ``True`` if *label* was recorded."""
        return label in self._timings

    @property
    def timings(self) -> Dict[str, float]:
        """Ordered dict of all accumulated timings (label -> seconds)."""
        return dict(self._timings)

    @property
    def enabled(self) -> bool:
        """``True`` if any timings have been recorded."""
        return bool(self._timings)


# ---------------------------------------------------------------------------
# PerfCollector
# ---------------------------------------------------------------------------

_FLAMEGRAPH_SEARCH: List[str] = [
    "/opt/flamegraph",
    "/usr/share/flamegraph",
]


def _find_flamegraph_dir(override: Optional[Path]) -> Path:
    candidates: List[Optional[Path]] = []
    if override is not None:
        candidates.append(override)
    if "FLAMEGRAPH_DIR" in os.environ:
        candidates.append(Path(os.environ["FLAMEGRAPH_DIR"]))
    candidates.extend(Path(p) for p in _FLAMEGRAPH_SEARCH)

    for c in candidates:
        if c is None:
            continue
        if (c / "stackcollapse-perf.pl").is_file() and (c / "flamegraph.pl").is_file():
            return c

    searched = ", ".join(str(c) for c in candidates if c is not None)
    raise FileNotFoundError(
        f"FlameGraph scripts not found (searched: {searched}). "
        "Install to /opt/flamegraph or pass --flamegraph-dir. "
        "See: https://github.com/brendangregg/FlameGraph"
    )


@dataclass
class PerfCollector:
    """Wraps kernel runs with ``perf record`` and generates a cumulative flamegraph.

    Usage::

        pc = PerfCollector.discover(workdir)
        cmd, _ = pc.record_args("my_shader")
        subprocess.run(cmd + [str(exe)], ...)
        pc.generate_flamegraph(workdir / "flamegraph.svg")
    """

    workdir: Path
    perf: Path  # path to perf binary
    stackcollapse: Path  # path to stackcollapse-perf.pl
    flamegraph_pl: Path  # path to flamegraph.pl
    freq: int = 99  # sampling frequency (Hz)

    _data_files: List[Path] = field(default_factory=list, repr=False)

    @classmethod
    def discover(
        cls,
        workdir: Path,
        flamegraph_dir: Optional[Path] = None,
    ) -> "PerfCollector":
        """Locate perf and FlameGraph scripts; raise ``FileNotFoundError`` if missing."""
        perf_bin = shutil.which("perf")
        if perf_bin is None:
            raise FileNotFoundError(
                "perf not found. Install with: sudo apt-get install linux-tools-$(uname -r) linux-tools-common"
            )
        fg_dir = _find_flamegraph_dir(flamegraph_dir)
        return cls(
            workdir=workdir,
            perf=Path(perf_bin),
            stackcollapse=fg_dir / "stackcollapse-perf.pl",
            flamegraph_pl=fg_dir / "flamegraph.pl",
        )

    def record_args(self, label: str) -> Tuple[List[str], Path]:
        """Return ``(perf record prefix command, output .perf.data path)`` for *label*.

        The returned prefix must be prepended to the executable + its args when
        invoking subprocess.  The .perf.data path is registered internally so
        ``generate_flamegraph`` picks it up automatically.
        """
        out = self.workdir / f"{label}.perf.data"
        self._data_files.append(out)
        args = [
            str(self.perf),
            "record",
            "-g",
            "-F",
            str(self.freq),
            "--call-graph",
            "dwarf",  # full stacks through shared libs (plugin, HSA)
            "-q",
            "-o",
            str(out),
            "--",
        ]
        return args, out

    def generate_flamegraph(
        self,
        output_svg: Path,
        title: str = "Data Hazard Plugin — Cumulative",
    ) -> None:
        """Merge all collected perf.data files and write a cumulative flamegraph SVG."""
        present = [f for f in self._data_files if f.is_file()]
        if not present:
            print(
                "WARNING: no perf.data files found; flamegraph not generated.",
                flush=True,
            )
            return

        folded_parts: List[str] = []
        for data_file in present:
            script = subprocess.run(
                [str(self.perf), "script", "-i", str(data_file)],
                capture_output=True,
                text=True,
            )
            fold = subprocess.run(
                ["perl", str(self.stackcollapse)],
                input=script.stdout,
                capture_output=True,
                text=True,
            )
            if fold.stdout.strip():
                folded_parts.append(fold.stdout)

        if not folded_parts:
            print(
                "WARNING: no stack samples collected; flamegraph not generated.",
                flush=True,
            )
            return

        svg = subprocess.run(
            [
                "perl",
                str(self.flamegraph_pl),
                "--title",
                title,
                "--width",
                "1600",
                "--colors",
                "hot",
            ],
            input="\n".join(folded_parts),
            capture_output=True,
            text=True,
        )
        output_svg.write_text(svg.stdout)
        print(f"Flamegraph written to {output_svg}")
