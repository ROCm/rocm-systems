# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Live progress on stderr, in the same section form as the final report."""

from __future__ import annotations

import sys
from typing import Sequence, TextIO

from bench.runner import RunConfig, RunResult
from bench.text_report import MIN_WIDTH, banner

BAR_WIDTH = 20
CLEAR_LINE = "\r\033[K"


def _bar(done: int, total: int) -> str:
    filled = round(BAR_WIDTH * done / total) if total else 0
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _label(name: str, role: str, name_w: int) -> str:
    return f"  {name.ljust(name_w)}  {role.ljust(8)}"


def _out(stream: TextIO | None) -> TextIO:
    return stream if stream is not None else sys.stderr


def role_of(index: int) -> str:
    return "baseline" if index == 0 else "tool"


def announce(
    commands: Sequence[tuple[str, str]], cfg: RunConfig, stream: TextIO | None = None
) -> None:
    name_w = max(len(n) for n, _ in commands)
    lines = [
        banner("RUNNING", MIN_WIDTH),
        (
            f"  {len(commands)} command(s), one after another. Each one: {cfg.warmup} warm-up run(s) "
            f"(not counted), then {cfg.runs} measured run(s)."
        ),
        "  Below, the bar shows how many measured runs of the current command are finished.",
        "  The report with the results follows when all commands are done.",
        "",
    ]
    lines += [
        f"{_label(n, role_of(i), name_w)}  {cmd}" for i, (n, cmd) in enumerate(commands)
    ]
    print("\n".join(lines) + "\n", file=_out(stream))


def warming_up(
    name: str, role: str, name_w: int, cfg: RunConfig, stream: TextIO | None = None
) -> None:
    out = _out(stream)
    if out.isatty():
        print(
            f"{CLEAR_LINE}{_label(name, role, name_w)}  warming up ({cfg.warmup} run(s), not counted)",
            end="",
            file=out,
            flush=True,
        )


def update(
    name: str,
    role: str,
    name_w: int,
    done: int,
    total: int,
    stream: TextIO | None = None,
) -> None:
    out = _out(stream)
    if out.isatty():
        print(
            f"{CLEAR_LINE}{_label(name, role, name_w)}  [{_bar(done, total)}] {done}/{total} runs",
            end="",
            file=out,
            flush=True,
        )


def finish(
    name: str,
    role: str,
    name_w: int,
    results: list[RunResult],
    total: int,
    stream: TextIO | None = None,
) -> None:
    failed = [r for r in results if not r.ok]
    status = "done" if not failed else f"{len(failed)} FAILED (first: {failed[0].error})"
    out = _out(stream)
    prefix = CLEAR_LINE if out.isatty() else ""
    print(
        f"{prefix}{_label(name, role, name_w)}  [{_bar(len(results), total)}] "
        f"{len(results)}/{total} runs  {status}",
        file=out,
    )
