#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Shared workload driver for the range replay perf regression tests.

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perf-common"))

from perf_stats import parse_marker  # noqa: E402


def run_case(
    testapp: Path,
    client: Path,
    passes: int,
    ballast_mb: int,
    dispatches: int,
    ranges: int,
    warmup: int,
) -> float:
    """One timed run of the workload. Returns the in-app wall time in milliseconds.

    Raises unless the tool confirmed every range was actually replayed. A declined range skips
    the snapshot, the pass loop and the restore, so it is far cheaper than a replayed one -- a
    regression that started declining these ranges would otherwise show up as a large speedup
    and pass every ceiling in this suite.
    """
    env = os.environ.copy()
    env["RR_PERF_PASSES"] = str(passes)
    preload = client.resolve()
    if env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = f"{preload}:{env['LD_PRELOAD']}"
    else:
        env["LD_PRELOAD"] = str(preload)

    proc = subprocess.run(
        [
            str(testapp.resolve()),
            str(ballast_mb),
            str(dispatches),
            str(ranges),
            str(warmup),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    label = f"P={passes} K={dispatches}"
    if proc.returncode != 0:
        raise RuntimeError(f"{label} run failed rc={proc.returncode}\n{out}")
    if "[rr-perf] PASS" not in out:
        raise RuntimeError(f"{label} missing PASS marker\n{out}")

    # The tool sees every range the application opened, warmup included, while only the last
    # `ranges` of them are inside the timed region.
    opened = ranges + warmup
    tool = parse_client_marker(out)
    if tool["ranges"] != opened or tool["replayed"] != opened:
        raise RuntimeError(
            f"{label} expected {opened} replayed range(s) ({ranges} timed + {warmup} warmup), "
            f"tool reported ranges={tool['ranges']} replayed={tool['replayed']}\n{out}"
        )
    if tool["dispatches"] != dispatches * opened:
        raise RuntimeError(
            f"{label} tool recorded {tool['dispatches']} dispatches, expected "
            f"{dispatches * opened}\n{out}"
        )

    return float(parse_marker(out, "rr-perf")["wall_ms"])


def parse_client_marker(text: str) -> dict:
    """Parse the tool's `[rr-perf-client] key=value ...` summary line."""
    prefix = "[rr-perf-client]"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        fields: dict = {}
        for token in stripped[len(prefix) :].split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            try:
                fields[key] = int(value)
            except ValueError:
                fields[key] = value
        if "replayed" in fields:
            return fields
    raise AssertionError(f"missing {prefix} summary in output")
