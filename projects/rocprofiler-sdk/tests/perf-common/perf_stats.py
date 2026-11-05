#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Shared sampling and reporting helpers for the profiler performance regression tests.
#
# CI runners are shared, containerized and multi-tenant, so a single timed run is not a
# measurement. Callers here take several samples after a warmup and compare medians, and the
# absolute cost-model ceilings are advisory by default -- only the relative checks gate CI.

import json
import os
import statistics
from pathlib import Path

_FALSY = frozenset(("", "0", "false", "no", "off"))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in _FALSY


def parse_marker(text: str, tag: str) -> dict:
    """Parse a `[tag] key=value key=value ...` line from workload output.

    Split into key/value pairs rather than matched against a fixed regex, so that adding a
    field to a workload's marker line cannot silently stop the parse from matching -- which
    would turn a reported number into a missing-marker failure at the far end of a CI run.
    """
    prefix = f"[{tag}]"
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
                fields[key] = float(value) if "." in value else int(value)
            except ValueError:
                fields[key] = value

        if "wall_ms" in fields:
            return fields

    raise AssertionError(f"missing {prefix} wall_ms marker in output")


def strict_ceilings() -> bool:
    """Whether breaching an absolute cost-model ceiling should fail rather than warn.

    Off by default. An absolute wall-time bound on a shared runner says more about the
    neighbours than about the code, so enforcing it turns unrelated load into a red PR. Nightly
    runs on a known machine, and local runs, can turn it back on to get the stricter check.
    """
    return _env_flag("ROCPROFILER_PERF_STRICT_CEILING")


def repeat_measure(run_once, repeat: int, warmup: int = 1, label: str = "") -> dict:
    """Sample a measurement several times and summarize it.

    `run_once` is a callable returning milliseconds. Warmup results are discarded: the first
    run pays for ballast page faults, profiler attach and code-object load, and none of those
    repeat. Reports the median rather than the mean so one descheduled run cannot move it.
    """
    # Silently clamping a bad repeat count to 1 would turn "this suite forgot to ask for
    # repetition" into a single-sample measurement that still reports as if it were sampled.
    assert repeat >= 1, f"repeat must be at least 1, got {repeat}"
    assert warmup >= 0, f"warmup cannot be negative, got {warmup}"

    for _ in range(warmup):
        run_once()

    samples = [run_once() for _ in range(repeat)]
    low, high = min(samples), max(samples)
    stats = {
        "samples_ms": samples,
        "runs": len(samples),
        "median_ms": statistics.median(samples),
        "min_ms": low,
        "max_ms": high,
        # Spread is reported rather than asserted on: a noisy sample set makes any conclusion
        # from these numbers weaker, and that is worth seeing in the log even when the test
        # passes.
        "spread": (high / low) if low > 0 else float("inf"),
    }
    if label:
        print(
            f"[perf] {label} median={stats['median_ms']:.1f} ms over {len(samples)} runs "
            f"(min={low:.1f} max={high:.1f} spread={stats['spread']:.2f}x)"
        )
    return stats


def check_ceiling(value_ms: float, ceiling_ms: float, label: str) -> bool:
    """Absolute cost-model check. Advisory unless strict_ceilings() is on."""
    if value_ms <= ceiling_ms:
        print(
            f"[perf] {label} {value_ms:.1f} ms <= cost-model ceiling {ceiling_ms:.1f} ms"
        )
        return True

    msg = f"{label} {value_ms:.1f} ms exceeds cost-model ceiling {ceiling_ms:.1f} ms"
    if strict_ceilings():
        raise AssertionError(msg)
    print(
        f"[perf] WARNING: {msg} "
        "(advisory; set ROCPROFILER_PERF_STRICT_CEILING=1 to enforce)"
    )
    return False


def write_results(env_var: str, payload: dict):
    """Write a results JSON when `env_var` names a path, and return where it went.

    The environment variable has to be checked as a string before it becomes a Path:
    `Path("")` is `PosixPath('.')`, which is truthy, so testing the Path instead makes an unset
    variable write into the current working directory.
    """
    target = os.environ.get(env_var, "").strip()
    if not target:
        return None

    out_path = Path(target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[perf] wrote results to {out_path}")
    return out_path
