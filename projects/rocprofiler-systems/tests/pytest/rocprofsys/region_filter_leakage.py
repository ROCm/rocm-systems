# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Perfetto validators for events recorded outside ROCPROFSYS_SELECTED_REGIONS windows."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from .validators import ValidationResult

# Default margin for boundary alignment (nanoseconds).
_DEFAULT_TOLERANCE_NS = 1_000_000  # 1 ms

# pthread gotcha APIs (see pthread_mutex_gotcha / pthread_create_gotcha).
PTHREAD_NAME_GLOBS = (
    "pthread_create",
    "pthread_join",
    "pthread_mutex_*",
    "pthread_rwlock_*",
    "pthread_spin_*",
    "pthread_barrier_*",
)

# Kernels are validated by name elsewhere; skip in the time-window pass.
_KERNEL_NAME_GLOB = "CodeBlock_*"


@dataclass(frozen=True)
class LeakageCheck:
    """One category of slices to verify against selected region time windows."""

    label: str
    categories: tuple[str, ...]
    name_globs: tuple[str, ...] = ()
    exclude_name_globs: tuple[str, ...] = ()
    max_outside_slices: int = 0


DEFAULT_LEAKAGE_CHECKS: tuple[LeakageCheck, ...] = (
    LeakageCheck(
        label="pthread APIs",
        categories=("pthread",),
        name_globs=PTHREAD_NAME_GLOBS,
    ),
    LeakageCheck(
        label="HIP / ROCm API",
        categories=("rocm_hip_api", "rocm_hip_stream"),
        exclude_name_globs=(_KERNEL_NAME_GLOB,),
    ),
    LeakageCheck(
        label="kernel dispatch",
        categories=("rocm_kernel_dispatch",),
    ),
    LeakageCheck(
        label="thread / timer sampling",
        categories=("timer_sampling", "process_sampling"),
    ),
    LeakageCheck(
        label="filtered-out ROCTx markers",
        categories=("rocm_marker_api",),
    ),
)

COUNTER_TRACK_GLOBS = (
    "device_busy_*",
    "device_temp*",
    "device_power*",
    "device_memory_usage*",
)


def parse_selected_regions(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [r.strip() for r in value.split(",") if r.strip()]
    return [r.strip() for r in value if r.strip()]


def _open_trace_processor(trace_path: Path):
    from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig

    env_path = os.environ.get("ROCPROFSYS_TRACE_PROC_SHELL")
    config = TraceProcessorConfig(bin_path=env_path) if env_path else None
    if config:
        return TraceProcessor(trace=str(trace_path), config=config)
    return TraceProcessor(trace=str(trace_path))


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _fetch_region_intervals(tp, selected_regions: list[str]) -> list[tuple[int, int]]:
    if not selected_regions:
        return []

    region_list = ", ".join(f"'{r}'" for r in selected_regions)
    query = f"""
        SELECT ts, dur
        FROM slice
        WHERE category = 'rocm_marker_api' AND name IN ({region_list})
        ORDER BY ts
    """
    intervals = []
    for row in tp.query(query):
        start = int(row.ts)
        dur = int(row.dur) if row.dur is not None else 0
        end = start + dur if dur > 0 else start + 1
        intervals.append((start, end))
    return _merge_intervals(intervals)


def _slice_fully_inside(
    start: int,
    end: int,
    allowed: list[tuple[int, int]],
    tolerance_ns: int,
) -> bool:
    if not allowed:
        return False
    for win_start, win_end in allowed:
        if start >= win_start - tolerance_ns and end <= win_end + tolerance_ns:
            return True
    return False


def _build_name_predicate(
    name_globs: tuple[str, ...], exclude_globs: tuple[str, ...]
) -> str:
    parts = []
    if name_globs:
        glob_expr = " OR ".join(f"name GLOB '{g}'" for g in name_globs)
        parts.append(f"({glob_expr})")
    for g in exclude_globs:
        parts.append(f"name NOT GLOB '{g}'")
    if not parts:
        return "1"
    return " AND ".join(parts)


def _check_slices_outside_regions(
    tp,
    check: LeakageCheck,
    allowed_intervals: list[tuple[int, int]],
    selected_regions: list[str],
    tolerance_ns: int,
    max_report: int = 8,
) -> list[str]:
    cat_list = ", ".join(f"'{c}'" for c in check.categories)
    name_pred = _build_name_predicate(check.name_globs, check.exclude_name_globs)

    extra = ""
    if check.label == "filtered-out ROCTx markers" and selected_regions:
        allowed = ", ".join(f"'{r}'" for r in selected_regions)
        extra = f" AND name NOT IN ({allowed}) AND name GLOB 'Region*'"

    query = f"""
        SELECT name, category, ts, dur
        FROM slice
        WHERE category IN ({cat_list}) AND {name_pred}{extra}
        ORDER BY ts
        LIMIT 500
    """

    violations: list[str] = []
    for row in tp.query(query):
        start = int(row.ts)
        dur = int(row.dur) if row.dur is not None else 0
        end = start + dur if dur > 0 else start + 1
        if _slice_fully_inside(start, end, allowed_intervals, tolerance_ns):
            continue
        violations.append(
            f"  - [{row.category}] {row.name} @ {start}..{end} (dur={dur})"
        )
        if len(violations) >= max_report:
            violations.append("  - ... (truncated)")
            break
    return violations


def _check_counters_outside_regions(
    tp,
    allowed_intervals: list[tuple[int, int]],
    tolerance_ns: int,
    max_outside: int,
    max_report: int = 5,
) -> list[str]:
    glob_expr = " OR ".join(f"name GLOB '{g}'" for g in COUNTER_TRACK_GLOBS)
    query = f"""
        SELECT ct.name, c.ts
        FROM counter c
        JOIN counter_track ct ON c.track_id = ct.id
        WHERE {glob_expr}
        ORDER BY c.ts
        LIMIT 2000
    """
    outside = 0
    total = 0
    samples: list[str] = []
    for row in tp.query(query):
        total += 1
        ts = int(row.ts)
        if _slice_fully_inside(ts, ts + 1, allowed_intervals, tolerance_ns):
            continue
        outside += 1
        if len(samples) < max_report:
            samples.append(f"  - {row.name} @ {ts}")
    if outside <= max_outside:
        return []
    lines = [
        f"  GPU counter samples outside selected regions: {outside}/{total} "
        f"(allowed <= {max_outside}; may reflect sampling rate vs short regions)"
    ]
    lines.extend(samples)
    if outside > max_report:
        lines.append("  - ... (truncated)")
    return lines


def validate_region_filter_leakage(
    trace_path: Path,
    selected_regions: str | list[str],
    checks: Optional[tuple[LeakageCheck, ...]] = None,
    *,
    check_counters: bool = True,
    counter_max_outside: int = 0,
    tolerance_ns: int = _DEFAULT_TOLERANCE_NS,
) -> ValidationResult:
    """Fail when traced slices fall outside union of selected ROCTx region windows."""
    if not trace_path.exists():
        return ValidationResult(False, f"Trace file not found: {trace_path}")

    regions = parse_selected_regions(selected_regions)
    if not regions:
        return ValidationResult(False, "No selected regions provided")

    try:
        tp = _open_trace_processor(trace_path)
    except ImportError as exc:
        return ValidationResult(False, f"perfetto Python bindings not available: {exc}")
    except Exception as exc:
        return ValidationResult(False, f"Perfetto open failed: {exc}")

    allowed = _fetch_region_intervals(tp, regions)
    if not allowed:
        return ValidationResult(
            False,
            f"No rocm_marker_api intervals found for regions {regions!r}",
        )

    check_list = checks if checks is not None else DEFAULT_LEAKAGE_CHECKS
    sections: list[str] = [
        f"Selected regions: {', '.join(regions)}",
        f"Allowed time windows ({len(allowed)} merged interval(s)) from rocm_marker_api",
    ]

    failed = False
    for check in check_list:
        violations = _check_slices_outside_regions(
            tp, check, allowed, regions, tolerance_ns
        )
        if len(violations) > check.max_outside_slices:
            failed = True
            sections.append(f"{check.label} outside region windows:")
            sections.extend(violations)

    if check_counters:
        counter_lines = _check_counters_outside_regions(
            tp, allowed, tolerance_ns, counter_max_outside
        )
        if counter_lines:
            failed = True
            sections.append("GPU PMC / AMD-SMI counters:")
            sections.extend(counter_lines)

    if failed:
        return ValidationResult(False, "\n".join(sections))

    return ValidationResult(
        True,
        "No leakage outside selected region windows for configured checks",
    )


def validate_pthread_outside_region_filter(
    trace_path: Path,
    api_names: tuple[str, ...] = ("pthread_join", "pthread_create"),
    timeout: int = 120,
) -> ValidationResult:
    """Backward-compatible strict pthread count check (any pthread slice fails)."""
    del timeout  # unused; kept for API compatibility
    if not trace_path.exists():
        return ValidationResult(False, f"Trace file not found: {trace_path}")

    try:
        tp = _open_trace_processor(trace_path)
    except ImportError as exc:
        return ValidationResult(False, f"perfetto Python bindings not available: {exc}")
    except Exception as exc:
        return ValidationResult(False, f"Perfetto query failed: {exc}")

    name_filters = " OR ".join(f"name GLOB '*{n}*'" for n in api_names)
    query = f"""
        SELECT name, COUNT(*) AS cnt
        FROM slice
        WHERE category = 'pthread' AND ({name_filters})
        GROUP BY name
        ORDER BY name
    """
    rows = list(tp.query(query))
    if not rows:
        return ValidationResult(
            True,
            "No pthread join/create slices found (expected with region filter)",
        )

    lines = [
        "pthread APIs recorded while ROCPROFSYS_SELECTED_REGIONS is set "
        "(expected none outside active regions):"
    ]
    for row in rows:
        lines.append(f"  - {row.name}: {row.cnt}")
    return ValidationResult(False, "\n".join(lines))
