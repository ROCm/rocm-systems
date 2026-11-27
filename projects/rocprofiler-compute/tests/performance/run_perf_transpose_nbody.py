#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Time profile (all blocks except roofline) + analyze for Fibonacci workload.

Pass --rocprof-compute once (or omit for this tree) to print timings. Pass it
two or more times to print a comparison table with deltas vs the first path.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTS_DIR = Path(__file__).resolve().parents[1]
_APP_PATH = Path(__file__).resolve().parent / "fib_transpose_nbody_app.py"
_DEFAULT_ROCPROF = _REPO_ROOT / "src" / "rocprof-compute"
_WORKLOAD_NAME = "fib_transpose_nbody"
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import common  # noqa: E402

from utils.utils_common import (  # noqa: E402
    PC_SAMPLING_BLOCK_IDS,
    ROOFLINE_BLOCK_IDS,
    get_arch_panel_id_to_alias,
)


@dataclass
class PhaseTiming:
    profile_s: float
    analyze_s: float
    total_s: float
    label: str
    workload_dir: Path


def resolve_rocprof_compute_command(path: Path) -> List[str]:
    """Map a file or project root to an argv prefix for rocprof-compute."""
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        src_entry = resolved / "src" / "rocprof-compute"
        if src_entry.is_file():
            return [sys.executable, str(src_entry)]
        for relative in ("bin/rocprof-compute", "install/bin/rocprof-compute"):
            candidate = resolved / relative
            if candidate.is_file():
                return [str(candidate)]
        raise FileNotFoundError(f"no rocprof-compute under directory: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"rocprof-compute path not found: {resolved}")
    if resolved.name == "rocprof-compute" and resolved.parent.name == "src":
        return [sys.executable, str(resolved)]
    return [str(resolved)]


def label_for_path(path: Path, instance_index: int) -> str:
    """Stable unique label: index, leaf name, and a short path digest."""
    resolved = path.expanduser().resolve()
    if resolved.is_file() and resolved.parent.name == "src":
        leaf = resolved.parents[1].name
    elif resolved.is_dir():
        leaf = resolved.name
    else:
        leaf = resolved.name
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{instance_index:02d}_{leaf}_{digest}"


def find_workload_dir(search_root: Path) -> Path:
    """Return the directory containing sysinfo.csv under search_root."""
    matches = sorted(search_root.rglob("sysinfo.csv"))
    if not matches:
        raise FileNotFoundError(f"no sysinfo.csv under {search_root} after profile")
    return matches[0].parent


def detect_gpu_arch() -> str:
    """Return the first gfx* Name from rocminfo."""
    completed = subprocess.run(
        ["rocminfo"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"rocminfo failed (rc={completed.returncode})")
    for line in completed.stdout.splitlines():
        match = re.search(r"^\s*Name:\s+(gfx[0-9a-zA-Z]+)", line)
        if match:
            return match.group(1).lower()
    raise RuntimeError("could not detect GPU arch from rocminfo")


def non_roofline_block_ids(arch: str) -> List[str]:
    """Top-level block ids for arch, excluding roof / PC sampling / membw."""
    panels = get_arch_panel_id_to_alias(arch)
    if not panels:
        raise RuntimeError(f"no analysis config template for arch {arch}")
    excluded = set(ROOFLINE_BLOCK_IDS) | set(PC_SAMPLING_BLOCK_IDS) | {"30"}
    block_ids = [
        panel_id
        for panel_id in panels
        if panel_id not in excluded and not panel_id.startswith("4.")
    ]
    return sorted(block_ids, key=lambda panel_id: int(panel_id))


def run_timed(
    command: List[str],
    env: Dict[str, str],
    cwd: Path,
) -> Tuple[float, int]:
    start = time.perf_counter()
    completed = subprocess.run(command, env=env, cwd=str(cwd), check=False)
    elapsed = time.perf_counter() - start
    return elapsed, completed.returncode


def run_instance(
    rocprof_path: Path,
    work_parent: Path,
    instance_index: int,
    iters: int,
    bodies: int,
    matrix_dim: int,
    device: int,
    experimental: bool,
    no_native_tool: bool,
) -> PhaseTiming:
    label = label_for_path(rocprof_path, instance_index)
    instance_dir = work_parent / label
    instance_dir.mkdir(parents=True, exist_ok=True)

    rocprof_cmd = resolve_rocprof_compute_command(rocprof_path)
    # Do not prepend this checkout's src onto PYTHONPATH: an installed
    # entrypoint must load its own libexec modules, not this tree's.
    env = os.environ.copy()

    experimental_args = ["--experimental"] if experimental else []
    native_tool_args = ["--no-native-tool"] if no_native_tool else []
    block_ids = non_roofline_block_ids(detect_gpu_arch())
    profile_cmd = [
        *rocprof_cmd,
        "profile",
        *experimental_args,
        *native_tool_args,
        "-b",
        *block_ids,
        "--overwrite",
        "-n",
        _WORKLOAD_NAME,
        "-d",
        str(instance_dir),
        "--",
        sys.executable,
        str(_APP_PATH),
        "--iters",
        str(iters),
        "--bodies",
        str(bodies),
        "--matrix-dim",
        str(matrix_dim),
        "--device",
        str(device),
    ]
    profile_s, profile_rc = run_timed(profile_cmd, env=env, cwd=_REPO_ROOT)
    if profile_rc != 0:
        raise RuntimeError(
            f"profile failed (rc={profile_rc}) for {label}: {profile_cmd}"
        )

    workload_dir = find_workload_dir(instance_dir)
    analyze_cmd = [
        *rocprof_cmd,
        "analyze",
        *experimental_args,
        "-p",
        str(workload_dir),
        "-q",
    ]
    analyze_s, analyze_rc = run_timed(analyze_cmd, env=env, cwd=_REPO_ROOT)
    if analyze_rc != 0:
        raise RuntimeError(
            f"analyze failed (rc={analyze_rc}) for {label}: {analyze_cmd}"
        )

    return PhaseTiming(
        profile_s=profile_s,
        analyze_s=analyze_s,
        total_s=profile_s + analyze_s,
        label=label,
        workload_dir=workload_dir,
    )


def print_single(timing: PhaseTiming) -> None:
    print(f"rocprof-compute: {timing.label}")
    print(f"  workload: {timing.workload_dir}")
    print(f"  profile_s: {timing.profile_s:.3f}")
    print(f"  analyze_s: {timing.analyze_s:.3f}")
    print(f"  total_s:   {timing.total_s:.3f}")


def print_comparison(timings: List[PhaseTiming]) -> None:
    baseline = timings[0]
    headers = ["metric", baseline.label]
    for timing in timings[1:]:
        headers.append(timing.label)
        headers.append(f"d vs {baseline.label}")
        headers.append(f"% vs {baseline.label}")

    rows = []  # type: List[List[str]]
    for metric_name, attr in (
        ("profile_s", "profile_s"),
        ("analyze_s", "analyze_s"),
        ("total_s", "total_s"),
    ):
        row = [metric_name, f"{getattr(baseline, attr):.3f}"]
        base_value = getattr(baseline, attr)
        for timing in timings[1:]:
            value = getattr(timing, attr)
            delta = value - base_value
            pct = (delta / base_value * 100.0) if base_value else 0.0
            row.append(f"{value:.3f}")
            row.append(f"{delta:+.3f}")
            row.append(f"{pct:+.1f}%")
        rows.append(row)

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    print(_format_table_row(headers, widths))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(_format_table_row(row, widths))


def _format_table_row(cells: List[str], widths: List[int]) -> str:
    return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Performance harness: HipRTC Fibonacci tiled transpose+nbody "
            "profile (all blocks except roofline) + analyze"
        )
    )
    parser.add_argument(
        "--rocprof-compute",
        action="append",
        type=Path,
        default=None,
        help=(
            "Path to rocprof-compute entrypoint or project root "
            "(repeatable; default: this tree)"
        ),
    )
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--bodies", type=int, default=81920)
    parser.add_argument(
        "--matrix-dim",
        type=int,
        default=32768,
        help=(
            "Square matrix side for tiled transpose "
            "(default: 32768, ~8 GiB for two buffers)"
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="HIP device id passed to the app (default: 0)",
    )
    parser.add_argument(
        "-d",
        "--work-dir",
        type=Path,
        default=None,
        help="Parent directory for workload output",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Retain workload directories after the run",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Pass --experimental to profile and analyze (e.g. gfx1201)",
    )
    parser.add_argument(
        "--no-native-tool",
        action="store_true",
        help="Pass --no-native-tool to profile (skip native counter tool build)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    common.check_resource_allocation()
    paths = args.rocprof_compute or [_DEFAULT_ROCPROF]
    created_tmp = False
    if args.work_dir is None:
        work_parent = Path(f"/tmp/rocprof-compute-perf-{os.getpid()}")
        created_tmp = True
    else:
        work_parent = args.work_dir.expanduser().resolve()
    work_parent.mkdir(parents=True, exist_ok=True)

    timings = []  # type: List[PhaseTiming]
    try:
        for instance_index, path in enumerate(paths):
            print(f"=== running {path} ===", flush=True)
            timings.append(
                run_instance(
                    path,
                    work_parent=work_parent,
                    instance_index=instance_index,
                    iters=args.iters,
                    bodies=args.bodies,
                    matrix_dim=args.matrix_dim,
                    device=args.device,
                    experimental=args.experimental,
                    no_native_tool=args.no_native_tool,
                )
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if created_tmp and not args.keep:
            shutil.rmtree(work_parent, ignore_errors=True)

    if len(timings) == 1:
        print_single(timings[0])
    else:
        print_comparison(timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
