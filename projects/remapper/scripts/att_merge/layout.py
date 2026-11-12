#!/usr/bin/env python3
"""Arrange rocprofv3's per-pass thread trace output of a remapped dispatch into the layout the merge reads.

Usage: layout.py DISPATCH_DIR ROCPROF_OUTPUT_DIR [--merge] [--no-sequence-check]

DISPATCH_DIR is the remapper's `dispatch-<id>/` (manifest.json, report.json, passes.json). rocprofv3 decodes each
traced pass into `ui_output_agent_<A>_dispatch_<D>_pass_<P>/` somewhere under ROCPROF_OUTPUT_DIR. Every accepted
traced pass becomes `DISPATCH_DIR/att/<run>-traces/sweep-NNN/cu-CC-simd-S-attempt-0/ui_output_agent_<A>_dispatch_1`
(a symlink), and every discarded (drifted) one goes under `<run>-traces/rejected-drift/`. With --merge, run.py then
merges the table run into `DISPATCH_DIR/analysis/` against the natural and original runs that exist.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UI = re.compile(r"ui_output_agent_(\d+)_dispatch_(\d+)_pass_(\d+)$")


def find_pass_dirs(root, dispatch_id):
    """pass -> (agent, path) for every decoded pass of `dispatch_id` under `root`."""
    found = {}
    for p in root.rglob("ui_output_agent_*_dispatch_*_pass_*"):
        m = UI.match(p.name)
        if not m or not p.is_dir() or int(m[2]) != dispatch_id:
            continue
        pass_index = int(m[3])
        if pass_index in found:
            raise ValueError(
                f"pass {pass_index} of dispatch {dispatch_id} decoded twice: {found[pass_index][1]} and {p}"
            )
        found[pass_index] = (int(m[1]), p)
    return found


def check_target(ui, record):
    """The pass was traced where the remapper asked; rocprofv3 writes replay_target.json next to each pass."""
    target = ui / "replay_target.json"
    if not target.exists():
        raise ValueError(
            f"{ui}: no replay_target.json; rocprofv3 did not record the pass's trace target"
        )
    got = json.loads(target.read_text())
    want = record["trace"]
    if (got["target_cu"], got["simd_select"]) != (want["target_cu"], want["simd"]):
        raise ValueError(
            f"{ui}: traced cu={got['target_cu']} simd={got['simd_select']}, "
            f"the remapper planned cu={want['target_cu']} simd={want['simd']}"
        )


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=True)


def arrange(dispatch_dir, rocprof_dir):
    """Build the trace trees; returns {run name: traces dir}."""
    passes = json.loads((dispatch_dir / "passes.json").read_text())
    dispatch_id = passes["dispatch_id"]
    decoded = find_pass_dirs(rocprof_dir, dispatch_id)
    runs = {}
    missing = []
    for record in passes["passes"]:
        run = record.get("run")
        if run is None:
            continue
        if record["pass"] not in decoded:
            if record["accepted"]:
                missing.append(record["pass"])
            continue
        agent, ui = decoded[record["pass"]]
        check_target(ui, record)
        traces = dispatch_dir / "att" / f"{run}-traces"
        runs[run] = traces
        cu, simd, sweep = (
            record["trace"]["target_cu"],
            record["trace"]["simd"],
            record["sweep"] + 1,
        )
        cell = f"cu-{cu:02d}-simd-{simd}"
        if record["accepted"]:
            dst = traces / f"sweep-{sweep:03d}" / f"{cell}-attempt-0"
        else:
            dst = (
                traces
                / "rejected-drift"
                / f"sweep-{sweep:03d}-{cell}-try-{record['attempt']}"
            )
        link(ui, dst / f"ui_output_agent_{agent}_dispatch_1")
    if missing:
        raise ValueError(
            f"traced passes {missing} of dispatch {dispatch_id} have no decoded output under {rocprof_dir}"
        )
    return runs


def merge(dispatch_dir, runs, sequence_check=True):
    report = json.loads((dispatch_dir / "report.json").read_text())
    if report.get("status") != "ok":
        raise ValueError(
            f"the remap of {dispatch_dir} did not succeed: {report.get('reason', report.get('status'))}"
        )
    if "table" not in runs:
        raise ValueError("no traced passes of the remapped kernel")
    model = report["collection"]["launch_model"]
    cmd = [
        sys.executable,
        str(HERE / "run.py"),
        "--manifest",
        str(dispatch_dir / "manifest.json"),
        "--traces",
        str(runs["table"]),
        "--out",
        str(dispatch_dir / "analysis"),
    ]
    if "natural" in runs:
        cmd += ["--baseline", str(runs["natural"])]
    if "original" in runs:
        cmd += ["--original", str(runs["original"])]
    if model == "sparse":
        cmd.append("--sparse")
    if model == "admission":
        cmd.append("--admission")
    if not sequence_check:
        cmd.append("--no-sequence-check")
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dispatch_dir", type=Path)
    ap.add_argument("rocprof_dir", type=Path)
    ap.add_argument(
        "--merge",
        action="store_true",
        help="run the merge and audits after arranging the traces",
    )
    ap.add_argument(
        "--no-sequence-check",
        action="store_true",
        help="accept traced waves whose instruction sequences differ between sweeps",
    )
    args = ap.parse_args()
    runs = arrange(args.dispatch_dir, args.rocprof_dir)
    for name, path in sorted(runs.items()):
        print(f"remapper: {name} traces in {path}")
    if args.merge:
        sys.exit(merge(args.dispatch_dir, runs, not args.no_sequence_check))


if __name__ == "__main__":
    main()
