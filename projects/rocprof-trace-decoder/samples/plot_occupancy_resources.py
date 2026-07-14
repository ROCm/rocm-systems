#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import add_common_args, decode_traces, load_inputs, prepare_output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot active waves plus SGPR/VGPR allocation over time."
    )
    add_common_args(parser)
    args = parser.parse_args()

    inputs, artifacts = load_inputs(args)
    traces = decode_traces(
        inputs.att_paths,
        code_index=artifacts.code_index,
    )

    rows = occupancy_resource_rows(traces)
    out_dir = prepare_output_dir(args.dir)
    plot_path = out_dir / "occupancy_resources.png"

    plot_occupancy_resources(rows, plot_path)
    print(f"Wrote {plot_path}")
    return 0


def occupancy_resource_rows(traces) -> list[dict[str, int]]:
    events = []
    for trace_idx, trace in enumerate(traces):
        for dispatch in trace.records.dispatches:
            events.append((dispatch.time, 0, trace_idx, dispatch))
        for occupancy in trace.records.occupancy:
            events.append((occupancy.time, 1, trace_idx, occupancy))
    events.sort(key=lambda item: (item[0], item[1]))

    resources: dict[tuple[int, int, int], tuple[int, int]] = {}
    active_waves: dict[tuple[int, int, int, int, int, int, int], tuple[int, int]] = {}
    active_sgprs = 0
    active_vgprs = 0
    rows: list[dict[str, int]] = []

    for time, kind, trace_idx, record in events:
        if kind == 0:
            resources[(trace_idx, record.me_id, record.pipe_id)] = (record.sgprs, record.vgprs)
            continue

        key = (
            trace_idx,
            record.cu,
            record.simd,
            record.wave_id,
            record.me_id,
            record.pipe_id,
            record.workgroup_id,
        )
        sgprs, vgprs = resources.get((trace_idx, record.me_id, record.pipe_id), (0, 0))
        if record.start:
            if key not in active_waves:
                active_waves[key] = (sgprs, vgprs)
                active_sgprs += sgprs
                active_vgprs += vgprs
        elif key in active_waves:
            old_sgprs, old_vgprs = active_waves.pop(key)
            active_sgprs -= old_sgprs
            active_vgprs -= old_vgprs

        rows.append(
            {
                "time": int(time),
                "active_waves": len(active_waves),
                "active_sgprs": active_sgprs,
                "active_vgprs": active_vgprs,
            }
        )

    return rows


def plot_occupancy_resources(rows: list[dict[str, int]], output_path) -> None:
    if not rows:
        raise SystemExit("No occupancy records were decoded.")

    import matplotlib.pyplot as plt

    start = rows[0]["time"]
    x = [row["time"] - start for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    series = [
        ("active_waves", "Active waves"),
        ("active_sgprs", "Active SGPR allocation"),
        ("active_vgprs", "Active VGPR allocation"),
    ]
    for ax, (field, label) in zip(axes, series):
        ax.step(x, [row[field] for row in rows], where="post")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Trace time")
    fig.suptitle("Occupancy and register allocation over time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
