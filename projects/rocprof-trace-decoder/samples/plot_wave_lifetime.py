#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import (
    add_common_args,
    decode_traces,
    instruction_text,
    load_inputs,
    prepare_output_dir,
    wave_idle_time,
)
from rocprof_trace_decoder import InstCategory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot wave lifetime against waitcnt, non-VALU, VALU, and idle latency."
    )
    add_common_args(parser)
    args = parser.parse_args()

    inputs, artifacts = load_inputs(args)
    traces = decode_traces(
        inputs.att_paths,
        code_index=artifacts.code_index,
    )

    rows = wave_lifetime_rows(traces, artifacts)
    out_dir = prepare_output_dir(args.dir)
    plot_path = out_dir / "wave_lifetime.png"

    plot_wave_lifetime(rows, plot_path)
    print(f"Wrote {plot_path}")
    return 0


def wave_lifetime_rows(traces, artifacts) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for trace in traces:
        for wave in trace.records.waves:
            waitcnt_latency = 0
            non_valu_latency = 0
            valu_latency = 0
            for inst in wave.instructions:
                is_valu = int(inst.category) == int(InstCategory.VALU)
                text = instruction_text(artifacts, inst.pc).strip()
                is_wait = text.startswith("s_wait")
                if is_wait:
                    waitcnt_latency += inst.duration
                elif is_valu:
                    valu_latency += inst.duration
                else:
                    non_valu_latency += inst.duration

            rows.append(
                {
                    "lifetime": max(wave.end_time - wave.begin_time, 0),
                    "waitcnt_latency": waitcnt_latency,
                    "non_valu_latency": non_valu_latency,
                    "valu_latency": valu_latency,
                    "idle_time": wave_idle_time(wave),
                }
            )
    return rows


def plot_wave_lifetime(rows: list[dict[str, int]], output_path) -> None:
    if not rows:
        raise SystemExit("No wave records were decoded.")

    import matplotlib.pyplot as plt
    import numpy as np

    datasets = [
        ("waitcnt_latency", "s_wait latency"),
        ("non_valu_latency", "non-VALU latency"),
        ("valu_latency", "VALU latency"),
        ("idle_time", "idle time"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    for ax, (field, label) in zip(axes.flat, datasets):
        x = np.array([row[field] for row in rows], dtype=float)
        y = np.array([row["lifetime"] for row in rows], dtype=float)
        ax.scatter(x, y, alpha=0.6, s=16, color="#ff0000", label="waves")
        if len(np.unique(x)) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            x0, x1 = x.min(), x.max()
            ax.plot(
                [x0, x1],
                [slope * x0 + intercept, slope * x1 + intercept],
                color="#003366",
                linestyle="--",
                linewidth=2,
                label="linear fit",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Wave lifetime")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Wave lifetime vs latency and idle time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
