#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

from marker_common import (
    add_marker_args,
    load_marker_traces,
    marker_name,
    marker_selected,
    marker_time,
)
from rocprof_trace_decoder import MarkerFlags, MarkerRecordKind


def folded_stacks(markers, args=None) -> dict[str, int]:
    states: dict[tuple[int, int, int, int], tuple[list[str], int]] = {}
    folded: dict[str, int] = defaultdict(int)

    for marker in sorted(markers, key=marker_time):
        if marker.record_kind != MarkerRecordKind.HEADER:
            continue
        if not marker_selected(marker, args):
            continue
        time = marker_time(marker)
        key = (
            marker.code_object_id,
            marker.shaderdata.cu,
            marker.shaderdata.simd,
            marker.shaderdata.wave_id,
        )
        stack, previous = states.get(key, ([], time))
        if marker.marker_flags & MarkerFlags.NEW_WAVE:
            stack = []
            previous = time

        if stack and time > previous:
            folded[";".join(stack)] += time - previous

        if marker.marker_flags & MarkerFlags.EXIT_PREVIOUS and stack:
            stack.pop()
        if marker.marker_flags & MarkerFlags.ENTER:
            stack.append(marker_name(marker, args))

        states[key] = (stack, time)
    return dict(folded)


def render_svg(folded: dict[str, int]) -> str:
    rows = sorted(folded.items(), key=lambda item: item[1], reverse=True)
    width = 1200
    row_height = 24
    height = max(60, 40 + row_height * len(rows))
    maximum = max((cycles for _, cycles in rows), default=1)
    body = []
    for index, (stack, cycles) in enumerate(rows):
        y = 28 + index * row_height
        bar_width = max(1, int((width - 360) * cycles / maximum))
        body.append(
            f'<rect x="350" y="{y - 14}" width="{bar_width}" height="18" fill="#e67e22"/>'
            f'<text x="8" y="{y}" font-size="13">{html.escape(stack)}</text>'
            f'<text x="{355 + bar_width}" y="{y}" font-size="12">{cycles}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<style>text{font-family:monospace}</style>'
        '<text x="8" y="18" font-size="16">SQTT marker stacks</text>'
        + "".join(body)
        + "</svg>"
    )


def speedscope(folded: dict[str, int]) -> dict:
    frame_ids = {}
    frames = []
    samples = []
    weights = []
    for stack, cycles in sorted(folded.items()):
        sample = []
        for frame in stack.split(";"):
            if frame not in frame_ids:
                frame_ids[frame] = len(frames)
                frames.append({"name": frame})
            sample.append(frame_ids[frame])
        samples.append(sample)
        weights.append(cycles)
    return {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": {"frames": frames},
        "profiles": [
            {
                "type": "sampled",
                "name": "SQTT markers",
                "unit": "none",
                "startValue": 0,
                "endValue": sum(weights),
                "samples": samples,
                "weights": weights,
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate folded stacks from decoded marker records."
    )
    add_marker_args(parser)
    parser.add_argument("--format", choices=("folded", "svg", "speedscope"), default="folded")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file; folded output defaults to stdout."
    )
    args = parser.parse_args()

    merged: dict[str, int] = defaultdict(int)
    for trace in load_marker_traces(args):
        for stack, cycles in folded_stacks(trace.records.markers, args).items():
            merged[stack] += cycles

    if args.format == "folded":
        output = "".join(f"{stack} {cycles}\n" for stack, cycles in sorted(merged.items()))
    elif args.format == "svg":
        output = render_svg(merged)
    else:
        output = json.dumps(speedscope(merged), indent=2)

    if args.output:
        args.output.write_text(output)
        print(f"Wrote {args.output}")
    else:
        print(output, end="" if output.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
