#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Plots allgather sweeps as an SVG: latency and bus bandwidth against total size.

    plot-allgather.py -o allgather.svg "xio=xio_r1.csv,xio_r2.csv" "HIP fused=fused_r1.csv"

Each argument is label=csv[,csv...]. CSVs hold the columns written by copy-engine-allgather
(`Size [B]`, `Device Latency [us] (Mean)`, `Bus Bandwidth [GiB/s]`). With several CSVs for one
label, one value per size is plotted: the median (default), which hides a stalled run when most
runs are clean, or with --statistic min the best latency and bandwidth of each size, which hides
stalls altogether. Needs only the Python standard library.
"""

import argparse
import csv
import math
import statistics
import sys

SIZE = "Size [B]"
LATENCY = "Device Latency [us] (Mean)"
BUS_BANDWIDTH = "Bus Bandwidth [GiB/s]"
COLORS = ["#222222", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["square", "circle", "triangle", "diamond", "circle", "square"]


def read_series(paths, statistic):
    """Latency and bus bandwidth per size over |paths|, sorted by size. |statistic| is "median"
    or "min" (the lowest latency and the highest bandwidth)."""
    values = {}
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                size = int(row[SIZE])
                values.setdefault(size, []).append((float(row[LATENCY]), float(row[BUS_BANDWIDTH])))
    if statistic == "min":
        return [
            (size, min(v[0] for v in samples), max(v[1] for v in samples))
            for size, samples in sorted(values.items())
        ]
    return [
        (
            size,
            statistics.median(v[0] for v in samples),
            statistics.median(v[1] for v in samples),
        )
        for size, samples in sorted(values.items())
    ]


def format_size(size):
    for unit, factor in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if size >= factor and size % factor == 0:
            return f"{size // factor}{unit}"
    return str(size)


def log_ticks(low, high):
    ticks = []
    exponent = math.floor(math.log10(low))
    while 10**exponent <= high:
        for mantissa in (1, 2, 5):
            tick = mantissa * 10**exponent
            if low <= tick <= high:
                ticks.append(tick)
        exponent += 1
    return ticks


def linear_ticks(high):
    step = 10 ** math.floor(math.log10(high))
    for candidate in (1, 2, 2.5, 5, 10):
        if high / (candidate * step) <= 8:
            step *= candidate
            break
    return [step * i for i in range(int(high / step) + 1)]


def marker(kind, x, y, color):
    if kind == "square":
        return f'<rect x="{x - 3:.1f}" y="{y - 3:.1f}" width="6" height="6" fill="{color}"/>'
    if kind == "triangle":
        return f'<polygon points="{x:.1f},{y - 4:.1f} {x - 4:.1f},{y + 3:.1f} {x + 4:.1f},{y + 3:.1f}" fill="{color}"/>'
    if kind == "diamond":
        return f'<polygon points="{x:.1f},{y - 4:.1f} {x + 4:.1f},{y:.1f} {x:.1f},{y + 4:.1f} {x - 4:.1f},{y:.1f}" fill="{color}"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}"/>'


def panel(left, top, width, height, title, ylabel, series, column, log_y, sizes):
    """One chart; |series| is [(label, rows)], |column| 1 for latency or 2 for bandwidth."""
    out = []
    x_low, x_high = math.log2(sizes[0]), math.log2(sizes[-1])
    ys = [row[column] for _, rows in series for row in rows]
    if log_y:
        y_low, y_high = min(ys) / 1.3, max(ys) * 1.3
        y_of = lambda v: math.log10(v)
        ticks = log_ticks(y_low, y_high)
        y_min, y_max = y_of(y_low), y_of(y_high)
    else:
        y_high = max(ys) * 1.05
        ticks = linear_ticks(y_high)
        y_of = lambda v: v
        y_min, y_max = 0.0, y_high

    def px(size):
        return left + (math.log2(size) - x_low) / (x_high - x_low) * width

    def py(value):
        return top + height - (y_of(value) - y_min) / (y_max - y_min) * height

    out.append(f'<text x="{left + width / 2}" y="{top - 12}" text-anchor="middle" font-size="15" font-weight="bold">{title}</text>')
    out.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#888"/>')
    for tick in ticks:
        y = py(tick)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#ddd"/>')
        label = f"{tick:g}"
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{label}</text>')
    for size in sizes:
        x = px(size)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" stroke="#eee"/>')
        out.append(f'<text x="{x:.1f}" y="{top + height + 16}" text-anchor="middle" font-size="10">{format_size(size)}</text>')
    out.append(f'<text x="{left + width / 2}" y="{top + height + 36}" text-anchor="middle" font-size="12">total size (bytes)</text>')
    out.append(f'<text transform="translate({left - 46},{top + height / 2}) rotate(-90)" text-anchor="middle" font-size="12">{ylabel}</text>')
    for index, (_, rows) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        points = " ".join(f"{px(row[0]):.1f},{py(row[column]):.1f}" for row in rows)
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        for row in rows:
            out.append(marker(MARKERS[index % len(MARKERS)], px(row[0]), py(row[column]), color))
    return out


def render(series):
    sizes = sorted({row[0] for _, rows in series for row in rows})
    width, height = 1180, 470
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    out += panel(70, 50, 480, 330, "Latency per collective", "time (us)", series, 1, True, sizes)
    out += panel(650, 50, 480, 330, "Bus bandwidth", "GiB/s", series, 2, False, sizes)
    x = 70
    for index, (label, _) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        out.append(f'<line x1="{x}" y1="{height - 18}" x2="{x + 26}" y2="{height - 18}" stroke="{color}" stroke-width="2"/>')
        out.append(marker(MARKERS[index % len(MARKERS)], x + 13, height - 18, color))
        out.append(f'<text x="{x + 34}" y="{height - 14}" font-size="12">{label}</text>')
        x += 40 + 7 * len(label) + 20
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-o", "--output", required=True, help="SVG file to write")
    parser.add_argument("--statistic", choices=["median", "min"], default="median",
                        help="how to combine the CSVs of one label (default median)")
    parser.add_argument("series", nargs="+", help="label=csv[,csv...]")
    args = parser.parse_args()

    series = []
    for argument in args.series:
        label, separator, paths = argument.partition("=")
        if not separator or not paths:
            sys.exit(f"error: '{argument}' is not label=csv[,csv...]")
        series.append((label, read_series(paths.split(","), args.statistic)))
    with open(args.output, "w") as handle:
        handle.write(render(series))


if __name__ == "__main__":
    main()
