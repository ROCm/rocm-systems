#!/usr/bin/env python3
"""
Interactive (plotly) scatter + range plot of RCCL kernel dispatch times.

X-axis: message size (categorical)
Y-axis: kernel dispatch time (us), log scale
Shows individual points, median line, P25-P75 and min-max range bands.
Separate subplots for out-of-place and in-place.  Outputs an interactive
HTML file (requires plotly).

Profiling data comes from:
    mpirun -np 8 rocprofv3 --kernel-trace --marker-trace -d <outdir> -f csv -- \\
        ./all_reduce_perf -b 8 -e 1G -f 2 -g 1

Usage:
    python plot_kernel_scatter.py --profile <rocprofv3-output-dir> \\
                                  --output rccl_scatter.html
"""

import argparse
import csv
import os
import glob
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict


def find_rank0_files(directory):
    kernel_files = sorted(glob.glob(os.path.join(directory, '*_kernel_trace.csv')))
    marker_files = sorted(glob.glob(os.path.join(directory, '*_marker_api_trace.csv')))
    if not kernel_files or not marker_files:
        kernel_files = sorted(glob.glob(os.path.join(directory, '*', '*_kernel_trace.csv')))
        marker_files = sorted(glob.glob(os.path.join(directory, '*', '*_marker_api_trace.csv')))
    if not kernel_files:
        raise FileNotFoundError(f"No kernel_trace.csv found in {directory}")
    if not marker_files:
        raise FileNotFoundError(f"No marker_api_trace.csv found in {directory}")
    return kernel_files[0], marker_files[0]


def parse_markers(marker_file):
    markers = []
    with open(marker_file) as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
        for row in reader:
            func = row['Function']
            if not func.startswith('size='):
                continue
            parts = func.split()
            size = int(parts[0].split('=')[1])
            inplace = int(parts[1].split('=')[1])
            start = int(row['Start_Timestamp'])
            end = int(row['End_Timestamp'])
            markers.append((size, inplace, start, end))
    return markers


def parse_kernels(kernel_file):
    kernels = []
    with open(kernel_file) as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
        for row in reader:
            name = row['Kernel_Name']
            start = int(row['Start_Timestamp'])
            end = int(row['End_Timestamp'])
            kernels.append((name, start, end, end - start))
    kernels.sort(key=lambda x: x[1])
    return kernels


def correlate(markers, kernels):
    rccl_prefixes = ('ncclDevKernel', 'mscclKernel')
    result = defaultdict(list)
    kidx = 0
    for size, inplace, m_start, m_end in markers:
        while kidx < len(kernels) and kernels[kidx][1] < m_start:
            kidx += 1
        j = kidx
        while j < len(kernels) and kernels[j][1] <= m_end:
            name, k_start, k_end, dur_ns = kernels[j]
            if any(name.startswith(p) for p in rccl_prefixes):
                result[(size, inplace)].append(dur_ns / 1000.0)
            j += 1
    return result


def human_size(nbytes):
    if nbytes < 1024:
        return f"{nbytes}B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes // 1024}KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes // (1024 * 1024)}MB"
    else:
        return f"{nbytes / (1024**3):.0f}GB"


def make_plot(data, output_path):
    # Separate by inplace
    oop_sizes = sorted(set(s for s, ip in data if ip == 0))
    ip_sizes = sorted(set(s for s, ip in data if ip == 1))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Out-of-place', 'In-place'),
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    colors = {
        'oop': {'points': 'rgba(33, 150, 243, 0.4)', 'median': '#1565C0',
                'band': 'rgba(33, 150, 243, 0.15)', 'border': 'rgba(33, 150, 243, 0.4)'},
        'ip':  {'points': 'rgba(76, 175, 80, 0.4)', 'median': '#2E7D32',
                'band': 'rgba(76, 175, 80, 0.15)', 'border': 'rgba(76, 175, 80, 0.4)'},
    }

    for row, (sizes, label, col) in enumerate([
        (oop_sizes, 'oop', colors['oop']),
        (ip_sizes, 'ip', colors['ip']),
    ], start=1):
        if not sizes:
            continue

        medians = []
        mins = []
        maxs = []
        p25s = []
        p75s = []
        scatter_x = []
        scatter_y = []
        x_labels = []

        for s in sizes:
            vals = data.get((s, row - 1), [])
            if not vals:
                continue
            arr = np.array(vals)
            medians.append(np.median(arr))
            mins.append(np.min(arr))
            maxs.append(np.max(arr))
            p25s.append(np.percentile(arr, 25))
            p75s.append(np.percentile(arr, 75))
            x_labels.append(human_size(s))

            # Jitter points slightly for visibility
            jitter = np.random.uniform(-0.15, 0.15, len(arr))
            for jit, v in zip(jitter, arr):
                scatter_x.append(human_size(s))
                scatter_y.append(v)

        # Min-max range band (filled area between min and max)
        fig.add_trace(go.Scatter(
            x=x_labels + x_labels[::-1],
            y=maxs + mins[::-1],
            fill='toself',
            fillcolor=col['band'],
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=(row == 1),
            name='Min-Max range',
            hoverinfo='skip',
        ), row=row, col=1)

        # P25-P75 range band
        fig.add_trace(go.Scatter(
            x=x_labels + x_labels[::-1],
            y=p75s + p25s[::-1],
            fill='toself',
            fillcolor=col['border'],
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=(row == 1),
            name='P25-P75 range',
            hoverinfo='skip',
        ), row=row, col=1)

        # Individual points
        fig.add_trace(go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode='markers',
            marker=dict(size=4, color=col['points']),
            showlegend=(row == 1),
            name='Individual dispatches',
            hovertemplate='Size: %{x}<br>Time: %{y:.2f} us<extra></extra>',
        ), row=row, col=1)

        # Median line
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=medians,
            mode='lines+markers',
            line=dict(color=col['median'], width=2),
            marker=dict(size=6, color=col['median']),
            showlegend=(row == 1),
            name='Median',
            hovertemplate='Size: %{x}<br>Median: %{y:.2f} us<extra></extra>',
        ), row=row, col=1)

    fig.update_layout(
        title=dict(
            text='AllReduce Kernel Dispatch Latency — Custom RCCL (8 GPUs)',
            font=dict(size=16),
        ),
        height=900,
        width=1400,
        template='plotly_white',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=12),
        ),
        hovermode='x unified',
    )

    for row in [1, 2]:
        fig.update_yaxes(title_text='Kernel time (us)', row=row, col=1,
                         type='log',
                         dtick=None)
        fig.update_xaxes(
            title_text='Message size' if row == 2 else '',
            row=row, col=1,
            type='category',
            tickangle=-45,
            tickfont=dict(size=10),
        )

    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Saved interactive plot to {output_path}")

    # Also save a static image if kaleido is available
    png_path = output_path.replace('.html', '.png')
    try:
        fig.write_image(png_path, width=1400, height=900, scale=2)
        print(f"Saved static image to {png_path}")
    except Exception:
        print("(kaleido not available, skipping PNG export)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, help='rocprofv3 output directory')
    parser.add_argument('--output', default='/work/lmeadows/rccl_scatter.html',
                        help='Output HTML path')
    args = parser.parse_args()

    kf, mf = find_rank0_files(args.profile)
    print(f"Kernel trace: {kf}")
    print(f"Marker trace: {mf}")

    markers = parse_markers(mf)
    kernels = parse_kernels(kf)
    data = correlate(markers, kernels)

    print(f"Sizes: {len(set(s for s,_ in data))}  "
          f"Total dispatches: {sum(len(v) for v in data.values())}")

    make_plot(data, args.output)


if __name__ == '__main__':
    main()
