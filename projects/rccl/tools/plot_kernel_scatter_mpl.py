#!/usr/bin/env python3
"""
Scatter + range plot of RCCL kernel dispatch times from rocprofv3 traces.

X-axis: message size
Y-axis: kernel dispatch time (us)
Shows individual points, median line, P5-P95 and IQR range bands.
Auto-segments by protocol transition (>60% median jump) with per-segment
linear Y scaling for readability.  Separate columns for out-of-place and
in-place.

Profiling data comes from:
    mpirun -np 8 rocprofv3 --kernel-trace --marker-trace -d <outdir> -f csv -- \\
        ./all_reduce_perf -b 8 -e 1G -f 2 -g 1

Usage:
    python plot_kernel_scatter_mpl.py --profile <rocprofv3-output-dir> \\
                                      --output /work/lmeadows/rccl_scatter.png
"""

import argparse
import csv
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def find_rank0_files(directory):
    kernel_files = sorted(glob.glob(os.path.join(directory, '*_kernel_trace.csv')))
    marker_files = sorted(glob.glob(os.path.join(directory, '*_marker_api_trace.csv')))
    if not kernel_files or not marker_files:
        kernel_files = sorted(glob.glob(os.path.join(directory, '*', '*_kernel_trace.csv')))
        marker_files = sorted(glob.glob(os.path.join(directory, '*', '*_marker_api_trace.csv')))
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
        return '%dB' % nbytes
    elif nbytes < 1024 * 1024:
        return '%dKB' % (nbytes // 1024)
    elif nbytes < 1024 * 1024 * 1024:
        return '%dMB' % (nbytes // (1024 * 1024))
    else:
        return '%dGB' % (nbytes // (1024 ** 3))


def compute_stats(data, sizes, ip):
    """Compute robust stats using P5/P95 for range instead of min/max."""
    medians, lo, hi, p25s, p75s, all_points = [], [], [], [], [], []
    for s in sizes:
        vals = data.get((s, ip), [])
        if vals:
            arr = np.array(vals)
            medians.append(np.median(arr))
            lo.append(np.percentile(arr, 5))
            hi.append(np.percentile(arr, 95))
            p25s.append(np.percentile(arr, 25))
            p75s.append(np.percentile(arr, 75))
            all_points.append(arr)
        else:
            medians.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
            p25s.append(np.nan)
            p75s.append(np.nan)
            all_points.append(np.array([]))
    return (np.array(medians), np.array(lo), np.array(hi),
            np.array(p25s), np.array(p75s), all_points)


def find_segments(medians):
    """Find protocol transition points where median time jumps significantly.
    
    Uses a two-pass approach: first find all jumps > 60%, then merge
    segments that are too small (< 3 sizes) into their neighbors.
    Only keeps the top few most significant transitions.
    """
    if len(medians) < 4:
        return []

    # Compute ratio of each point to the running median of the previous 2-3 points
    ratios = []
    for i in range(1, len(medians)):
        window = medians[max(0, i - 3):i]
        baseline = np.median(window[window > 0]) if np.any(window > 0) else 1
        if baseline > 0:
            ratios.append((i, medians[i] / baseline))
        else:
            ratios.append((i, 1.0))

    # Sort by ratio magnitude, pick the top transitions
    ratios.sort(key=lambda x: x[1], reverse=True)

    # Keep only transitions with ratio > 1.6 (60% jump)
    candidates = [(idx, r) for idx, r in ratios if r > 1.6]

    # Remove candidates that are too close together (within 2 indices)
    breaks = []
    for idx, r in candidates:
        if not breaks or all(abs(idx - b) > 2 for b in breaks):
            breaks.append(idx)
        if len(breaks) >= 3:
            break

    breaks.sort()
    return breaks


def draw_panel(ax, x_indices, sizes, medians, lo, hi, p25s, p75s, all_points,
               dark_color, light_color, title, breaks):
    # P5-P95 band
    ax.fill_between(x_indices, lo, hi, alpha=0.12, color=light_color,
                     label='P5–P95')
    # IQR band
    ax.fill_between(x_indices, p25s, p75s, alpha=0.30, color=light_color,
                     label='P25–P75')

    # Individual points with jitter
    for i, arr in enumerate(all_points):
        if len(arr):
            jitter = np.random.uniform(-0.2, 0.2, len(arr))
            ax.scatter(i + jitter, arr, s=10, alpha=0.45, color=light_color,
                       edgecolors='none', zorder=2)

    # Median line
    ax.plot(x_indices, medians, '-o', color=dark_color, linewidth=2.2,
            markersize=5, zorder=3, label='Median')

    # P5/P95 boundary lines
    ax.plot(x_indices, lo, '--', color=dark_color, linewidth=0.7, alpha=0.4)
    ax.plot(x_indices, hi, '--', color=dark_color, linewidth=0.7, alpha=0.4)

    # Protocol transition markers
    for bi in breaks:
        ax.axvline(bi - 0.5, color='#D32F2F', linewidth=1.5, linestyle='--',
                   alpha=0.7, zorder=4)

    ax.set_ylabel('Kernel time (us)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.25, which='both')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)


def make_plot(data, output_path):
    sizes = sorted(set(s for s, _ in data))
    x_indices = np.arange(len(sizes))
    x_labels = [human_size(s) for s in sizes]

    # Compute stats for both
    oop_stats = compute_stats(data, sizes, 0)
    ip_stats = compute_stats(data, sizes, 1)

    # Find segmentation from out-of-place medians (both should be similar)
    breaks = find_segments(oop_stats[0])

    # Determine segment boundaries for Y-scale grouping
    # Add 0 at start and len at end
    seg_edges = [0] + breaks + [len(sizes)]
    segments = []
    for i in range(len(seg_edges) - 1):
        segments.append((seg_edges[i], seg_edges[i + 1]))

    n_seg = len(segments)
    fig, axes = plt.subplots(n_seg, 2, figsize=(18, 4.5 * n_seg),
                             squeeze=False)
    fig.suptitle('AllReduce Kernel Dispatch Time — Custom RCCL (8x MI300X)',
                 fontsize=15, fontweight='bold', y=0.99)

    oop_colors = ('#1565C0', '#64B5F6')
    ip_colors = ('#2E7D32', '#66BB6A')

    for si, (s_start, s_end) in enumerate(segments):
        sl = slice(s_start, s_end)
        xi = x_indices[sl] 
        sz_range = '%s – %s' % (human_size(sizes[s_start]),
                                human_size(sizes[s_end - 1]))

        # Breaks that fall within this segment (for internal transitions)
        local_breaks = [b for b in breaks if s_start < b < s_end]

        for col, (stats, colors, place_label) in enumerate([
            (oop_stats, oop_colors, 'Out-of-place'),
            (ip_stats, ip_colors, 'In-place'),
        ]):
            ax = axes[si, col]
            meds, lo, hi, p25, p75, pts = [x[sl] for x in stats]

            draw_panel(ax, xi, sizes[s_start:s_end],
                       meds, lo, hi, p25, p75, pts,
                       colors[0], colors[1],
                       '%s  (%s)' % (place_label, sz_range),
                       local_breaks)

            ax.set_xticks(xi)
            ax.set_xticklabels(x_labels[s_start:s_end],
                               rotation=45, ha='right', fontsize=10)

            # Linear Y scale with tight limits
            valid = np.concatenate([a for a in pts if len(a)])
            if len(valid):
                p5 = np.percentile(valid, 2)
                p98 = np.percentile(valid, 98)
                margin = (p98 - p5) * 0.15
                ax.set_ylim(max(0, p5 - margin), p98 + margin)

    for ax in axes[-1]:
        ax.set_xlabel('Message size', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Saved %s' % output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', default='/work/lmeadows/rccl_scatter.png')
    args = parser.parse_args()

    kf, mf = find_rank0_files(args.profile)
    markers = parse_markers(mf)
    kernels = parse_kernels(kf)
    data = correlate(markers, kernels)

    sizes = sorted(set(s for s, _ in data))
    print('Sizes: %d  Dispatches: %d' % (len(sizes), sum(len(v) for v in data.values())))

    make_plot(data, args.output)


if __name__ == '__main__':
    main()
