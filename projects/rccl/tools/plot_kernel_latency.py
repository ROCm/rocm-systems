#!/usr/bin/env python3
"""
Plot box-and-whisker charts of RCCL kernel dispatch times from rocprofv3 traces.

Reads kernel_trace.csv and marker_api_trace.csv from rocprofv3 output directories,
correlates kernel dispatches to roctx marker ranges (size=N inplace={0,1}), and
produces per-size box plots comparing two builds.

Usage:
    python plot_kernel_latency.py --custom /tmp/rccl_prof/custom/hostname/
                                  --system /tmp/rccl_prof/system/hostname/
                                  --output /tmp/rccl_latency.png
"""

import argparse
import csv
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def find_rank0_files(directory):
    """Find the kernel_trace and marker_api_trace CSVs for rank 0 (first PID)."""
    kernel_files = sorted(glob.glob(os.path.join(directory, '*_kernel_trace.csv')))
    marker_files = sorted(glob.glob(os.path.join(directory, '*_marker_api_trace.csv')))
    if not kernel_files or not marker_files:
        # Try one level deeper (hostname subdirectory)
        kernel_files = sorted(glob.glob(os.path.join(directory, '*', '*_kernel_trace.csv')))
        marker_files = sorted(glob.glob(os.path.join(directory, '*', '*_marker_api_trace.csv')))
    if not kernel_files:
        raise FileNotFoundError(f"No kernel_trace.csv found in {directory}")
    if not marker_files:
        raise FileNotFoundError(f"No marker_api_trace.csv found in {directory}")
    return kernel_files[0], marker_files[0]


def parse_markers(marker_file):
    """Parse roctx marker ranges into list of (size, inplace, start_ts, end_ts)."""
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
    """Parse kernel dispatches into list of (name, start_ts, end_ts, duration_ns)."""
    kernels = []
    with open(kernel_file) as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
        for row in reader:
            name = row['Kernel_Name']
            start = int(row['Start_Timestamp'])
            end = int(row['End_Timestamp'])
            kernels.append((name, start, end, end - start))
    return kernels


def correlate(markers, kernels):
    """
    For each marker range, collect durations of RCCL kernel dispatches
    (ncclDevKernel or mscclKernel) that start within that range.
    Returns dict: (size, inplace) -> list of kernel durations in microseconds.
    """
    rccl_prefixes = ('ncclDevKernel', 'mscclKernel')
    result = defaultdict(list)

    kidx = 0
    for size, inplace, m_start, m_end in markers:
        # Advance kernel index to first kernel that could overlap
        while kidx < len(kernels) and kernels[kidx][1] < m_start:
            kidx += 1
        # Collect all RCCL kernels within this marker range
        j = kidx
        while j < len(kernels) and kernels[j][1] <= m_end:
            name, k_start, k_end, dur_ns = kernels[j]
            if any(name.startswith(p) for p in rccl_prefixes):
                result[(size, inplace)].append(dur_ns / 1000.0)  # ns -> us
            j += 1

    return result


def load_profile(directory):
    """Load and correlate a single profiling run."""
    kf, mf = find_rank0_files(directory)
    markers = parse_markers(mf)
    kernels = parse_kernels(kf)
    # Sort kernels by start timestamp for the sweep
    kernels.sort(key=lambda x: x[1])
    return correlate(markers, kernels)


def human_size(nbytes):
    """Format byte count as human-readable string."""
    if nbytes < 1024:
        return f"{nbytes}B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes // 1024}KB"
    else:
        return f"{nbytes // (1024 * 1024)}MB"


def make_plots(custom_data, system_data, output_path):
    """
    Create box-and-whisker plots. Split into multiple figures grouped by
    similar Y-scale ranges so bars are readable.
    """
    all_sizes = sorted(set(
        s for s, ip in list(custom_data.keys()) + list(system_data.keys())
    ))

    # Group sizes by approximate median latency range for readable Y-axes.
    # Compute median per size (across both builds, out-of-place).
    size_medians = {}
    for s in all_sizes:
        vals = custom_data.get((s, 0), []) + system_data.get((s, 0), [])
        if vals:
            size_medians[s] = np.median(vals)
        else:
            size_medians[s] = 0

    # Split into groups where max/min median ratio < ~4x
    groups = []
    current_group = []
    group_min = None
    for s in all_sizes:
        med = size_medians[s]
        if med == 0:
            continue
        if group_min is None:
            group_min = med
            current_group.append(s)
        elif med / group_min > 4.0 or len(current_group) >= 8:
            groups.append(current_group)
            current_group = [s]
            group_min = med
        else:
            current_group.append(s)
    if current_group:
        groups.append(current_group)

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 5 * n_groups),
                             squeeze=False)
    fig.suptitle('AllReduce Kernel Dispatch Latency: Custom vs System RCCL',
                 fontsize=14, fontweight='bold', y=0.98)

    colors_custom = {'oop': '#2196F3', 'ip': '#64B5F6'}
    colors_system = {'oop': '#FF5722', 'ip': '#FF8A65'}

    for gi, group_sizes in enumerate(groups):
        ax = axes[gi, 0]
        n = len(group_sizes)
        positions = []
        labels = []
        box_data = []
        box_colors = []

        for i, s in enumerate(group_sizes):
            base = i * 5  # spacing between size groups
            label = human_size(s)

            # Custom out-of-place
            d = custom_data.get((s, 0), [])
            positions.append(base + 0)
            box_data.append(d if d else [0])
            box_colors.append(colors_custom['oop'])

            # Custom in-place
            d = custom_data.get((s, 1), [])
            positions.append(base + 1)
            box_data.append(d if d else [0])
            box_colors.append(colors_custom['ip'])

            # System out-of-place
            d = system_data.get((s, 0), [])
            positions.append(base + 2)
            box_data.append(d if d else [0])
            box_colors.append(colors_system['oop'])

            # System in-place
            d = system_data.get((s, 1), [])
            positions.append(base + 3)
            box_data.append(d if d else [0])
            box_colors.append(colors_system['ip'])

            labels.append((base + 1.5, label))

        bp = ax.boxplot(box_data, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks([x for x, _ in labels])
        ax.set_xticklabels([l for _, l in labels], fontsize=11, fontweight='bold')
        ax.set_ylabel('Kernel time (us)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(positions[0] - 1, positions[-1] + 1)

        # Set Y limits based on whisker extents with some padding
        all_whisker_vals = [w.get_ydata().max() for w in bp['whiskers']] + \
                           [w.get_ydata().min() for w in bp['whiskers']]
        if all_whisker_vals:
            ymin = max(0, min(all_whisker_vals) * 0.85)
            ymax = max(all_whisker_vals) * 1.10
            ax.set_ylim(ymin, ymax)

        range_label = f"{human_size(group_sizes[0])} - {human_size(group_sizes[-1])}"
        ax.set_title(f'Message sizes: {range_label}', fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_custom['oop'], alpha=0.8, label='Custom out-of-place'),
        Patch(facecolor=colors_custom['ip'], alpha=0.8, label='Custom in-place'),
        Patch(facecolor=colors_system['oop'], alpha=0.8, label='System out-of-place'),
        Patch(facecolor=colors_system['ip'], alpha=0.8, label='System in-place'),
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper left', fontsize=10,
                      ncol=2, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Also print summary table
    print(f"\n{'Size':>8s}  {'Custom OOP':>12s}  {'Custom IP':>12s}  "
          f"{'System OOP':>12s}  {'System IP':>12s}  {'N(cust)':>8s}  {'N(sys)':>8s}")
    print("-" * 90)
    for s in all_sizes:
        def fmt(data, key):
            d = data.get(key, [])
            if not d:
                return "N/A", 0
            return f"{np.median(d):8.2f}", len(d)
        c_oop, nc_oop = fmt(custom_data, (s, 0))
        c_ip, nc_ip = fmt(custom_data, (s, 1))
        s_oop, ns_oop = fmt(system_data, (s, 0))
        s_ip, ns_ip = fmt(system_data, (s, 1))
        print(f"{human_size(s):>8s}  {c_oop:>12s}  {c_ip:>12s}  "
              f"{s_oop:>12s}  {s_ip:>12s}  "
              f"{nc_oop + nc_ip:>8d}  {ns_oop + ns_ip:>8d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--custom', required=True, help='rocprofv3 output dir for custom RCCL')
    parser.add_argument('--system', required=True, help='rocprofv3 output dir for system RCCL')
    parser.add_argument('--output', default='/tmp/rccl_latency.png', help='Output PNG path')
    args = parser.parse_args()

    print("Loading custom RCCL profile...")
    custom_data = load_profile(args.custom)
    print(f"  {sum(len(v) for v in custom_data.values())} kernel dispatches across "
          f"{len(custom_data)} size/inplace combos")

    print("Loading system RCCL profile...")
    system_data = load_profile(args.system)
    print(f"  {sum(len(v) for v in system_data.values())} kernel dispatches across "
          f"{len(system_data)} size/inplace combos")

    make_plots(custom_data, system_data, args.output)


if __name__ == '__main__':
    main()
