#!/usr/bin/env python3
"""
Bayesian changepoint detection and piecewise model fitting for RCCL kernel
dispatch latency data.  Generates a multi-page PDF presentation:

  Page 1: Full-range overview with segmented regression
  Pages 2-N: Per-segment detail with linear Y, IQR bands, model equation
  Final page: Box-and-whisker for noisy sizes (CV > threshold)
"""

import argparse
import csv
import glob
import os
import numpy as np
from scipy import stats as sp_stats
from collections import defaultdict
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


# ── Data loading ─────────────────────────────────────────────────────────────

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
    elif nbytes < 1024**2:
        return '%dKB' % (nbytes // 1024)
    elif nbytes < 1024**3:
        return '%dMB' % (nbytes // 1024**2)
    else:
        return '%dGB' % (nbytes // 1024**3)


# ── Outlier rejection ────────────────────────────────────────────────────────

def reject_outliers(vals):
    """Remove points beyond 3*IQR from median."""
    if len(vals) < 4:
        return vals
    med = np.median(vals)
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    fence = max(iqr * 3, med * 0.5)
    return vals[np.abs(vals - med) <= fence]


# ── Segment model fitting ───────────────────────────────────────────────────

def fit_constant(sizes_bytes, times_us):
    n = len(times_us)
    c = np.mean(times_us)
    ss = np.sum((times_us - c)**2)
    sigma2 = ss / n if n > 0 else 1e-10
    k = 2
    bic = n * np.log(sigma2 + 1e-30) + k * np.log(n) if n > 0 else 1e30
    return dict(model='constant', params=dict(c=c), sigma=np.sqrt(sigma2),
                bic=bic, k=k, n=n)


def fit_affine(sizes_bytes, times_us):
    n = len(times_us)
    if n < 3:
        return fit_constant(sizes_bytes, times_us)
    X = np.column_stack([np.ones(n), sizes_bytes])
    beta, _, _, _ = np.linalg.lstsq(X, times_us, rcond=None)
    pred = X @ beta
    ss = np.sum((times_us - pred)**2)
    sigma2 = ss / n
    k = 3
    bic = n * np.log(sigma2 + 1e-30) + k * np.log(n)
    bw = 1.0 / (beta[1] * 1e6) if abs(beta[1]) > 1e-20 else np.inf
    return dict(model='affine', params=dict(a=beta[0], b=beta[1], bw_GBs=bw),
                sigma=np.sqrt(sigma2), bic=bic, k=k, n=n)


def fit_quadratic(sizes_bytes, times_us):
    n = len(times_us)
    if n < 4:
        return fit_affine(sizes_bytes, times_us)
    X = np.column_stack([np.ones(n), sizes_bytes, sizes_bytes**2])
    beta, _, _, _ = np.linalg.lstsq(X, times_us, rcond=None)
    pred = X @ beta
    ss = np.sum((times_us - pred)**2)
    sigma2 = ss / n
    k = 4
    bic = n * np.log(sigma2 + 1e-30) + k * np.log(n)
    return dict(model='quadratic', params=dict(a=beta[0], b=beta[1], c=beta[2]),
                sigma=np.sqrt(sigma2), bic=bic, k=k, n=n)


def fit_log(sizes_bytes, times_us):
    n = len(times_us)
    if n < 3:
        return fit_constant(sizes_bytes, times_us)
    log_s = np.log(sizes_bytes + 1)
    X = np.column_stack([np.ones(n), log_s])
    beta, _, _, _ = np.linalg.lstsq(X, times_us, rcond=None)
    pred = X @ beta
    ss = np.sum((times_us - pred)**2)
    sigma2 = ss / n
    k = 3
    bic = n * np.log(sigma2 + 1e-30) + k * np.log(n)
    return dict(model='log', params=dict(a=beta[0], b=beta[1]),
                sigma=np.sqrt(sigma2), bic=bic, k=k, n=n)


def best_segment_fit(sizes_bytes, times_us):
    candidates = [
        fit_constant(sizes_bytes, times_us),
        fit_affine(sizes_bytes, times_us),
        fit_quadratic(sizes_bytes, times_us),
        fit_log(sizes_bytes, times_us),
    ]
    return min(candidates, key=lambda x: x['bic'])


def predict(fit, sizes_bytes):
    p = fit['params']
    if fit['model'] == 'constant':
        return np.full_like(sizes_bytes, p['c'], dtype=float)
    elif fit['model'] == 'affine':
        return p['a'] + p['b'] * sizes_bytes
    elif fit['model'] == 'quadratic':
        return p['a'] + p['b'] * sizes_bytes + p['c'] * sizes_bytes**2
    elif fit['model'] == 'log':
        return p['a'] + p['b'] * np.log(sizes_bytes + 1)
    return np.full_like(sizes_bytes, np.nan, dtype=float)


def model_equation(fit):
    p = fit['params']
    if fit['model'] == 'constant':
        return 't = %.2f us' % p['c']
    elif fit['model'] == 'affine':
        bw = p.get('bw_GBs', 0)
        if abs(bw) < 1e-3 or abs(bw) > 1e6:
            return 't = %.2f + %.4g · size  (us)' % (p['a'], p['b'])
        return 't = %.1f us + size / %.2f GB/s' % (p['a'], bw)
    elif fit['model'] == 'quadratic':
        return 't = %.2f + %.2g·s + %.2g·s²  (us)' % (p['a'], p['b'], p['c'])
    elif fit['model'] == 'log':
        return 't = %.2f + %.3f · ln(size)  (us)' % (p['a'], p['b'])
    return fit['model']


# ── Changepoint search ───────────────────────────────────────────────────────

def evaluate_segmentation(size_indices, sizes_bytes, times_us, breakpoints, n_unique):
    edges = [0] + list(breakpoints) + [n_unique]
    total_bic = 0
    fits = []
    for i in range(len(edges) - 1):
        mask = (size_indices >= edges[i]) & (size_indices < edges[i + 1])
        if np.sum(mask) < 2:
            total_bic += 1e10
            fits.append(None)
            continue
        fit = best_segment_fit(sizes_bytes[mask], times_us[mask])
        total_bic += fit['bic']
        fits.append(fit)
    total_bic += len(breakpoints) * np.log(len(times_us))
    return total_bic, fits


def search_changepoints(unique_sizes, size_indices, sizes_bytes, times_us,
                         n_cp_range=(2, 3)):
    n_sizes = len(unique_sizes)
    print('\n--- Bayesian Changepoint Search (BIC) ---')
    print('Unique sizes: %d   Total observations: %d' % (n_sizes, len(times_us)))

    results = {}
    for n_cp in range(n_cp_range[0], n_cp_range[1] + 1):
        best = None
        for combo in combinations(range(2, n_sizes - 1), n_cp):
            bic, fits = evaluate_segmentation(size_indices, sizes_bytes,
                                               times_us, list(combo), n_sizes)
            if best is None or bic < best[0]:
                best = (bic, list(combo), fits)
        results[n_cp] = best
        bps = best[1]
        bp_labels = [human_size(unique_sizes[b]) for b in bps]
        print('  %d changepoint(s):  BIC = %.1f   breaks at %s' % (
            n_cp, best[0], bp_labels))

    # Pick best across the range
    best_ncp = min(results, key=lambda k: results[k][0])
    return results[best_ncp]


# ── Plotting ─────────────────────────────────────────────────────────────────

SEG_COLORS = ['#1565C0', '#E65100', '#2E7D32', '#7B1FA2']


def page_overview(pdf, sizes, raw_data, edges, fits):
    """Page 1: Full-range overview with log Y scale."""
    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('AllReduce Kernel Dispatch Time — Custom RCCL, 8x MI300X',
                 fontsize=15, fontweight='bold')

    x_all = np.arange(len(sizes))
    n_seg = len(edges) - 1

    for si in range(n_seg):
        s0, s1 = edges[si], edges[si + 1]
        color = SEG_COLORS[si % len(SEG_COLORS)]
        fit = fits[si]

        # Scatter cleaned points
        for ii in range(s0, s1):
            vals = reject_outliers(np.array(raw_data.get((sizes[ii], 0), [])))
            if len(vals):
                jitter = np.random.uniform(-0.15, 0.15, len(vals))
                ax_main.scatter(ii + jitter, vals, s=12, alpha=0.35,
                                color=color, edgecolors='none', zorder=2)

        # Median markers
        meds = [np.median(raw_data.get((sizes[ii], 0), [0])) for ii in range(s0, s1)]
        ax_main.plot(range(s0, s1), meds, 'o', color=color, markersize=5,
                     zorder=3, alpha=0.6)

        # Fitted curve
        seg_sizes = np.array([float(sizes[ii]) for ii in range(s0, s1)])
        curve_x = np.linspace(s0, s1 - 1, 200)
        curve_sizes = np.interp(curve_x, range(s0, s1), seg_sizes)
        curve_y = predict(fit, curve_sizes)
        eq = model_equation(fit)
        label = 'Seg %d (%s–%s): %s' % (si + 1, human_size(sizes[s0]),
                                          human_size(sizes[s1 - 1]), eq)
        ax_main.plot(curve_x, curve_y, '-', color=color, linewidth=2.5,
                     zorder=4, label=label)

        # Residuals
        for ii in range(s0, s1):
            vals = reject_outliers(np.array(raw_data.get((sizes[ii], 0), [])))
            if len(vals):
                pred_v = predict(fit, np.full(len(vals), float(sizes[ii])))
                resid = vals - pred_v
                jitter = np.random.uniform(-0.15, 0.15, len(vals))
                ax_resid.scatter(ii + jitter, resid, s=8, alpha=0.4,
                                 color=color, edgecolors='none')

    # Changepoint lines
    for bp in edges[1:-1]:
        for ax in [ax_main, ax_resid]:
            ax.axvline(bp - 0.5, color='#D32F2F', linewidth=1.5,
                       linestyle='--', alpha=0.7, zorder=5)

    ax_main.set_xticks(x_all)
    ax_main.set_xticklabels([human_size(s) for s in sizes], rotation=45,
                             ha='right', fontsize=8)
    ax_main.set_ylabel('Kernel time (us)', fontsize=12)
    ax_main.set_yscale('log')
    ax_main.grid(True, alpha=0.2, which='both')
    ax_main.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
    ax_main.set_xlim(-0.5, len(sizes) - 0.5)

    ax_resid.axhline(0, color='black', linewidth=0.8)
    ax_resid.set_xticks(x_all)
    ax_resid.set_xticklabels([human_size(s) for s in sizes], rotation=45,
                              ha='right', fontsize=8)
    ax_resid.set_ylabel('Residual (us)', fontsize=11)
    ax_resid.set_xlabel('Message size', fontsize=11)
    ax_resid.grid(True, alpha=0.2)
    ax_resid.set_xlim(-0.5, len(sizes) - 0.5)
    ax_resid.set_title('Residuals (data − model)', fontsize=10, loc='left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_segment_detail(pdf, sizes, raw_data, s0, s1, fit, seg_idx):
    """Per-segment detail page with linear Y, IQR bands, annotations."""
    seg_sizes = sizes[s0:s1]
    n = len(seg_sizes)
    x = np.arange(n)
    color = SEG_COLORS[seg_idx % len(SEG_COLORS)]
    light = color + '40'  # with alpha

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})

    eq = model_equation(fit)
    fig.suptitle('Segment %d: %s – %s    [%s]' % (
        seg_idx + 1, human_size(seg_sizes[0]), human_size(seg_sizes[-1]),
        fit['model'].capitalize()),
        fontsize=14, fontweight='bold')

    # Compute stats per size
    meds, p25s, p75s, p5s, p95s = [], [], [], [], []
    all_clean = []
    for s in seg_sizes:
        vals = reject_outliers(np.array(raw_data.get((s, 0), [])))
        all_clean.append(vals)
        if len(vals):
            meds.append(np.median(vals))
            p25s.append(np.percentile(vals, 25))
            p75s.append(np.percentile(vals, 75))
            p5s.append(np.percentile(vals, 5))
            p95s.append(np.percentile(vals, 95))
        else:
            meds.append(np.nan)
            p25s.append(np.nan)
            p75s.append(np.nan)
            p5s.append(np.nan)
            p95s.append(np.nan)

    meds = np.array(meds)
    p25s = np.array(p25s)
    p75s = np.array(p75s)
    p5s = np.array(p5s)
    p95s = np.array(p95s)

    # P5-P95 band
    ax_top.fill_between(x, p5s, p95s, alpha=0.10, color=color, label='P5–P95')
    # IQR band
    ax_top.fill_between(x, p25s, p75s, alpha=0.25, color=color, label='P25–P75 (IQR)')

    # Individual points
    for i, vals in enumerate(all_clean):
        if len(vals):
            jitter = np.random.uniform(-0.2, 0.2, len(vals))
            ax_top.scatter(i + jitter, vals, s=16, alpha=0.4, color=color,
                           edgecolors='none', zorder=2)

    # Median line
    ax_top.plot(x, meds, 'o-', color=color, linewidth=2, markersize=6,
                zorder=3, label='Median')

    # Model fit curve
    seg_sizes_f = np.array([float(s) for s in seg_sizes])
    curve_x = np.linspace(0, n - 1, 200)
    curve_sizes = np.interp(curve_x, x, seg_sizes_f)
    curve_y = predict(fit, curve_sizes)
    ax_top.plot(curve_x, curve_y, '--', color='black', linewidth=2, alpha=0.7,
                zorder=4, label='Fit: %s' % eq)

    # Annotate median values
    for i, (m, s) in enumerate(zip(meds, seg_sizes)):
        if not np.isnan(m):
            ax_top.annotate('%.1f' % m, (i, m), textcoords='offset points',
                            xytext=(0, 10), fontsize=7, ha='center', color='#333',
                            fontweight='bold')

    ax_top.set_xticks(x)
    ax_top.set_xticklabels([human_size(s) for s in seg_sizes], fontsize=10,
                            fontweight='bold')
    ax_top.set_ylabel('Kernel time (us)', fontsize=12)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(fontsize=10, framealpha=0.95)

    # Tight Y limits: a bit beyond the P5-P95 range
    valid_lo = np.nanmin(p5s)
    valid_hi = np.nanmax(p95s)
    margin = (valid_hi - valid_lo) * 0.2
    ax_top.set_ylim(max(0, valid_lo - margin), valid_hi + margin)

    # Stats text box
    sigma = fit['sigma']
    stats_text = 'Model: %s\n%s\nσ = %.2f us' % (fit['model'], eq, sigma)
    if fit['model'] == 'affine':
        bw = fit['params'].get('bw_GBs', 0)
        if 0.001 < abs(bw) < 1e6:
            stats_text += '\nBandwidth: %.2f GB/s' % bw
            stats_text += '\nBase latency: %.1f us' % fit['params']['a']
    ax_top.text(0.98, 0.97, stats_text, transform=ax_top.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='#ccc', alpha=0.95),
                family='monospace')

    # Bottom: residuals
    for i, vals in enumerate(all_clean):
        if len(vals):
            pred_v = predict(fit, np.full(len(vals), float(seg_sizes[i])))
            resid = vals - pred_v
            jitter = np.random.uniform(-0.2, 0.2, len(vals))
            ax_bot.scatter(i + jitter, resid, s=14, alpha=0.5, color=color,
                           edgecolors='none')

    ax_bot.axhline(0, color='black', linewidth=0.8)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([human_size(s) for s in seg_sizes], fontsize=10,
                            fontweight='bold')
    ax_bot.set_ylabel('Residual (us)', fontsize=11)
    ax_bot.set_xlabel('Message size', fontsize=11)
    ax_bot.grid(True, alpha=0.25)
    ax_bot.set_title('Residuals (data − model)', fontsize=10, loc='left')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_noisy_boxwhisker(pdf, sizes, raw_data, cv_threshold=8.0):
    """Box-and-whisker detail for high-variance sizes."""
    # Identify noisy sizes
    noisy = []
    for s in sizes:
        vals = np.array(raw_data.get((s, 0), []))
        if len(vals) < 3:
            continue
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        cv = 100 * np.std(vals) / med if med > 0 else 0
        # Also check IQR relative to median
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        iqr_pct = 100 * iqr / med if med > 0 else 0
        if cv > cv_threshold or iqr_pct > 5:
            noisy.append((s, vals, med, cv, iqr_pct, mad))

    if not noisy:
        return

    # Group into pages of max 10 sizes
    page_size = 10
    for pg_start in range(0, len(noisy), page_size):
        chunk = noisy[pg_start:pg_start + page_size]
        n = len(chunk)

        fig, ax = plt.subplots(figsize=(max(12, n * 1.5), 7))
        fig.suptitle('Kernel Time Variability — High-CV Sizes (out-of-place)',
                     fontsize=14, fontweight='bold')

        box_data_raw = []
        box_data_clean = []
        labels = []
        for s, vals, med, cv, iqr_pct, mad in chunk:
            clean = reject_outliers(vals)
            box_data_raw.append(vals)
            box_data_clean.append(clean)
            labels.append('%s\nmed=%.1f\nCV=%.0f%%\nMAD=%.1f' % (
                human_size(s), med, cv, mad))

        positions_raw = np.arange(n) * 3
        positions_clean = positions_raw + 1

        bp_raw = ax.boxplot(box_data_raw, positions=positions_raw, widths=0.6,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker='.', markersize=4, alpha=0.4),
                            medianprops=dict(color='black', linewidth=1.5))
        for patch in bp_raw['boxes']:
            patch.set_facecolor('#BBDEFB')
            patch.set_alpha(0.7)

        bp_clean = ax.boxplot(box_data_clean, positions=positions_clean, widths=0.6,
                              patch_artist=True, showfliers=True,
                              flierprops=dict(marker='.', markersize=4, alpha=0.4),
                              medianprops=dict(color='black', linewidth=1.5))
        for patch in bp_clean['boxes']:
            patch.set_facecolor('#C8E6C9')
            patch.set_alpha(0.7)

        # Individual points overlaid
        for i, (vals_r, vals_c) in enumerate(zip(box_data_raw, box_data_clean)):
            jitter_r = np.random.uniform(-0.15, 0.15, len(vals_r))
            ax.scatter(positions_raw[i] + jitter_r, vals_r, s=10, alpha=0.3,
                       color='#1565C0', edgecolors='none', zorder=2)
            jitter_c = np.random.uniform(-0.15, 0.15, len(vals_c))
            ax.scatter(positions_clean[i] + jitter_c, vals_c, s=10, alpha=0.3,
                       color='#2E7D32', edgecolors='none', zorder=2)

        ax.set_xticks(positions_raw + 0.5)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Kernel time (us)', fontsize=12)
        ax.grid(axis='y', alpha=0.25)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#BBDEFB', alpha=0.7, label='Raw (all points)'),
            Patch(facecolor='#C8E6C9', alpha=0.7, label='Cleaned (outliers removed)'),
        ], fontsize=10, loc='upper right')

        # Zoom Y to show the IQR detail — set limits from cleaned data
        all_clean = np.concatenate(box_data_clean)
        q1 = np.percentile(all_clean, 5)
        q3 = np.percentile(all_clean, 95)
        span = q3 - q1
        ax.set_ylim(max(0, q1 - span * 0.5), q3 + span * 0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # If there are outliers way out of range, add an unzoomed version
        all_raw = np.concatenate(box_data_raw)
        if np.max(all_raw) > q3 + span * 2:
            fig2, ax2 = plt.subplots(figsize=(max(12, n * 1.5), 5))
            fig2.suptitle('Same data — full Y range showing outliers',
                         fontsize=12, fontstyle='italic')
            bp2 = ax2.boxplot(box_data_raw, positions=positions_raw, widths=0.6,
                              patch_artist=True, showfliers=True,
                              flierprops=dict(marker='x', markersize=5,
                                              color='red', alpha=0.6),
                              medianprops=dict(color='black', linewidth=1.5))
            for patch in bp2['boxes']:
                patch.set_facecolor('#BBDEFB')
                patch.set_alpha(0.7)
            ax2.set_xticks(positions_raw)
            ax2.set_xticklabels([human_size(s) for s, *_ in chunk], fontsize=9)
            ax2.set_ylabel('Kernel time (us)', fontsize=11)
            ax2.grid(axis='y', alpha=0.25)
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', default='/work/lmeadows/rccl_analysis.pdf')
    args = parser.parse_args()

    kf, mf = find_rank0_files(args.profile)
    markers = parse_markers(mf)
    kernels = parse_kernels(kf)
    raw_data = correlate(markers, kernels)

    sizes = sorted(set(s for s, ip in raw_data if ip == 0))

    # Build cleaned flat arrays for changepoint search
    all_sizes_b, all_times, all_idx = [], [], []
    n_rejected = 0
    for i, s in enumerate(sizes):
        vals = np.array(raw_data.get((s, 0), []))
        clean = reject_outliers(vals)
        n_rejected += len(vals) - len(clean)
        for v in clean:
            all_sizes_b.append(float(s))
            all_times.append(v)
            all_idx.append(i)
    all_sizes_b = np.array(all_sizes_b)
    all_times = np.array(all_times)
    all_idx = np.array(all_idx, dtype=int)
    print('Outlier rejection: removed %d points' % n_rejected)

    # Noise characterization
    print('\n=== Noise Characterization (out-of-place) ===')
    print('%8s  %8s  %8s  %8s  %8s  %8s  %6s' % (
        'Size', 'Median', 'Std', 'IQR', 'CV%', 'MAD', 'N'))
    for s in sizes:
        vals = np.array(raw_data.get((s, 0), []))
        if len(vals) == 0:
            continue
        med = np.median(vals)
        std = np.std(vals)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        cv = 100 * std / med if med > 0 else 0
        mad = np.median(np.abs(vals - med))
        print('%8s  %8.2f  %8.2f  %8.2f  %7.1f%%  %8.2f  %6d' % (
            human_size(s), med, std, iqr, cv, mad, len(vals)))

    # Changepoint search — force 2 or 3
    best_bic, best_bps, best_fits = search_changepoints(
        sizes, all_idx, all_sizes_b, all_times, n_cp_range=(2, 3))

    edges = [0] + best_bps + [len(sizes)]
    n_seg = len(edges) - 1

    print('\n=== Best Segmentation: %d segment(s), BIC = %.1f ===' % (
        n_seg, best_bic))
    for si in range(n_seg):
        s0, s1 = edges[si], edges[si + 1]
        fit = best_fits[si]
        print('\nSegment %d: %s – %s  (%d sizes)' % (
            si + 1, human_size(sizes[s0]), human_size(sizes[s1 - 1]), s1 - s0))
        print('  Model: %s' % fit['model'])
        print('  Equation: %s' % model_equation(fit))
        print('  sigma: %.3f us' % fit['sigma'])

    # Generate PDF
    with PdfPages(args.output) as pdf:
        page_overview(pdf, sizes, raw_data, edges, best_fits)
        for si in range(n_seg):
            page_segment_detail(pdf, sizes, raw_data, edges[si], edges[si + 1],
                                best_fits[si], si)
        page_noisy_boxwhisker(pdf, sizes, raw_data)

    print('\nSaved %s' % args.output)


if __name__ == '__main__':
    main()
