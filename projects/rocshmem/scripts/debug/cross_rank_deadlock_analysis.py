#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################
"""
cross_rank_deadlock_analysis.py — Coalesce rocSHMEM deadlock analysis output
across MPI ranks.

Reads all pe*_pid_*.txt files produced by attach_deadlock_analysis.sh in a
shared output directory, groups identical backtrace patterns across ranks, and
highlights patterns that are uncommon (present in only a minority of ranks).

Usage:
    python3 cross_rank_deadlock_analysis.py <output_dir> [--color|--no-color]
"""

import os
import re
import sys
import glob
from collections import defaultdict

# ---------------------------------------------------------------------------
# ANSI color support
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _strip_ansi(s):
    return _ANSI_RE.sub('', s)


class Colors:
    def __init__(self, enabled):
        if enabled:
            self.RESET       = '\033[0m'
            self.BOLD        = '\033[1m'
            self.HEADER      = '\033[1;34m'   # bold blue
            self.GROUP       = '\033[1;36m'   # bold cyan
            self.COMMON      = '\033[1;32m'   # bold green
            self.UNCOMMON    = '\033[1;33m'   # bold yellow
            self.UNIQUE      = '\033[1;31m'   # bold red
            self.HINT        = '\033[36m'     # cyan
            self.DIM         = '\033[2m'      # dim
        else:
            self.RESET = self.BOLD = self.HEADER = self.GROUP = ''
            self.COMMON = self.UNCOMMON = self.UNIQUE = self.HINT = self.DIM = ''


def _detect_color():
    val = os.environ.get('ROCSHMEM_DEADLOCK_COLOR', 'always').lower()
    if val in ('1', 'yes', 'true', 'always'):
        return True
    if val in ('0', 'no', 'false', 'never'):
        return False
    return sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_GROUP_HEADER_RE = re.compile(r'^--- Group \d+ \((\d+) wavefront')
_HINT_RE         = re.compile(r'\[HINT\]\s*(.*)')
_ROCSHMEM_WF_RE  = re.compile(r'(\d+) wavefront\(s\) inside rocSHMEM')
_ADDR_RE         = re.compile(r'0x[0-9a-fA-F]+')


def _parse_file(path):
    """
    Parse a single pe*_pid_*.txt analysis file.

    Returns a list of group dicts:
        {
          'wf_count': int,
          'frames':   [str, ...],   # already-normalized frame lines
          'hint':     str or None,
        }
    and the total wavefront count inside rocSHMEM (from the Summary section).
    """
    groups = []
    cur = None
    in_backtrace = False
    rocshmem_wfs = 0

    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for raw in fh:
                line = _strip_ansi(raw.rstrip())

                m = _GROUP_HEADER_RE.search(line)
                if m:
                    cur = {'wf_count': int(m.group(1)), 'frames': [], 'hint': None}
                    groups.append(cur)
                    in_backtrace = False
                    continue

                if cur is None:
                    m = _ROCSHMEM_WF_RE.search(line)
                    if m:
                        rocshmem_wfs = int(m.group(1))
                    continue

                stripped = line.strip()
                if stripped == 'Backtrace:':
                    in_backtrace = True
                elif in_backtrace and re.match(r'#\d+', stripped):
                    cur['frames'].append(stripped)
                elif in_backtrace and stripped.startswith('['):
                    in_backtrace = False
                    hm = _HINT_RE.search(stripped)
                    if hm:
                        cur['hint'] = hm.group(1).strip()
                elif in_backtrace and not stripped:
                    # Blank line ends the backtrace block (no hint/stuck line present)
                    in_backtrace = False
                elif not in_backtrace:
                    hm = _HINT_RE.search(line)
                    if hm:
                        cur['hint'] = hm.group(1).strip()
                    sm = _ROCSHMEM_WF_RE.search(line)
                    if sm:
                        rocshmem_wfs = int(sm.group(1))
    except OSError:
        pass

    return groups, rocshmem_wfs


def _backtrace_key(frames):
    """Stable key for a backtrace: normalize hex addresses (ASLR) before comparing."""
    return tuple(_ADDR_RE.sub('<addr>', f) for f in frames)


# ---------------------------------------------------------------------------
# Cross-rank coalescing
# ---------------------------------------------------------------------------

def coalesce(output_dir):
    """
    Load all pe*_pid_*.txt files, coalesce by backtrace key across ranks,
    and return a sorted list of pattern dicts and per-rank metadata.
    """
    files = sorted(glob.glob(os.path.join(output_dir, 'pe*_pid_*.txt')))
    if not files:
        return [], {}

    # Map rank -> list of groups, rocshmem_wfs
    rank_data = {}
    for path in files:
        fname = os.path.basename(path)
        # Extract rank from filename: pe<rank>_pid_<pid>.txt
        m = re.match(r'^pe(\w+)_pid_\d+\.txt$', fname)
        rank = m.group(1) if m else fname
        groups, rocshmem_wfs = _parse_file(path)
        if rank not in rank_data:
            rank_data[rank] = {'groups': [], 'rocshmem_wfs': 0}
        rank_data[rank]['groups'].extend(groups)
        rank_data[rank]['rocshmem_wfs'] += rocshmem_wfs

    # Coalesce: backtrace_key -> {ranks: set, wf_total: int, hint: str, frames: list}
    patterns = defaultdict(lambda: {'ranks': set(), 'wf_total': 0,
                                    'hint': None, 'frames': []})
    for rank, data in rank_data.items():
        for grp in data['groups']:
            key = _backtrace_key(grp['frames'])
            pat = patterns[key]
            pat['ranks'].add(rank)
            pat['wf_total'] += grp['wf_count']
            pat['frames'] = grp['frames']
            if grp['hint'] and pat['hint'] is None:
                pat['hint'] = grp['hint']

    # Sort: most ranks first, then most wavefronts
    sorted_patterns = sorted(
        patterns.values(),
        key=lambda p: (len(p['ranks']), p['wf_total']),
        reverse=True,
    )
    return sorted_patterns, rank_data


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _rank_label(rank_set, total_ranks):
    n = len(rank_set)
    if n == total_ranks:
        return 'COMMON'
    if n == 1:
        return 'UNIQUE'
    return 'UNCOMMON'


def report(output_dir, color=None):
    if color is None:
        color = _detect_color()
    c = Colors(color)

    patterns, rank_data = coalesce(output_dir)
    all_ranks = sorted(rank_data.keys(),
                       key=lambda r: int(r) if r.isdigit() else r)
    total_ranks = len(all_ranks)

    print(f'{c.HEADER}=== Cross-Rank Deadlock Analysis ==={c.RESET}')
    print(f'Directory: {output_dir}')
    print(f'Ranks with analysis files: {total_ranks}  '
          f'({", ".join(all_ranks)})')
    print(f'Unique backtrace patterns: {len(patterns)}')
    print()

    n_common   = sum(1 for p in patterns if len(p['ranks']) == total_ranks)
    n_uncommon = sum(1 for p in patterns if 1 < len(p['ranks']) < total_ranks)
    n_unique   = sum(1 for p in patterns if len(p['ranks']) == 1)

    for idx, pat in enumerate(patterns, start=1):
        n_ranks = len(pat['ranks'])
        label   = _rank_label(pat['ranks'], total_ranks)

        if label == 'COMMON':
            color_label = f'{c.COMMON}{label}: {n_ranks}/{total_ranks} ranks{c.RESET}'
        elif label == 'UNIQUE':
            color_label = f'{c.UNIQUE}{label}: rank {next(iter(pat["ranks"]))}{c.RESET}'
        else:
            color_label = f'{c.UNCOMMON}{label}: {n_ranks}/{total_ranks} ranks{c.RESET}'

        print(f'{c.GROUP}--- Pattern {idx} [{color_label}{c.GROUP}] ---{c.RESET}')

        rank_list = ', '.join(
            sorted(pat['ranks'], key=lambda r: int(r) if r.isdigit() else r)
        )
        print(f'  Ranks: {rank_list}')
        print(f'  Total wavefronts: {pat["wf_total"]}')
        print('  Backtrace:')
        for frame in pat['frames']:
            print(f'    {frame}')
        if pat['hint']:
            print(f'  {c.HINT}[HINT] {pat["hint"]}{c.RESET}')
        print()

    # Per-rank rocSHMEM wavefront summary
    print(f'{c.HEADER}=== Per-Rank rocSHMEM Wavefront Count ==={c.RESET}')
    for rank in all_ranks:
        wfs = rank_data[rank]['rocshmem_wfs']
        print(f'  Rank {rank}: {wfs} wf(s) inside rocSHMEM')
    print()

    # Overall summary
    print(f'{c.HEADER}=== Summary ==={c.RESET}')
    print(f'  {c.COMMON}{n_common} pattern(s) common to all {total_ranks} rank(s){c.RESET}')
    if n_uncommon:
        print(f'  {c.UNCOMMON}{n_uncommon} pattern(s) present in some rank(s) only{c.RESET}')
    if n_unique:
        print(f'  {c.UNIQUE}{n_unique} pattern(s) present in exactly one rank{c.RESET}')

    # Most common hint across all patterns
    all_hints = [p['hint'] for p in patterns if p['hint']]
    if all_hints:
        from collections import Counter
        dominant_hint = Counter(all_hints).most_common(1)[0][0]
        print(f'  Most common hint: {c.HINT}{dominant_hint}{c.RESET}')


# ---------------------------------------------------------------------------
# Entry point (standalone use)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = sys.argv[1:]
    color = None
    output_dir = None

    for a in args:
        if a == '--color':
            color = True
        elif a == '--no-color':
            color = False
        elif not a.startswith('-'):
            output_dir = a

    if output_dir is None:
        print(f'Usage: {sys.argv[0]} <output_dir> [--color|--no-color]', file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f'ERROR: Not a directory: {output_dir}', file=sys.stderr)
        sys.exit(1)

    report(output_dir, color=color)
