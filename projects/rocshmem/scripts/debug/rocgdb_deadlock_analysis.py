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
rocSHMEM Deadlock Analysis Script for rocgdb (AMD ROCm GDB)

Coalesces identical GPU wavefront backtraces, identifies threads stuck in
rocSHMEM API calls, and provides hints about the likely deadlock cause.

Usage modes:
  1. Batch attach (shell script):
       ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1 \\
         rocgdb -batch -p <pid> -x rocgdb_deadlock_analysis.py

  2. Launch mode:
       mpirun -n 2 -x ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1 rocgdb -batch -x rocgdb_deadlock_analysis.py -ex run --args <exe> <args>

  3. Interactive (already attached):
       (rocgdb) source rocgdb_deadlock_analysis.py
       (rocgdb) rocshmem-deadlock-analyze [output_file]
"""

import gdb
import os
import re
import sys

# ---------------------------------------------------------------------------
# Color support
# ---------------------------------------------------------------------------

class Colors:
    """
    ANSI color/style codes for terminal output.

    Instantiate with ``enabled=True`` to get real escape sequences,
    or ``enabled=False`` to get empty strings for every attribute
    (plain-text mode, safe for file output or non-ANSI terminals).

    Color scheme:
      HEADER   — bold blue        : top-level section banners (=== ... ===)
      GROUP    — bold bright blue : per-group banners (--- Group N ---)
      API      — bold green       : rocSHMEM API entry-point annotation
      STUCK    — bold bright red  : [rocSHMEM] Stuck-in line
      HINT     — bold cyan        : [HINT] line
      LOC      — bold cyan        : innermost deadlock frame (the wait loop)
      WARN     — red              : warnings / no-GPU-threads notice
      SUMMARY_BAD  — bold bright red : wavefronts stuck count in summary
      SUMMARY_OK   — bold green   : wavefronts-not-stuck count in summary
      RESET    — resets all attributes
    """

    def __init__(self, enabled: bool):
        if enabled:
            self.HEADER      = '\033[1;34m'   # bold blue
            self.GROUP       = '\033[1;94m'   # bold bright blue
            self.API         = '\033[1;32m'   # bold green
            self.STUCK       = '\033[1;91m'   # bold bright red
            self.HINT        = '\033[1;36m'   # bold cyan
            self.LOC         = '\033[1;36m'   # bold cyan
            self.WARN        = '\033[31m'      # red
            self.SUMMARY_BAD = '\033[1;91m'   # bold bright red
            self.SUMMARY_OK  = '\033[1;32m'   # bold green
            self.RESET       = '\033[0m'
        else:
            self.HEADER      = ''
            self.GROUP       = ''
            self.API         = ''
            self.STUCK       = ''
            self.HINT        = ''
            self.LOC         = ''
            self.WARN        = ''
            self.SUMMARY_BAD = ''
            self.SUMMARY_OK  = ''
            self.RESET       = ''


def _color_enabled_default(output_file) -> bool:
    """
    Determine whether color should be enabled by default.

    Checks the ``ROCSHMEM_DEADLOCK_COLOR`` environment variable:
      ``always`` (default) — always enable (override with ``0`` / ``no`` to disable).
      ``auto``             — enable iff the output file is a TTY.
      ``1`` / ``yes``      — always enable.
      ``0`` / ``no``       — always disable.
    """
    val = os.environ.get('ROCSHMEM_DEADLOCK_COLOR', 'always').lower()
    if val in ('1', 'yes', 'true', 'always'):
        return True
    if val in ('0', 'no', 'false', 'never'):
        return False
    # 'auto': enable only when writing to a real terminal
    try:
        return output_file.isatty()
    except AttributeError:
        return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex to identify GPU threads from the "info threads" output line.
# ROCm 7 format: "AMDGPU Wave 2:1:1:1 (0,0,0)/0"
#   where the parenthesised triple is (wg_x,wg_y,wg_z) and the last number
#   is the wave-within-workgroup index.
# Older rocgdb format: "AMDGPU Thread X.Y (GPU, WG (x,y,z), WF (n))"
_GPU_WAVE_RE = re.compile(
    r'AMDGPU Wave \S+ \((\d+),(\d+),(\d+)\)/(\d+)'
)
_GPU_THREAD_LEGACY_RE = re.compile(
    r'AMDGPU Thread \d+\.\d+ \(GPU, WG \((\d+),(\d+),(\d+)\), WF \((\d+)\)\)'
)

# After thread.switch(), "thread" command emits e.g.:
#   [Current thread is 6, lane 0 (AMDGPU Lane 2:1:1:1/0 (0,0,0)[0,0,0])]
# We parse WG coords and wavefront/lane from this.
_LANE_RE = re.compile(
    r'AMDGPU Lane \S+/(\d+) \((\d+),(\d+),(\d+)\)'
)

# Regexes for backtrace normalization (stripping volatile parts)
_ADDR_RE   = re.compile(r'0x[0-9a-fA-F]+')
_ARGS_RE   = re.compile(r'\(.*?\)')

# Regex for detecting rocSHMEM public API frames.
#
# Public API functions (declared in include/rocshmem/*.hpp) are free functions
# in the global namespace, so they appear in backtraces as bare "rocshmem_*"
# names with no "::" qualifier anywhere in the frame line.  Internal
# implementation lives in the "rocshmem::" namespace and always has "rocshmem::"
# somewhere in the frame text.
#
# Detection strategy:
#   1. Skip any frame that contains "rocshmem::" — it is an internal symbol.
#   2. In the remaining frames, match "rocshmem_<name>(" to extract the API name.
#
# This covers all ~1000 public device API functions without enumerating them.
_API_FRAME_RE = re.compile(r'\b(rocshmem_[A-Za-z0-9_]+)\s*\(')

# Matches only _wg and _wave collective API calls.
_WG_WAVE_API_RE = re.compile(r'\b(rocshmem_[A-Za-z0-9_]+_(wg|wave))\s*\(')

# Matches named parameters in a backtrace argument list: "name=value"
_NAMED_PARAM_RE = re.compile(r'\b(\w+)=([^,)]+)')

# Parses active lane entries from "info lanes" output.
#
# rocgdb 7 format (one lane per line):
#   "  * 0   A   AMDGPU Lane 2:1:1:1/0 (0,0,0)[0,0,0]"
#   "    1   A   AMDGPU Lane 2:1:1:1/0 (0,0,0)[1,0,0]"
#   "    2   I   AMDGPU Lane 2:1:1:1/0 (0,0,0)[2,0,0]"   # inactive
# Groups: (lane_id, wi_x, wi_y, wi_z)  — only rows with state 'A'.
_LANE_INFO_RE = re.compile(
    r'^\s*\*?\s*(\d+)\s+A\s+AMDGPU\s+Lane\s+\S+/\d+\s+\(\d+,\d+,\d+\)\[(\d+),(\d+),(\d+)\]',
    re.MULTILINE
)


def _extract_raw_args(frame):
    """
    Extract the raw argument string from a single backtrace frame line.

    Given a line like::

        #3  rocshmem_putmem_wg (ctx=0x…, dest=0x…, nelems=64, pe=1) at file.hpp:42

    returns ``'ctx=0x…, dest=0x…, nelems=64, pe=1'``.
    """
    paren = frame.find('(')
    if paren == -1:
        return ''
    at_pos = frame.rfind(' at ')
    sub = frame[paren + 1: at_pos if at_pos != -1 else len(frame)]
    end = sub.rfind(')')
    return sub[:end].strip() if end != -1 else sub.strip()


def _parse_named_args(args_str):
    """
    Return a dict of ``{name: value}`` from a ``'a=1, b=2'``-style argument
    string.  Hex addresses are normalized to ``'<addr>'`` so that per-wavefront
    pointer values do not produce false mismatches.
    """
    result = {}
    for m in _NAMED_PARAM_RE.finditer(args_str):
        name = m.group(1)
        val  = _ADDR_RE.sub('<addr>', m.group(2).strip())
        result[name] = val
    return result


def _differing_params(arg_dicts):
    """
    Given a list of ``{name: value}`` dicts (one per wavefront), return the
    sorted list of parameter names whose values are not identical across all
    dicts.
    """
    all_keys = sorted(set().union(*arg_dicts))
    return [k for k in all_keys
            if len({d.get(k, '<missing>') for d in arg_dicts}) > 1]

# Default set of backtrace substrings used to cull non-rocSHMEM blocking groups.
# These are GPU synchronization primitives that are not rocSHMEM deadlocks:
#   - HIP workgroup barriers (__syncthreads and variants)
#   - AMDGCN scalar barrier intrinsic and workgroup barrier wrapper
#   - Named sync intrinsic
#   - OCKL grid / multi-grid / global-workspace-sync barriers
#   - Cooperative-groups sync (grid_group, thread_block, multi_grid_group)
_DEFAULT_CULL_PATTERNS = [
    '__syncthreads',            # HIP workgroup barrier (and _count/_and/_or)
    '__builtin_amdgcn_s_barrier',  # direct AMDGCN s_barrier intrinsic
    '__work_group_barrier',     # workgroup barrier with memory-fence flags
    '__named_sync',             # named barrier intrinsic
    '__ockl_grid_sync',         # OCKL grid-level sync
    '__ockl_gws_barrier',       # OCKL global-workspace barrier
    '__ockl_multi_grid_sync',   # OCKL multi-grid sync
    'cooperative_groups',       # HIP cooperative-groups (grid/thread_block/multi_grid)
]

# ---------------------------------------------------------------------------
# rocSHMEM backend statistics support
# ---------------------------------------------------------------------------
#
# Stats are only meaningful when the library was built with -DPROFILE=ON.
# Without that flag, ROCStats and ROCHostStats are both NullStats<N> (empty
# structs, size 1), and every getStat() returns 0.
#
# Memory layout (PROFILE=ON):
#   Context::ctxStats     — Stats<NUM_STATS>         — array of uint64_t
#                           written by GPU wavefronts via device-side atomicAdd;
#                           lives in hipMalloc device memory (rocgdb can read it).
#   Context::ctxHostStats — HostStats<NUM_HOST_STATS> — array of atomic_ullong
#                           written by host threads via std::atomic; lives in host
#                           heap memory allocated with the Context object.
#
# Backend::globalStats and Backend::globalHostStats are always zero because
# accumulateStats() is never called.  Stats are read by walking the per-context
# objects directly: ctx_array (device pool) for device stats and list_of_ctxs
# (host context vector) for host stats.
#
# Stat names and their array indices are read directly from the inferior's
# DWARF debug info (rocshmem::rocshmem_stats and rocshmem::rocshmem_host_stats
# enum types) so that new entries added to the enum are automatically picked
# up without any change to this script.


def _load_stat_enum(enum_type_name, strip_prefix, terminator_name):
    """
    Read a rocSHMEM stats enum from the inferior's debug info and return
    a list of ``(index, display_name)`` pairs sorted by index.

    ``strip_prefix`` is removed from the front of each enumerator name and
    the result is lowercased to form the display name
    (e.g. ``'NUM_BARRIER_ALL_WG'`` with prefix ``'NUM_'`` → ``'barrier_all_wg'``).
    The ``terminator_name`` entry (e.g. ``'NUM_STATS'``) is excluded.

    Returns ``None`` if the enum type cannot be found in the debug info.
    """
    try:
        enum_type = gdb.lookup_type(enum_type_name)
    except gdb.error:
        return None

    entries = []
    for field in enum_type.fields():
        # GDB may return namespace-qualified names (e.g. 'rocshmem::NUM_HOST_PUT').
        # Strip any leading 'namespace::' qualifiers before comparing or slicing.
        bare = field.name.rsplit('::', 1)[-1] if field.name else ''
        if bare == terminator_name:
            continue
        display = bare[len(strip_prefix):].lower()
        entries.append((int(field.enumval), display))

    entries.sort(key=lambda x: x[0])
    return entries

# Hint rules: ordered most-specific first. First match wins.
# Each entry: (substring_to_find_in_frame_text, hint_message)
_HINT_RULES = [
    # --- mlx5 (Mellanox/ConnectX) backend ---
    (
        'mlx5_poll_cq_until',
        'Waiting for NIC completion (CQ polling). '
        'NIC completions are not arriving — check if the NIC is responsive '
        'and if the remote PE is also stuck.'
    ),
    (
        'acquire_lock',
        'Waiting for SQ spinlock held by another wavefront. '
        'The lock holder is likely itself deadlocked in mlx5_poll_cq_until.'
    ),
    (
        'mlx5_quiet',
        'Quiet operation waiting for all outstanding RMA ops to complete. '
        'Check NIC health and completion queue state.'
    ),
    # --- bnxt (Broadcom) backend ---
    (
        'bnxt_poll_cq_until',
        'Waiting for NIC completion (CQ polling). '
        'NIC completions are not arriving — check if the bnxt NIC is responsive '
        'and if the remote PE is also stuck.'
    ),
    (
        'bnxt_post_wqe_rma',
        'Waiting for bnxt SQ spinlock held by another wavefront. '
        'The lock holder is likely itself deadlocked in bnxt_poll_cq_until.'
    ),
    (
        'bnxt_quiet',
        'Quiet operation waiting for all outstanding RMA ops to complete (bnxt). '
        'Check NIC health and completion queue state.'
    ),
    # --- ionic (AMD/Pensando) backend ---
    (
        'ionic_quiet_internal_ccqe_single',
        'Waiting for NIC completion in CCQE mode (ionic, single-thread path). '
        'NIC completions are not arriving — check if the ionic NIC is responsive '
        'and if the remote PE is also stuck.'
    ),
    (
        'ionic_quiet_internal_ccqe',
        'Waiting for NIC completion in CCQE mode (ionic). '
        'NIC completions are not arriving — check if the ionic NIC is responsive '
        'and if the remote PE is also stuck.'
    ),
    (
        'ionic_quiet_internal',
        'Waiting for NIC completion (CQ polling, ionic). '
        'NIC completions are not arriving — check if the ionic NIC is responsive '
        'and if the remote PE is also stuck.'
    ),
    (
        'spin_lock_acquire_unique',
        'Waiting for ionic SQ doorbell spinlock (unique/exclusive) held by another '
        'wavefront. The lock holder may itself be stuck in ionic_quiet_internal.'
    ),
    (
        'spin_lock_acquire_shared',
        'Waiting for ionic CQ spinlock (shared) held by another wavefront. '
        'The lock holder may itself be stuck in ionic_quiet_internal.'
    ),
    (
        'ionic_quiet',
        'Quiet operation waiting for all outstanding RMA ops to complete (ionic). '
        'Check NIC health and completion queue state.'
    ),
    # --- shared / IPC / user-level waits (all backends) ---
    (
        'wait_until_any',
        'Waiting for any element of a multi-element condition '
        '(barrier/sync or user rocshmem_wait_until_any). '
        'Check if the remote PE is alive and making progress.'
    ),
    (
        'wait_until_all',
        'Waiting for all elements of a multi-element condition '
        '(barrier/sync or user rocshmem_wait_until_all). '
        'Check if the remote PE is alive and making progress.'
    ),
    (
        'wait_until_some',
        'Waiting for some elements of a multi-element condition '
        '(user rocshmem_wait_until_some). '
        'Check if the remote PE is alive and making progress.'
    ),
    (
        'wait_until',
        'Waiting for remote PE memory update (barrier/sync or user rocshmem_wait_until). '
        'Check if the remote PE is alive and making progress.'
    ),
]

# Maximum number of WG coordinates to show before truncating
_MAX_WGS_SHOWN = 12


# ---------------------------------------------------------------------------
# DeadlockAnalyzer
# ---------------------------------------------------------------------------

class DeadlockAnalyzer:
    """
    Analyzes GPU wavefront backtraces from a rocgdb session to identify
    deadlocks in rocSHMEM applications.
    """

    def __init__(self, inferiors=None, output_file=None, color=None,
                 cull_patterns=None, lane_check=False, show_stats=False):
        """
        Parameters
        ----------
        inferiors : list[gdb.Inferior] or None
            Inferiors to analyze. If None, all valid inferiors are used.
        output_file : file-like or None
            Where to write the report. Defaults to sys.stdout.
        color : bool or None
            True/False to force color on/off; None to auto-detect from the
            ``ROCSHMEM_DEADLOCK_COLOR`` environment variable and whether the
            output file is a TTY.
        cull_patterns : list[str] or None
            Substrings used to filter out backtrace groups that are stuck in
            non-rocSHMEM GPU synchronization primitives (e.g. __syncthreads,
            AMDGCN barriers, cooperative-groups grid sync).  Any group whose
            representative frame list contains at least one of these substrings
            is silently dropped from the report.

            ``None``  — culling disabled (default).
            ``[]``    — cull using ``_DEFAULT_CULL_PATTERNS``.
            ``[...]`` — cull using the supplied list.
        lane_check : bool
            When True, iterate over the active lanes of each wavefront that is
            inside a ``_wg`` or ``_wave`` collective call, collect per-lane
            argument values, and report any lanes whose arguments differ.
            This is slow (up to 64 ``bt`` calls per wavefront) so it is
            disabled by default.  Enable with ``--check-lanes``.
        show_stats : bool
            When True, read the rocSHMEM backend statistics counters
            (``Backend::globalStats`` and ``Backend::globalHostStats``) from
            the stopped process and append a ``=== rocSHMEM Stats ===``
            section to the report.  Requires the library to have been built
            with ``-DPROFILE=ON``; without that flag all counters are zero
            (``NullStats``).  Enabled with ``--stats``.
        """
        self.inferiors = inferiors
        self.out = output_file if output_file is not None else sys.stdout
        if color is None:
            color = _color_enabled_default(self.out)
        self.c = Colors(enabled=color)
        if cull_patterns is None:
            self.cull_patterns = None
        elif len(cull_patterns) == 0:
            self.cull_patterns = _DEFAULT_CULL_PATTERNS
        else:
            self.cull_patterns = list(cull_patterns)
        self.lane_check = lane_check
        self.show_stats = show_stats

    # ------------------------------------------------------------------
    # Thread collection
    # ------------------------------------------------------------------

    def collect_gpu_threads(self):
        """
        Return a list of (inferior, thread, wg_coords, wf_id) for every
        GPU wavefront thread visible in the current session.
        wg_coords is a tuple (x, y, z); wf_id is an int.
        Host/CPU threads are silently skipped.

        rocgdb identification strategy:
        - ROCm 7+: GPU wavefront threads have ptid of (pid, 1, nonzero) while
          host threads have ptid (pid, lwp_id, 0).  We also parse WG/WF
          coordinates from the "info threads" output or by running the "thread"
          command after switching.
        - Older rocgdb: thread name contains "AMDGPU Thread X.Y (GPU, WG ...)"
        """
        results = []
        inferiors = self.inferiors if self.inferiors else gdb.inferiors()
        for inf in inferiors:
            if not inf.is_valid():
                continue

            # Parse "info threads" output to extract WG/WF for each GPU wave.
            # Build a dict: gdb_thread_num -> (wg, wf)
            wg_map = {}  # gdb thread number -> (wg_tuple, wf_int)
            try:
                info_out = gdb.execute('info threads', to_string=True)
                for line in info_out.splitlines():
                    # ROCm 7 format line example:
                    #   "  6    AMDGPU Wave 2:1:1:1 (0,0,0)/0 "barrier_kernel" ..."
                    m_wave = _GPU_WAVE_RE.search(line)
                    if m_wave:
                        # extract thread number from beginning of line
                        tok = line.strip().lstrip('*').strip().split()
                        if tok:
                            try:
                                tnum = int(tok[0])
                            except ValueError:
                                continue
                            wg = (int(m_wave.group(1)),
                                  int(m_wave.group(2)),
                                  int(m_wave.group(3)))
                            wf = int(m_wave.group(4))
                            wg_map[tnum] = (wg, wf)
                        continue
                    # Legacy format
                    m_leg = _GPU_THREAD_LEGACY_RE.search(line)
                    if m_leg:
                        tok = line.strip().lstrip('*').strip().split()
                        if tok:
                            try:
                                tnum = int(tok[0])
                            except ValueError:
                                continue
                            wg = (int(m_leg.group(1)),
                                  int(m_leg.group(2)),
                                  int(m_leg.group(3)))
                            wf = int(m_leg.group(4))
                            wg_map[tnum] = (wg, wf)
            except gdb.error:
                pass

            for thread in inf.threads():
                if not thread.is_valid():
                    continue

                pid, lwp, tid = thread.ptid

                # GPU wavefront threads in ROCm 7+ have lwp==1 and tid!=0.
                # Host threads have lwp==their own TID and tid==0.
                is_gpu_by_ptid = (lwp == 1 and tid != 0)

                # Also accept by name (legacy format or future formats)
                name = thread.name or ''
                is_gpu_by_name = (
                    _GPU_WAVE_RE.search(name) is not None or
                    _GPU_THREAD_LEGACY_RE.search(name) is not None or
                    'AMDGPU' in name
                )

                if not (is_gpu_by_ptid or is_gpu_by_name):
                    continue

                # Try to get WG/WF from the pre-parsed map using thread number
                tnum = thread.num
                if tnum in wg_map:
                    wg, wf = wg_map[tnum]
                else:
                    # Fallback: switch to thread and run "thread" command
                    try:
                        thread.switch()
                        t_out = gdb.execute('thread', to_string=True)
                        m_lane = _LANE_RE.search(t_out)
                        if m_lane:
                            wf = int(m_lane.group(1))
                            wg = (int(m_lane.group(2)),
                                  int(m_lane.group(3)),
                                  int(m_lane.group(4)))
                        else:
                            # Cannot parse coordinates; use ptid as surrogate
                            wg = (0, 0, 0)
                            wf = tid
                    except gdb.error:
                        wg = (0, 0, 0)
                        wf = tid

                results.append((inf, thread, wg, wf))
        return results

    # ------------------------------------------------------------------
    # Backtrace collection
    # ------------------------------------------------------------------

    def get_backtrace(self, thread):
        """
        Switch to *thread* and return its backtrace as a list of frame strings
        (lines starting with '#').  Returns a single error entry on failure.
        """
        try:
            thread.switch()
            raw = gdb.execute('bt', to_string=True)
            frames = []
            for line in raw.splitlines():
                stripped = line.strip()
                if stripped.startswith('#'):
                    frames.append(stripped)
            return frames if frames else ['<empty backtrace>']
        except gdb.error as e:
            return [f'<backtrace error: {e}>']

    # ------------------------------------------------------------------
    # Normalization and coalescing
    # ------------------------------------------------------------------

    def normalize_backtrace(self, frames):
        """
        Return a canonical string key for *frames* by stripping volatile
        parts (addresses, argument values, source file paths) so that
        structurally identical backtraces from different wavefronts compare
        equal.
        """
        normalized = []
        for frame in frames:
            s = _ADDR_RE.sub('<addr>', frame)
            s = _ARGS_RE.sub('(...)', s)
            normalized.append(s.strip())
        return '\n'.join(normalized)

    def coalesce_backtraces(self, entries_with_bt):
        """
        Group entries by their normalized backtrace.

        Parameters
        ----------
        entries_with_bt : list[(inf, thread, wg, wf, frames)]

        Returns
        -------
        dict mapping canonical_key -> {'entries': [...], 'frames': [...]}
        where 'frames' is the representative (first-seen) raw frame list.
        """
        groups = {}
        for (inf, thread, wg, wf, frames) in entries_with_bt:
            key = self.normalize_backtrace(frames)
            if key not in groups:
                groups[key] = {'entries': [], 'frames': frames}
            groups[key]['entries'].append((inf, thread, wg, wf))
        return groups

    def cull_groups(self, groups):
        """
        Remove groups whose representative backtrace matches any pattern in
        ``self.cull_patterns``.  Returns ``(kept, n_culled, n_wf_culled)``
        where *kept* is the filtered dict, *n_culled* is the number of groups
        removed, and *n_wf_culled* is the total wavefront count removed.

        If ``self.cull_patterns`` is None, returns the dict unchanged.
        """
        if not self.cull_patterns:
            return groups, 0, 0

        kept = {}
        n_culled = 0
        n_wf_culled = 0
        for key, group_data in groups.items():
            frames = group_data['frames']
            frame_text = '\n'.join(frames)

            # Never cull a group that is inside rocSHMEM: a __syncthreads (or
            # any other barrier) found there is called internally by rocSHMEM
            # itself (e.g. rocshmem_putmem_wg) and must be reported.
            in_rocshmem = (
                'rocshmem::' in frame_text or
                self.detect_rocshmem_api_frame(frames) is not None
            )
            if in_rocshmem:
                kept[key] = group_data
                continue

            if any(pat in frame_text for pat in self.cull_patterns):
                n_culled += 1
                n_wf_culled += len(group_data['entries'])
            else:
                kept[key] = group_data
        return kept, n_culled, n_wf_culled

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def detect_rocshmem_api_frame(self, frames):
        """
        Walk frames from outermost (user side, highest index) inward to find
        the first rocSHMEM public API boundary.

        Public API functions appear as bare ``rocshmem_*`` names (global
        namespace, no ``::`` qualifier).  Internal implementation is always
        qualified with the ``rocshmem::`` namespace.  Frames containing
        ``rocshmem::`` are skipped; the first remaining frame that matches
        ``rocshmem_<name>(`` is the API entry point.

        Returns the matched function name string, or None.
        """
        for frame in reversed(frames):
            if 'rocshmem::' in frame:
                continue
            m = _API_FRAME_RE.search(frame)
            if m:
                return m.group(1)
        return None

    def detect_hint(self, frames):
        """
        Scan all frames for known deadlock-indicating patterns and return
        the most specific hint string, or None if no pattern is recognized.
        """
        frame_text = '\n'.join(frames)
        for substring, hint in _HINT_RULES:
            if substring in frame_text:
                return hint
        return None

    def _detect_wg_wave_api(self, frames):
        """
        Like detect_rocshmem_api_frame but restricted to ``_wg`` and ``_wave``
        collective API functions.  Returns ``(api_name, raw_frame)`` for the
        outermost matching frame, or ``(None, None)`` if none is found.
        """
        for frame in reversed(frames):
            if 'rocshmem::' in frame:
                continue
            m = _WG_WAVE_API_RE.search(frame)
            if m:
                return m.group(1), frame
        return None, None

    def analyze_collective_usage(self, entries_with_bt):
        """
        Scan all per-wavefront backtraces for incorrect ``_wg`` / ``_wave``
        collective API usage and return a list of issue dicts.

        Detectable violations
        ---------------------
        **Non-collective _wg call**
            Not all wavefronts in a workgroup are at the same ``_wg``
            primitive.  The API contract requires every wavefront in the WG to
            call the same function together.

        **_wg parameter mismatch**
            All wavefronts in a WG are in the same ``_wg`` call but with
            different argument values (after stripping pointer addresses).

        Not detectable
        --------------
        ``_wave`` primitives are collective only within a single wavefront
        (all 64 lanes must call with matching parameters).  Different
        wavefronts — even in the same WG — may independently call any
        ``_wave`` function with different parameters; that is valid.
        Lane-level divergence within a wavefront is not visible from
        wavefront-level backtraces and therefore cannot be checked here.

        Each returned dict has at minimum:
          ``type``  — ``'NON_COLLECTIVE_WG'`` or ``'PARAM_MISMATCH_WG'``
          ``pid``   — process ID
          ``wg``    — workgroup coordinates tuple
        """
        issues = []

        # Collect (inf.pid, wg) → list of per-wf info dicts
        wg_map = {}
        for (inf, thread, wg, wf, frames) in entries_with_bt:
            api_name, raw_frame = self._detect_wg_wave_api(frames)
            key = (inf.pid, wg)
            if key not in wg_map:
                wg_map[key] = []
            wg_map[key].append({
                'wf': wf,
                'api': api_name,         # e.g. 'rocshmem_putmem_wg' or None
                'frame': raw_frame,      # raw backtrace line for the API frame
                'frames': frames,        # full raw backtrace
            })

        for (pid, wg), wf_list in sorted(wg_map.items()):
            if len(wf_list) < 2:
                continue  # need at least two wavefronts to compare

            wg_entries = [e for e in wf_list if e['api'] and e['api'].endswith('_wg')]
            non_wg     = [e for e in wf_list if not (e['api'] and e['api'].endswith('_wg'))]

            # ---- _wg non-collective check --------------------------------
            if wg_entries and non_wg:
                # Some wavefronts are inside a _wg collective; others are not.
                issues.append({
                    'type': 'NON_COLLECTIVE_WG',
                    'pid': pid,
                    'wg': wg,
                    'in_collective':     [(e['wf'], e['api']) for e in wg_entries],
                    'not_in_collective': [(e['wf'], e['api']) for e in non_wg],
                })
            elif wg_entries:
                wg_api_names = {e['api'] for e in wg_entries}
                if len(wg_api_names) > 1:
                    # Wavefronts in the same WG are in *different* _wg calls.
                    issues.append({
                        'type': 'NON_COLLECTIVE_WG',
                        'pid': pid,
                        'wg': wg,
                        'in_collective':     [(e['wf'], e['api']) for e in wg_entries],
                        'not_in_collective': [],
                    })
                else:
                    # All in the same _wg call — check parameters.
                    api_name = next(iter(wg_api_names))
                    arg_dicts = [_parse_named_args(_extract_raw_args(e['frame']))
                                 for e in wg_entries if e['frame']]
                    if len(arg_dicts) >= 2:
                        diff = _differing_params(arg_dicts)
                        if diff:
                            issues.append({
                                'type': 'PARAM_MISMATCH_WG',
                                'pid': pid,
                                'wg': wg,
                                'api': api_name,
                                'differing': diff,
                                'per_wf': [
                                    (e['wf'], _parse_named_args(
                                        _extract_raw_args(e['frame'])))
                                    for e in wg_entries if e['frame']
                                ],
                            })

        return issues

    def collect_lane_args(self, thread, frames):
        """
        Switch to *thread* and iterate over its active lanes, collecting the
        per-lane argument values for the ``_wg``/``_wave`` collective API call
        visible in *frames*.

        Returns a list of ``(lane_id, work_item_xyz, args_dict)`` tuples — one
        entry per active lane.  ``args_dict`` is empty when rocgdb cannot parse
        arguments for that lane (e.g. optimised-out values).

        The caller is responsible for restoring the current thread/lane context
        if needed; this method leaves the thread/lane state in an indeterminate
        position after iteration.
        """
        api_name, _ = self._detect_wg_wave_api(frames)
        if not api_name:
            return []

        try:
            thread.switch()
            lane_info = gdb.execute('info lanes', to_string=True)
        except gdb.error:
            return []

        active_lanes = _LANE_INFO_RE.findall(lane_info)
        if not active_lanes:
            return []

        results = []
        for lane_id_str, wi_x, wi_y, wi_z in active_lanes:
            lane_id = int(lane_id_str)
            work_item = (int(wi_x), int(wi_y), int(wi_z))
            try:
                gdb.execute(f'lane {lane_id}', to_string=True)
                raw_bt = gdb.execute('bt', to_string=True)
                lane_frames = [l.strip() for l in raw_bt.splitlines()
                               if l.strip().startswith('#')]
                _, raw_frame = self._detect_wg_wave_api(lane_frames)
                args = _parse_named_args(_extract_raw_args(raw_frame)) if raw_frame else {}
            except gdb.error:
                args = {}
            results.append((lane_id, work_item, args))

        return results

    def analyze_lane_usage(self, entries_with_bt):
        """
        For each wavefront that is inside a ``_wg`` or ``_wave`` collective
        call, iterate over its active lanes and compare their per-lane argument
        values.  Any wavefront where at least two active lanes disagree on a
        parameter is reported as a ``'LANE_PARAM_MISMATCH'`` issue.

        This is the lane-level complement of ``analyze_collective_usage``:
        - ``_wg`` lane check — all lanes within a wavefront that is in a
          ``_wg`` call should agree on parameters (the wavefront participates
          as a unit; lanes should not diverge).
        - ``_wave`` lane check — the ``_wave`` contract requires ALL active
          lanes within a single wavefront to call the function with matching
          parameters; divergence here is a programming error.

        Returns a list of issue dicts, each with keys:
          ``type``      — ``'LANE_PARAM_MISMATCH'``
          ``pid``       — process ID
          ``wg``        — workgroup coordinates tuple
          ``wf``        — wavefront index within the WG
          ``api``       — collective API function name
          ``scope``     — ``'wg'`` or ``'wave'``
          ``differing`` — sorted list of parameter names that differ
          ``per_lane``  — list of ``(lane_id, work_item_xyz, args_dict)``
        """
        issues = []
        for (inf, thread, wg, wf, frames) in entries_with_bt:
            api_name, _ = self._detect_wg_wave_api(frames)
            if not api_name:
                continue

            lane_data = self.collect_lane_args(thread, frames)
            if len(lane_data) < 2:
                continue

            arg_dicts = [args for (_, _, args) in lane_data if args]
            if len(arg_dicts) < 2:
                continue

            diff = _differing_params(arg_dicts)
            if diff:
                scope = 'wg' if api_name.endswith('_wg') else 'wave'
                issues.append({
                    'type': 'LANE_PARAM_MISMATCH',
                    'pid': inf.pid,
                    'wg': wg,
                    'wf': wf,
                    'api': api_name,
                    'scope': scope,
                    'differing': diff,
                    'per_lane': lane_data,
                })
        return issues

    def _format_collective_issues(self, issues):
        """
        Render the list of collective-usage issues returned by
        ``analyze_collective_usage`` as a multi-line string.
        """
        c = self.c
        lines = []
        lines.append(f'{c.HEADER}=== Collective API Usage Issues ==={c.RESET}')

        for issue in issues:
            wg = issue['wg']
            wg_str = f'({wg[0]},{wg[1]},{wg[2]})'
            itype = issue['type']

            if itype == 'NON_COLLECTIVE_WG':
                in_col  = issue['in_collective']
                not_col = issue['not_in_collective']
                lines.append(
                    f'{c.STUCK}[BAD] WG {wg_str} — non-collective _wg call{c.RESET}')
                if in_col:
                    wf_list = ', '.join(f'WF {wf} ({api})' for wf, api in sorted(in_col))
                    lines.append(f'  In collective : {wf_list}')
                if not_col:
                    others = []
                    for wf, api in sorted(not_col):
                        others.append(f'WF {wf} ({api if api else "not in _wg call"})')
                    lines.append(f'  Not in collective: {", ".join(others)}')

            elif itype == 'PARAM_MISMATCH_WG':
                lines.append(
                    f'{c.STUCK}[BAD] WG {wg_str} — parameter mismatch in '
                    f'{issue["api"]}{c.RESET}')
                lines.append(
                    f'  Differing parameter(s): '
                    f'{c.HINT}{", ".join(issue["differing"])}{c.RESET}')
                for wf, args in sorted(issue['per_wf']):
                    param_str = ', '.join(
                        f'{k}={v}' for k, v in sorted(args.items())
                        if k in issue['differing']
                    )
                    lines.append(f'    WF {wf}: {param_str}')

            elif itype == 'LANE_PARAM_MISMATCH':
                scope = issue['scope']
                wf = issue['wf']
                lines.append(
                    f'{c.STUCK}[BAD] WG {wg_str} WF {wf} — lane parameter mismatch '
                    f'in {issue["api"]} (_{scope} contract){c.RESET}')
                lines.append(
                    f'  Differing parameter(s): '
                    f'{c.HINT}{", ".join(issue["differing"])}{c.RESET}')
                for lane_id, work_item, args in issue['per_lane']:
                    if not args:
                        continue
                    wi_str = f'[{work_item[0]},{work_item[1]},{work_item[2]}]'
                    param_str = ', '.join(
                        f'{k}={v}' for k, v in sorted(args.items())
                        if k in issue['differing']
                    )
                    lines.append(f'    Lane {lane_id} {wi_str}: {param_str}')

        lines.append('')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_wgwf_list(wgwf_set):
        wgwf_list = sorted(wgwf_set)
        parts = [f'({x},{y},{z})/{wf}' for (x, y, z, wf) in wgwf_list[:_MAX_WGS_SHOWN]]
        result = ','.join(parts)
        if len(wgwf_list) > _MAX_WGS_SHOWN:
            result += f' ... (+{len(wgwf_list) - _MAX_WGS_SHOWN} more)'
        return result

    # ------------------------------------------------------------------
    # Backend statistics
    # ------------------------------------------------------------------

    def collect_backend_stats(self):
        """
        Read rocSHMEM backend statistics from the stopped process.

        ``backend->globalStats`` and ``backend->globalHostStats`` are always zero
        because ``accumulateStats()`` is never called — GPU wavefronts write into
        per-context ``ctxStats`` objects (in device memory) and host threads write
        into per-context ``ctxHostStats`` objects (in host heap memory).

        This method therefore walks the per-context stat objects directly:

        * **Device stats** — cast ``backend`` to its concrete subclass
          (``IPCBackend``, ``ROBackend``, or ``GDABackend``), access ``ctx_array``
          (a ``hipMalloc``-allocated pool of device-context objects), and sum
          ``ctxStats`` across all pool slots.  rocgdb can read device memory
          directly; unused slots have all-zero stats so over-reading is safe.

        * **Host stats** — walk ``backend->list_of_ctxs`` (a
          ``std::vector<Context*>`` of all host-created contexts) and sum
          ``ctxHostStats`` from each.

        Returns a dict with keys:

        ``'profile_on'`` — bool; False when the library was built without
            ``-DPROFILE=ON`` (all counters would be zero anyway).
        ``'my_pe'`` — int; the PE rank read from ``Backend::my_pe``.
        ``'device'`` — dict mapping stat-name → non-zero count.
        ``'host'`` — dict mapping stat-name → non-zero count.
        ``'error'`` — str; present only when an error prevented reading.

        Returns None if ``rocshmem::backend`` is not found (symbol absent or
        the library was not initialised yet).
        """
        try:
            backend_ptr = gdb.parse_and_eval('rocshmem::backend')
        except gdb.error:
            return None

        if int(backend_ptr) == 0:
            return None

        # Detect PROFILE=ON via sizeof(ROCStats):
        #   Stats<N>     (PROFILE=ON)  → N * sizeof(uint64_t) bytes (> 1)
        #   NullStats<N> (PROFILE=OFF) → empty struct, sizeof == 1
        try:
            roc_stats_sz = int(gdb.parse_and_eval('sizeof(rocshmem::ROCStats)'))
        except gdb.error:
            roc_stats_sz = 0

        profile_on = (roc_stats_sz > 1)

        try:
            backend_val = backend_ptr.dereference()
            my_pe = int(backend_val['my_pe'])
        except gdb.error as e:
            return {'error': str(e), 'profile_on': profile_on}

        result = {'profile_on': profile_on, 'my_pe': my_pe,
                  'device': {}, 'host': {}}

        if not profile_on:
            return result

        # Read enum field names and indices directly from the inferior's
        # DWARF debug info so that new stats added to the enum are picked up
        # automatically without any change to this script.
        dev_names  = _load_stat_enum('rocshmem::rocshmem_stats',
                                     'NUM_', 'NUM_STATS')
        host_names = _load_stat_enum('rocshmem::rocshmem_host_stats',
                                     'NUM_HOST_', 'NUM_HOST_STATS')

        if dev_names is None or host_names is None:
            result['error'] = ('rocshmem::rocshmem_stats or '
                               'rocshmem::rocshmem_host_stats enum type not '
                               'found in debug info')
            return result

        ull_ptr = gdb.lookup_type('unsigned long long').pointer()

        # --- Device stats: walk the backend-specific ctx_array pool ---
        # BackendType enum: GDA_BACKEND=0, RO_BACKEND=1, IPC_BACKEND=2
        _BACKEND_SUBCLASS = {
            0: ('rocshmem::GDABackend', 'rocshmem::GDAContext'),
            1: ('rocshmem::ROBackend',  'rocshmem::ROContext'),
            2: ('rocshmem::IPCBackend', 'rocshmem::IPCContext'),
        }
        try:
            backend_type = int(backend_val['type'])
            mapping = _BACKEND_SUBCLASS.get(backend_type)
            if mapping is not None:
                backend_cls, ctx_cls = mapping
                sub_ptr = backend_ptr.cast(
                    gdb.lookup_type(backend_cls).pointer())
                sub = sub_ptr.dereference()
                ctx_array = sub['ctx_array']  # IPCContext* / ROContext* / GDAContext*

                # Pool size: envvar::max_num_contexts (a var<size_t> whose
                # underlying value lives in the 'value' member field).
                try:
                    n_ctxs = int(gdb.parse_and_eval(
                        'rocshmem::envvar::max_num_contexts.value'))
                except gdb.error:
                    n_ctxs = 32  # default from envvar.cpp

                ctx_type = gdb.lookup_type(ctx_cls)
                for i in range(n_ctxs):
                    ctx = (ctx_array + i).dereference()
                    dev_base = ctx['ctxStats'].address.cast(ull_ptr)
                    for idx, name in dev_names:
                        val = int((dev_base + idx).dereference())
                        if val:
                            result['device'][name] = (
                                result['device'].get(name, 0) + val)
        except gdb.error as e:
            result['error'] = 'device stats: ' + str(e)

        # --- Host stats: walk list_of_ctxs (all host-created contexts) ---
        # std::vector<Context*> uses libstdc++ internal layout:
        #   _M_impl._M_start  — pointer to first element
        #   _M_impl._M_finish — pointer one past the last element
        try:
            vec = backend_val['list_of_ctxs']
            vec_start  = vec['_M_impl']['_M_start']
            vec_finish = vec['_M_impl']['_M_finish']
            n_host_ctxs = int(vec_finish - vec_start)
            for i in range(n_host_ctxs):
                ctx_ptr = (vec_start + i).dereference()
                ctx = ctx_ptr.dereference()
                host_base = ctx['ctxHostStats'].address.cast(ull_ptr)
                for idx, name in host_names:
                    val = int((host_base + idx).dereference())
                    if val:
                        result['host'][name] = (
                            result['host'].get(name, 0) + val)
        except gdb.error as e:
            prev = result.get('error', '')
            result['error'] = (prev + '; ' if prev else '') + 'host stats: ' + str(e)

        return result

    def _format_stats(self, stats_result):
        """Render the dict returned by ``collect_backend_stats`` as a string."""
        c = self.c
        lines = [f'{c.HEADER}=== rocSHMEM Stats ==={c.RESET}']

        if stats_result is None:
            lines.append('  (rocshmem::backend symbol not found — library not '
                         'initialised or debug symbols absent)')
            lines.append('')
            return '\n'.join(lines)

        if 'error' in stats_result:
            lines.append(f'  (error reading stats: {stats_result["error"]})')
            lines.append('')
            return '\n'.join(lines)

        pe = stats_result.get('my_pe', '?')
        lines.append(f'  PE {pe}')

        if not stats_result['profile_on']:
            lines.append('  (library built without -DPROFILE=ON; all counters'
                         ' are zero)')
            lines.append('')
            return '\n'.join(lines)

        lines.append('  (counters reflect activity since process start)')

        for section, data in (('Device', stats_result['device']),
                               ('Host',   stats_result['host'])):
            if not data:
                lines.append(f'  {section}: (all zero)')
                continue
            lines.append(f'  {section}:')
            # Compute column width for alignment
            w = max(len(k) for k in data)
            for name, val in data.items():
                lines.append(f'    {name:<{w}}  {val}')

        lines.append('')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Main report
    # ------------------------------------------------------------------

    def report(self):
        """Interrupt (if needed), collect backtraces, and print the report."""
        out = self.out

        # Attempt to interrupt in case the process is still running.
        # In batch attach mode the process is already stopped, so this is a no-op.
        try:
            gdb.execute('interrupt', to_string=True)
        except gdb.error:
            pass

        # Collect GPU threads across all (selected) inferiors
        gpu_threads = self.collect_gpu_threads()

        if not gpu_threads:
            print(f'{self.c.WARN}No GPU wavefront threads found.{self.c.RESET}', file=out)
            print('(The process may not have GPU kernels running, or rocgdb '
                  'could not enumerate GPU threads.)', file=out)
            return

        # Gather backtraces
        entries_with_bt = []
        for (inf, thread, wg, wf) in gpu_threads:
            frames = self.get_backtrace(thread)
            entries_with_bt.append((inf, thread, wg, wf, frames))

        # Coalesce identical backtraces
        groups = self.coalesce_backtraces(entries_with_bt)

        # Optionally cull groups stuck in non-rocSHMEM GPU barriers / gridsync
        groups, n_culled, n_wf_culled = self.cull_groups(groups)

        # --- Header ---
        c = self.c
        pid_set = sorted({inf.pid for (inf, thread, wg, wf, _) in entries_with_bt})
        print(f'{c.HEADER}=== rocSHMEM Deadlock Analysis ==={c.RESET}', file=out)
        print(f"Process(es): {', '.join(str(p) for p in pid_set)}", file=out)
        print(f'Total GPU wavefronts analyzed: {len(gpu_threads)}', file=out)
        print(f'Unique backtrace groups: {len(groups)}', file=out)
        if n_culled:
            print(f'  (+ {n_culled} group(s) / {n_wf_culled} wavefront(s) culled'
                  f' — GPU barrier / gridsync)', file=out)
        print('', file=out)

        # --- Per-group detail ---
        rocshmem_wf_count = 0
        other_wf_count = 0

        for group_num, (key, group_data) in enumerate(groups.items(), start=1):
            entries = group_data['entries']   # list of (inf, thread, wg, wf)
            frames  = group_data['frames']

            wgwf_set = {(wg[0], wg[1], wg[2], wf) for (_, _, wg, wf) in entries}

            api_frame = self.detect_rocshmem_api_frame(frames)
            hint = self.detect_hint(frames)

            wgwf_str = self._format_wgwf_list(wgwf_set)

            print(f'{c.GROUP}--- Group {group_num} ({len(entries)} wavefront(s)) ---{c.RESET}', file=out)
            print(f'  WFs: {wgwf_str}', file=out)
            print('  Backtrace:', file=out)

            # The innermost deadlock frame is frame #0 when a known hint pattern
            # is present — highlight it so it stands out in deep call chains.
            hint_frame_idx = None
            hint_frame_text = None
            if hint:
                for idx, frame in enumerate(frames):
                    for substring, _ in _HINT_RULES:
                        if substring in frame:
                            hint_frame_idx = idx
                            hint_frame_text = frame
                            break
                    if hint_frame_idx is not None:
                        break

            for idx, frame in enumerate(frames):
                annotation = ''
                if api_frame and api_frame in frame:
                    annotation = f'  {c.API}<<< rocSHMEM API entry{c.RESET}'
                if idx == hint_frame_idx:
                    print(f'    {c.LOC}{frame}{c.RESET}{annotation}', file=out)
                else:
                    print(f'    {frame}{annotation}', file=out)

            if api_frame:
                print(f'  {c.STUCK}[rocSHMEM] Stuck in: {api_frame}{c.RESET}', file=out)

            in_rocshmem = api_frame or any('rocshmem::' in f for f in frames)
            if in_rocshmem:
                rocshmem_wf_count += len(entries)
            else:
                other_wf_count += len(entries)

            if hint:
                detected = f' (at: {hint_frame_text.strip()})' if hint_frame_text else ''
                print(f'  {c.HINT}[HINT] {hint}{detected}{c.RESET}', file=out)
            elif frames:
                print(f'  {c.LOC}[STUCK] Innermost frame: {frames[0].strip()}{c.RESET}', file=out)

            print('', file=out)

        # --- Collective API usage issues ---
        collective_issues = self.analyze_collective_usage(entries_with_bt)

        # --- Lane-level analysis (optional, slow) ---
        lane_issues = []
        if self.lane_check:
            lane_issues = self.analyze_lane_usage(entries_with_bt)

        all_issues = collective_issues + lane_issues
        if all_issues:
            print(self._format_collective_issues(all_issues), file=out)

        # --- Backend stats (optional) ---
        if self.show_stats:
            stats_result = self.collect_backend_stats()
            print(self._format_stats(stats_result), file=out)

        # --- Summary ---
        print(f'{c.HEADER}=== Summary ==={c.RESET}', file=out)
        stuck_str = f'{rocshmem_wf_count} wavefront(s) inside rocSHMEM'
        other_str = f'{other_wf_count} wavefront(s) in user code / other'
        if rocshmem_wf_count:
            stuck_str = f'{c.SUMMARY_BAD}{stuck_str}{c.RESET}'
        if other_wf_count == 0 and rocshmem_wf_count:
            other_str = f'{c.SUMMARY_OK}{other_str}{c.RESET}'
        print(f'  {stuck_str}', file=out)
        print(f'  {other_str}', file=out)
        if all_issues:
            n_bad = len(all_issues)
            bad_str = f'{c.SUMMARY_BAD}{n_bad} collective API issue(s) detected{c.RESET}'
            print(f'  {bad_str}', file=out)


# ---------------------------------------------------------------------------
# GDB command: rocshmem-deadlock-analyze [output_file]
# ---------------------------------------------------------------------------

class RocshmemDeadlockCommand(gdb.Command):
    """Analyze rocSHMEM deadlocks in the current inferior(s).

    Usage: rocshmem-deadlock-analyze [--color|--no-color] [--cull[=pat,...]]
                                     [--check-lanes] [--stats] [output_file]

    --color           Force colored output even when writing to a file.
    --no-color        Disable colored output even on a TTY.
    --cull            Cull groups stuck in GPU barriers / gridsync using the
                      built-in default pattern list (__syncthreads, AMDGCN
                      s_barrier, OCKL grid/gws sync, cooperative_groups, ...).
    --cull=p1,p2,...  Cull groups whose backtrace contains any of the
                      comma-separated substrings p1, p2, ...
    --check-lanes     Enable lane-level parameter mismatch detection for _wg
                      and _wave collective calls.  For each wavefront in such
                      a call, rocgdb switches to every active lane and compares
                      per-lane argument values (up to 64 bt calls per wavefront;
                      slow but thorough).
    --stats           Append a rocSHMEM Stats section showing non-zero API call
                      counters.  Device stats are summed across all slots in
                      the backend ctx_array pool (rocgdb reads device memory
                      directly); host stats are summed across all contexts in
                      list_of_ctxs.  Requires -DPROFILE=ON; counters reflect
                      activity since process start, not since the deadlock.

    When output_file is given the full report is written there;
    otherwise it is printed to stdout.  Color is auto-detected from the
    ROCSHMEM_DEADLOCK_COLOR environment variable or TTY status when neither
    flag is supplied.
    """

    def __init__(self):
        super().__init__('rocshmem-deadlock-analyze', gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        output_file = None
        color = None       # auto-detect
        cull_patterns = None  # culling disabled by default
        lane_check = False
        show_stats = False

        remaining = []
        for a in args:
            if a == '--color':
                color = True
            elif a == '--no-color':
                color = False
            elif a == '--cull':
                cull_patterns = []  # use _DEFAULT_CULL_PATTERNS
            elif a.startswith('--cull='):
                cull_patterns = [p for p in a[len('--cull='):].split(',') if p]
            elif a == '--check-lanes':
                lane_check = True
            elif a == '--stats':
                show_stats = True
            else:
                remaining.append(a)

        if remaining:
            try:
                output_file = open(remaining[0], 'w')
            except OSError as e:
                print(f'rocshmem-deadlock-analyze: cannot open {remaining[0]!r}: {e}')
                return
        try:
            DeadlockAnalyzer(output_file=output_file, color=color,
                             cull_patterns=cull_patterns,
                             lane_check=lane_check,
                             show_stats=show_stats).report()
        finally:
            if output_file is not None:
                output_file.close()


RocshmemDeadlockCommand()


# ---------------------------------------------------------------------------
# Automatic entry-point logic
# ---------------------------------------------------------------------------
# When ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1 the script runs the analysis as soon
# as the inferior stops (batch attach use case) or immediately if already
# stopped (attach mode where gdb stops the process on attach).
# ---------------------------------------------------------------------------

_analysis_done = False


def _auto_analyzer_kwargs():
    """
    Build DeadlockAnalyzer keyword arguments from environment variables for
    the auto-analyze path (ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1).

    ROCSHMEM_DEADLOCK_CULL:
      unset / empty  — culling disabled (default)
      "1"            — cull using _DEFAULT_CULL_PATTERNS
      "pat1,pat2,…"  — cull using the supplied comma-separated patterns

    ROCSHMEM_DEADLOCK_CHECK_LANES:
      "1"            — enable lane-level parameter mismatch detection

    ROCSHMEM_DEADLOCK_STATS:
      "1"            — call rocshmem_dump_stats() and append backend API
                       call counters to the report (requires PROFILE=ON)

    ROCSHMEM_DEADLOCK_COLOR:
      handled by _color_enabled_default(); not overridden here so the
      DeadlockAnalyzer constructor auto-detects it as usual.
    """
    kwargs = {}
    cull_env = os.environ.get('ROCSHMEM_DEADLOCK_CULL', '')
    if cull_env == '1':
        kwargs['cull_patterns'] = []        # triggers _DEFAULT_CULL_PATTERNS
    elif cull_env:
        kwargs['cull_patterns'] = [p for p in cull_env.split(',') if p]
    if os.environ.get('ROCSHMEM_DEADLOCK_CHECK_LANES', '0') == '1':
        kwargs['lane_check'] = True
    if os.environ.get('ROCSHMEM_DEADLOCK_STATS', '0') == '1':
        kwargs['show_stats'] = True
    return kwargs


def _on_stop_once(event):
    """Stop-event handler: run analysis exactly once, then disconnect."""
    global _analysis_done
    if _analysis_done:
        return
    _analysis_done = True
    try:
        gdb.events.stop.disconnect(_on_stop_once)
    except Exception:
        pass
    DeadlockAnalyzer(**_auto_analyzer_kwargs()).report()


def _setup_auto_analysis():
    """
    Called at module load time when ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1.
    If the inferior is already stopped (typical for batch -p <pid> attach),
    run analysis immediately.  Otherwise register a stop-event handler.
    """
    try:
        inf = gdb.selected_inferior()
    except Exception:
        return

    if inf.is_valid() and inf.pid != 0 and inf.threads():
        # Already stopped (attach mode): analyze now
        DeadlockAnalyzer(**_auto_analyzer_kwargs()).report()
    else:
        # Running or not yet started: wait for the first stop event
        gdb.events.stop.connect(_on_stop_once)


if os.environ.get('ROCSHMEM_DEADLOCK_AUTO_ANALYZE', '0') == '1':
    _setup_auto_analysis()
