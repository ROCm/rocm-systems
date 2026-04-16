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
       rocgdb -batch -x rocgdb_deadlock_analysis.py --args <exe> <args>

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

    def __init__(self, inferiors=None, output_file=None, color=None):
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
        """
        self.inferiors = inferiors
        self.out = output_file if output_file is not None else sys.stdout
        if color is None:
            color = _color_enabled_default(self.out)
        self.c = Colors(enabled=color)

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

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_wg_list(wg_set):
        wg_list = sorted(wg_set)
        parts = [f'({x},{y},{z})' for (x, y, z) in wg_list[:_MAX_WGS_SHOWN]]
        result = ','.join(parts)
        if len(wg_list) > _MAX_WGS_SHOWN:
            result += f' ... (+{len(wg_list) - _MAX_WGS_SHOWN} more)'
        return result

    @staticmethod
    def _format_wf_list(wf_set):
        return ','.join(str(w) for w in sorted(wf_set))

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

        # --- Header ---
        c = self.c
        pid_set = sorted({inf.pid for (inf, thread, wg, wf, _) in entries_with_bt})
        print(f'{c.HEADER}=== rocSHMEM Deadlock Analysis ==={c.RESET}', file=out)
        print(f"Process(es): {', '.join(str(p) for p in pid_set)}", file=out)
        print(f'Total GPU wavefronts analyzed: {len(gpu_threads)}', file=out)
        print(f'Unique backtrace groups: {len(groups)}', file=out)
        print('', file=out)

        # --- Per-group detail ---
        rocshmem_wf_count = 0
        other_wf_count = 0

        for group_num, (key, group_data) in enumerate(groups.items(), start=1):
            entries = group_data['entries']   # list of (inf, thread, wg, wf)
            frames  = group_data['frames']

            wg_set = {wg for (_, _, wg, _) in entries}
            wf_set = {wf for (_, _, _, wf) in entries}

            api_frame = self.detect_rocshmem_api_frame(frames)
            hint = self.detect_hint(frames)

            wg_str = self._format_wg_list(wg_set)
            wf_str = self._format_wf_list(wf_set)

            print(f'{c.GROUP}--- Group {group_num} ({len(entries)} wavefront(s)) ---{c.RESET}', file=out)
            print(f'  WGs: {wg_str}  WFs: {wf_str}', file=out)
            print('  Backtrace:', file=out)

            # The innermost deadlock frame is frame #0 when a known hint pattern
            # is present — highlight it so it stands out in deep call chains.
            hint_frame_idx = None
            if hint:
                for idx, frame in enumerate(frames):
                    for substring, _ in _HINT_RULES:
                        if substring in frame:
                            hint_frame_idx = idx
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
                print(f'  {c.HINT}[HINT] {hint}{c.RESET}', file=out)

            print('', file=out)

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


# ---------------------------------------------------------------------------
# GDB command: rocshmem-deadlock-analyze [output_file]
# ---------------------------------------------------------------------------

class RocshmemDeadlockCommand(gdb.Command):
    """Analyze rocSHMEM deadlocks in the current inferior(s).

    Usage: rocshmem-deadlock-analyze [--color|--no-color] [output_file]

    --color      Force colored output even when writing to a file.
    --no-color   Disable colored output even on a TTY.

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
        color = None  # auto-detect

        remaining = []
        for a in args:
            if a == '--color':
                color = True
            elif a == '--no-color':
                color = False
            else:
                remaining.append(a)

        if remaining:
            try:
                output_file = open(remaining[0], 'w')
            except OSError as e:
                print(f'rocshmem-deadlock-analyze: cannot open {remaining[0]!r}: {e}')
                return
        try:
            DeadlockAnalyzer(output_file=output_file, color=color).report()
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
    DeadlockAnalyzer().report()


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
        DeadlockAnalyzer().report()
    else:
        # Running or not yet started: wait for the first stop event
        gdb.events.stop.connect(_on_stop_once)


if os.environ.get('ROCSHMEM_DEADLOCK_AUTO_ANALYZE', '0') == '1':
    _setup_auto_analysis()
