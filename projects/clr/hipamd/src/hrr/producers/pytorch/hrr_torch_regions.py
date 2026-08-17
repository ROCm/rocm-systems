# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
#
# HRR region producer for PyTorch's HIP caching allocator.
#
# PyTorch calls hipMalloc once per segment and then carves per-tensor blocks out
# of it with host pointer arithmetic, so HRR records the segment and never sees
# the blocks. This writes the block layout down beside the capture, which is what
# lets replay tell "this pointer addresses a live tensor" from "this pointer is
# past the end of one".
#
# Usage: put this directory on PYTHONPATH via a sitecustomize.py, or import it
# from the application:
#
#     import hrr_torch_regions; hrr_torch_regions.start()
#
# It is inert unless HRR capture is active, so leaving it enabled costs one idle
# thread. The format it writes is specified in ../README.md; it calls no HRR
# function and links against nothing.

import os
import struct
import sys
import threading
import time

# ---- Wire format (must match hrr/hrr_regions.h) ----------------------------
_REGION_MAGIC = 0x52525248  # "HRRR"
_REGION_VERSION = 1
_REGION_EVENT = 0xFFFD

_ADD, _DEL = 0, 1
_BLOCK, _SEGMENT = 0, 1

_FILE_HEADER = struct.pack("<IHH", _REGION_MAGIC, _REGION_VERSION, 0)
# hrr_event_header: u16 type, u64 seq, u64 ts, u64 tid, u32 len, 2 pad bytes.
_EVENT_HEADER = "<HQQQI2x"
_REC = "<BBBBIQQq"
_REC_SIZE = 32
_BATCH_PREFIX = struct.calcsize(_EVENT_HEADER) + 8  # + u32 n, u32 flags

_INTERVAL_S = float(os.environ.get("HRR_REGIONS_INTERVAL_S", "2.0"))
_VERBOSE = os.environ.get("HRR_REGIONS_VERBOSE", "") not in ("", "0")
# PyTorch's trace ring depth. Must comfortably exceed the number of alloc/free
# events between two polls, or events are overwritten before we read them.
_MAX_ENTRIES = int(os.environ.get("HRR_REGIONS_MAX_ENTRIES", "1000000"))

# PyTorch device_trace actions that change what is live. free_requested and
# free_completed both map to DEL: the replayer's erase is idempotent, and
# treating the request as the end of the block's life is the conservative
# reading — a pointer used after free is exactly what we want reported.
_ACTIONS = {
    "alloc":          (_ADD, _BLOCK),
    "free_requested": (_DEL, _BLOCK),
    "free_completed": (_DEL, _BLOCK),
    "segment_alloc":  (_ADD, _SEGMENT),
    "segment_free":   (_DEL, _SEGMENT),
}


def _log(msg):
    if _VERBOSE:
        sys.stderr.write("[HRR regions] %s\n" % msg)
        sys.stderr.flush()


def _archive_dir():
    """The capture directory for this process, or None if capture is not active.

    This is the whole "is HRR recording?" check: the capture writer creates
    pid-<pid>/ when it opens, so the directory existing is the signal. Re-checked
    on every poll because capture starts at HIP init, which is normally after
    this module loads.
    """
    root = os.environ.get("HIP_HRR_CAPTURE_OUTPUT")
    if not root:
        return None
    d = os.path.join(root, "pid-%d" % os.getpid())
    return d if os.path.isdir(d) else None


class _Stream:
    """The output file. Reopened on fork, because the child gets its own pid."""

    def __init__(self):
        self._fh = None
        self._pid = None

    def write(self, records):
        """Append one batch. Single write() so a crash can only tear the tail."""
        if not records:
            return
        d = _archive_dir()
        if d is None:
            return
        if self._fh is None or self._pid != os.getpid():
            self.close()
            regions = os.path.join(d, "regions")
            os.makedirs(regions, exist_ok=True)
            self._fh = open(os.path.join(regions, "pytorch.hrrr"), "ab", buffering=0)
            self._pid = os.getpid()
            self._fh.write(_FILE_HEADER)

        now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        buf = bytearray()
        buf += struct.pack(_EVENT_HEADER, _REGION_EVENT, 0, now, 0,
                           _BATCH_PREFIX + _REC_SIZE * len(records))
        buf += struct.pack("<II", len(records), 0)
        for op, kind, device, base, size, mono_ns in records:
            buf += struct.pack(_REC, op, kind, device, 0, 0, base, size, mono_ns)
        self._fh.write(bytes(buf))

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None
        self._pid = None


def _baseline(snap):
    """The layout that already existed when we started watching.

    Blocks allocated before memory history was enabled — model weights,
    persistent buffers, RNG state — appear in no trace entry, so without this
    every kernel argument pointing at them would be reported as out of bounds.
    Stamped mono_ns=0, which the format defines as "live since before the
    stream", so they precede every real record.
    """
    out = []
    for seg in snap.get("segments", []) or []:
        base = int(seg.get("address", 0))
        device = int(seg.get("device", 0))
        out.append((_ADD, _SEGMENT, device, base, int(seg.get("total_size", 0)), 0))
        off = 0
        for b in seg.get("blocks", []) or []:
            size = int(b.get("size", 0))
            if b.get("state", "") == "active_allocated":
                out.append((_ADD, _BLOCK, device, base + off, size, 0))
            off += size
    return out


def _delta(snap, watermark):
    """Trace entries newer than `watermark`, as records. Returns (recs, mark).

    PyTorch stamps trace entries with CLOCK_REALTIME microseconds; HRR event
    headers use CLOCK_MONOTONIC nanoseconds. Sampling both clocks here and
    applying the difference is what makes the two streams comparable — see the
    clock section of ../README.md.
    """
    offset_ns = (time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                 - time.clock_gettime_ns(time.CLOCK_REALTIME))
    recs = []
    mark = watermark
    for device, entries in enumerate(snap.get("device_traces", []) or []):
        for e in entries:
            tu = int(e.get("time_us", 0))
            if tu <= watermark:
                continue
            action = _ACTIONS.get(e.get("action"))
            if action is None:
                continue
            op, kind = action
            recs.append((op, kind, device, int(e.get("addr", 0)),
                         int(e.get("size", 0)), tu * 1000 + offset_ns))
            mark = max(mark, tu)
    recs.sort(key=lambda r: r[5])
    return recs, mark


def _worker():
    # torch dlopens the HIP runtime, and capture opens its archive at HIP init,
    # so nothing here is available at interpreter startup. Wait for torch, then
    # for the archive directory to appear.
    torch = None
    for _ in range(600):  # up to ~10 min, for a slow model import
        torch = sys.modules.get("torch")
        if torch is not None and getattr(torch, "cuda", None) is not None:
            break
        time.sleep(1.0)
    if torch is None:
        _log("torch never imported; producer inert")
        return
    while _archive_dir() is None:
        time.sleep(1.0)

    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context=None, stacks="python",
            max_entries=_MAX_ENTRIES)
    except Exception as e:
        _log("could not enable memory history: %r" % e)
        return

    stream = _Stream()
    # os.fork() leaves this thread dead in the child while the capture writer
    # happily reopens under the new pid, so a forked worker would record nothing
    # at all. Restart there.
    try:
        os.register_at_fork(after_in_child=_start)
    except (AttributeError, ValueError):
        pass

    _log("capture active; polling every %ss" % _INTERVAL_S)
    watermark = 0
    seeded = False
    total = 0
    while True:
        try:
            snap = torch.cuda.memory._snapshot()
            if not seeded:
                base = _baseline(snap)
                stream.write(base)
                total += len(base)
                seeded = True
                _log("baseline: %d records" % len(base))
            recs, watermark = _delta(snap, watermark)
            if recs:
                stream.write(recs)
                total += len(recs)
                _log("wrote %d records (%d total)" % (len(recs), total))
        except Exception as e:  # never let the producer kill the workload
            _log("poll error: %r" % e)
        time.sleep(_INTERVAL_S)


def start():
    """Start the polling thread. Safe to call more than once."""
    for t in threading.enumerate():
        if t.name == "hrr-regions" and t.is_alive():
            return
    try:
        threading.Thread(target=_worker, name="hrr-regions", daemon=True).start()
    except RuntimeError:
        pass


_start = start  # os.register_at_fork wants a plain callable

if os.environ.get("HRR_REGIONS_AUTOSTART", "1") not in ("", "0"):
    start()
