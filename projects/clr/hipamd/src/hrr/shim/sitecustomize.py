# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
#
# HRR sub-allocation map capture shim.
#
# Auto-imported by CPython at interpreter startup when this directory is on
# PYTHONPATH. On a daemon thread it reads PyTorch's HIP caching-allocator state
# (torch.cuda.memory) and pushes two compact binary blobs into libamdhip64's
# HRR capture writer via the exported symbols
#   int  hipHrrCaptureActive(void);
#   void hipHrrCaptureSubAllocSnapshot(const void* blob, uint64_t len);
#   void hipHrrCaptureSubAllocTimeline(const void* blob, uint64_t len);
# the coarse segment->block layout (_snapshot) and the precise alloc/free
# timeline (memory-history device_traces). The writer stores each blob
# content-addressed (crash-safe temp+rename) and records a small
# HRR_SUBALLOC_SNAPSHOT / HRR_SUBALLOC_TIMELINE event, so HRR's existing
# checkpoint/crash-finalize durability preserves the map even when the captured
# run dies on a GPU fault. Binary layout MUST match hrr_suballoc.h.
#
# Canonical copy lives in-tree at hrr/shim/sitecustomize.py; the hrr-testing
# harness copy must be kept in sync with this file.
#
# Disabled unless capture is active, so importing it outside a capture run is a
# no-op beyond starting one idle polling thread.

import os
import struct
import sys
import threading
import time

_MAGIC = 0x42415348   # "HSAB"
_TL_MAGIC = 0x4C545348  # "HSTL"
_VERSION = 1
_INTERVAL_S = float(os.environ.get("HRR_SUBALLOC_INTERVAL_S", "2.0"))
_VERBOSE = os.environ.get("HRR_SUBALLOC_VERBOSE", "") not in ("", "0")
# Full alloc/free timeline (precise per-kernel layout). On by default; the
# timeline is what lets replay reconstruct the exact live block set at each
# kernel. Set HRR_SUBALLOC_TIMELINE=0 to fall back to coarse snapshots only.
_TIMELINE = os.environ.get("HRR_SUBALLOC_TIMELINE", "1") not in ("", "0")
# PyTorch trace ring depth. Must comfortably exceed the number of alloc/free
# events between two polls so none are lost before we read them.
_TL_MAX_ENTRIES = int(os.environ.get("HRR_SUBALLOC_TL_MAX_ENTRIES", "1000000"))

# PyTorch device_trace action -> timeline record action code (hrr_suballoc.h).
_TL_ACTIONS = {
    "alloc": 0,           # HRR_TL_ALLOC
    "free_requested": 1,  # HRR_TL_FREE (block leaves the active set)
    "free_completed": 1,  # idempotent erase at playback
    "segment_alloc": 2,   # HRR_TL_SEGMENT_ALLOC
    "segment_free": 3,    # HRR_TL_SEGMENT_FREE
}


def _log(msg):
    if _VERBOSE:
        sys.stderr.write("[HRR suballoc shim] %s\n" % msg)
        sys.stderr.flush()


def _load_api():
    import ctypes
    for soname in ("libamdhip64.so.7", "libamdhip64.so", "libamdhip64.so.6"):
        try:
            lib = ctypes.CDLL(soname)
        except OSError:
            continue
        if not hasattr(lib, "hipHrrCaptureSubAllocSnapshot"):
            continue
        active = lib.hipHrrCaptureActive
        active.restype = ctypes.c_int
        active.argtypes = []
        push = lib.hipHrrCaptureSubAllocSnapshot
        push.restype = None
        push.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        push_tl = None
        if hasattr(lib, "hipHrrCaptureSubAllocTimeline"):
            push_tl = lib.hipHrrCaptureSubAllocTimeline
            push_tl.restype = None
            push_tl.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        return ctypes, active, push, push_tl
    return None, None, None, None


def _build_blob(segments):
    out = bytearray()
    out += struct.pack("<IIQ", _MAGIC, _VERSION, len(segments))
    for seg in segments:
        addr = int(seg.get("address", 0))
        total = int(seg.get("total_size", 0))
        blocks = seg.get("blocks", []) or []
        out += struct.pack("<QQII", addr, total, len(blocks), 0)
        off = 0
        for b in blocks:
            size = int(b.get("size", 0))
            active = 1 if b.get("state", "") == "active_allocated" else 0
            out += struct.pack("<QQB7x", off, size, active)
            off += size
    return bytes(out)


def _enable_history(torch):
    # Record the alloc/free trace WITHOUT C++ context; "python" stacks are the
    # lightest accepted value (frames are ignored by us). Returns True on success.
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context=None, stacks="python",
            max_entries=_TL_MAX_ENTRIES)
        return True
    except Exception as e:
        _log("could not enable memory history: %r" % e)
        return False


def _build_baseline_delta(snap, seen):
    # Blocks allocated *before* history was enabled (model weights, persistent
    # buffers, RNG state, ...) are not in device_traces, so the replayer would
    # see kernel pointers into them as "inside a segment but no live block" =
    # false OOB (and the block-level guard would skip them). Seed the timeline
    # with those blocks, stamped at mono_ns=0 so they precede every real event.
    #
    # This is done LAZILY and ADDITIVELY on every poll rather than once at
    # enable time: capture goes active at HIP init, which happens long before a
    # large model finishes allocating its weights, so a single enable-time
    # baseline captures almost nothing (empirically ~1 block). By reconciling
    # every live snapshot against `seen` (blocks already emitted as baseline OR
    # already forwarded as a real device_traces alloc), we pick up each
    # persistent block the first time it becomes visible — without ever
    # double-counting a block the trace already covers. Callers MUST update
    # `seen` from the timeline FIRST (see _build_timeline_blob) so blocks that
    # were genuinely allocated during the run keep their real timestamp instead
    # of being pinned to mono_ns=0.
    segments = snap.get("segments", []) if isinstance(snap, dict) else []
    recs = []
    for seg in segments:
        base = int(seg.get("address", 0))
        total = int(seg.get("total_size", 0))
        off = 0
        seg_emitted = False
        for b in seg.get("blocks", []) or []:
            size = int(b.get("size", 0))
            if b.get("state", "") == "active_allocated":
                addr = base + off
                if addr not in seen:
                    if not seg_emitted:
                        recs.append((2, base, total))  # HRR_TL_SEGMENT_ALLOC
                        seg_emitted = True
                    recs.append((0, addr, size))       # HRR_TL_ALLOC
                    seen.add(addr)
            off += size
    if not recs:
        return None
    out = bytearray()
    out += struct.pack("<IIQ", _TL_MAGIC, _VERSION, len(recs))
    for code, addr, size in recs:
        out += struct.pack("<B7xQQq", code, addr, size, 0)
    return bytes(out)


def _build_timeline_blob(snap, watermark, seen):
    # Convert PyTorch trace entries newer than `watermark` (time_us) into a
    # timeline delta blob. time_us is CLOCK_REALTIME microseconds; convert to the
    # CLOCK_MONOTONIC ns used by HRR event headers via the current clock offset.
    # Every ALLOC address forwarded here is recorded in `seen` so the lazy
    # baseline (_build_baseline_delta) never re-emits a block the trace already
    # covers at its real timestamp.
    traces = snap.get("device_traces", []) if isinstance(snap, dict) else []
    offset_ns = (time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                 - time.clock_gettime_ns(time.CLOCK_REALTIME))
    recs = []
    new_watermark = watermark
    for dev in traces:
        for e in dev:
            tu = int(e.get("time_us", 0))
            if tu <= watermark:
                continue
            code = _TL_ACTIONS.get(e.get("action"))
            if code is None:
                continue
            addr = int(e.get("addr", 0))
            recs.append((tu, code, addr, int(e.get("size", 0))))
            if code == 0:      # HRR_TL_ALLOC — trace covers this block
                seen.add(addr)
            if tu > new_watermark:
                new_watermark = tu
    if not recs:
        return None, watermark
    recs.sort(key=lambda r: r[0])  # PyTorch trace order == time order
    out = bytearray()
    out += struct.pack("<IIQ", _TL_MAGIC, _VERSION, len(recs))
    for tu, code, addr, size in recs:
        out += struct.pack("<B7xQQq", code, addr, size, tu * 1000 + offset_ns)
    return bytes(out), new_watermark


def _worker():
    # libamdhip64 and its deps (librocprofiler-register, etc.) are not yet
    # resolvable at interpreter startup; they become loadable once torch
    # dlopens the HIP runtime. So wait for torch first, then resolve the
    # exported HRR symbols (retrying), then poll for capture to go active.
    torch = None
    for _ in range(600):  # up to ~10 min for a large model import
        torch = sys.modules.get("torch")
        if torch is not None and getattr(torch, "cuda", None) is not None:
            break
        time.sleep(1.0)
    if torch is None:
        _log("torch never imported; shim inert")
        return

    ctypes = active = push = push_tl = None
    for _ in range(120):
        ctypes, active, push, push_tl = _load_api()
        if push is not None:
            break
        time.sleep(1.0)
    if push is None:
        _log("libamdhip64 HRR sub-alloc symbols not found; shim inert")
        return

    # Wait for capture to be active (writer opens at HIP init).
    while True:
        try:
            if active() == 1:
                break
        except Exception:
            return
        time.sleep(1.0)

    use_tl = _TIMELINE and push_tl is not None and _enable_history(torch)
    # Addresses already accounted for in the replay's live set — either forwarded
    # as a real device_traces alloc or emitted as a mono_ns=0 baseline. Guards
    # the lazy additive baseline so each persistent block is seeded exactly once.
    seen = set()

    def _push_baseline(snap):
        # Emit a mono_ns=0 baseline delta for any live block not yet in `seen`.
        try:
            b = _build_baseline_delta(snap, seen)
            if b is not None:
                buf = (ctypes.c_char * len(b)).from_buffer_copy(b)
                push_tl(buf, len(b))
                if _VERBOSE:
                    _log("pushed baseline delta: %d records, %d bytes (seen=%d)"
                         % ((len(b) - 16) // 32, len(b), len(seen)))
        except Exception as e:
            _log("baseline delta error: %r" % e)

    if use_tl:
        # Initial baseline: whatever is already live at enable time. The poll
        # loop keeps augmenting it as more persistent blocks (e.g. weights that
        # are still loading) become visible.
        try:
            _push_baseline(torch.cuda.memory._snapshot())
        except Exception as e:
            _log("initial baseline error: %r" % e)
    _log("capture active; torch present — starting loop (interval=%ss, timeline=%s)"
         % (_INTERVAL_S, use_tl))
    last_blob = None
    n = 0
    n_tl = 0
    tl_recs = 0
    watermark = 0
    while True:
        try:
            if active() != 1:
                time.sleep(_INTERVAL_S)
                continue
            snap = torch.cuda.memory._snapshot()
            segments = snap.get("segments", []) if isinstance(snap, dict) else []
            if segments:
                blob = _build_blob(segments)
                # Skip pushing an unchanged layout back-to-back (compare full
                # content so a changed-but-same-length layout is never missed).
                if blob != last_blob:
                    buf = (ctypes.c_char * len(blob)).from_buffer_copy(blob)
                    push(buf, len(blob))
                    last_blob = blob
                    n += 1
                    if _VERBOSE:
                        nblk = sum(len(s.get("blocks", []) or []) for s in segments)
                        _log("pushed snapshot #%d: %d segments, %d blocks, %d bytes"
                             % (n, len(segments), nblk, len(blob)))
            if use_tl:
                # Forward the real trace FIRST so its alloc addresses populate
                # `seen` before the lazy baseline runs — otherwise a block that
                # was genuinely allocated during this interval would be pinned to
                # mono_ns=0 instead of keeping its true timestamp.
                tl_blob, watermark = _build_timeline_blob(snap, watermark, seen)
                if tl_blob is not None:
                    buf = (ctypes.c_char * len(tl_blob)).from_buffer_copy(tl_blob)
                    push_tl(buf, len(tl_blob))
                    n_tl += 1
                    # 16-byte header, 32-byte records.
                    tl_recs += (len(tl_blob) - 16) // 32
                    if _VERBOSE:
                        _log("pushed timeline #%d: %d new events (%d total), %d bytes"
                             % (n_tl, (len(tl_blob) - 16) // 32, tl_recs, len(tl_blob)))
                # Then seed any still-unseen live block (pre-history persistent
                # tensors) at mono_ns=0.
                _push_baseline(snap)
        except Exception as e:  # never let the shim kill the workload
            _log("snapshot error: %r" % e)
        time.sleep(_INTERVAL_S)


def _start():
    try:
        t = threading.Thread(target=_worker, name="hrr-suballoc", daemon=True)
        t.start()
    except Exception:
        pass


_start()
