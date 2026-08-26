#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Shared data loading and parsing for SQTT trace analysis tools.

Provides:
  - Funcmap extraction and parsing (from code object ELF sections)
  - Shaderdata record loading (from JSON trace output)
  - Occupancy/wave-span loading
  - Marker decoding
  - Address trace preprocessing (extracts addr_trace blocks from records)
  - Demangling utilities
  - Auto-discovery of trace directories and code objects
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from glob import glob
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Funcmap extraction
# ---------------------------------------------------------------------------

def find_llvm_tool(name: str) -> Optional[str]:
    """Find an LLVM tool, preferring ROCm installation."""
    import shutil
    candidates = [
        f"/opt/rocm/llvm/bin/{name}",
        f"/opt/rocm/lib/llvm/bin/{name}",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return shutil.which(name)


def read_funcmap(code_object: str) -> str:
    """Read .sqtt_funcmap section from a code object."""
    objcopy = find_llvm_tool("llvm-objcopy")
    if not objcopy:
        return ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            tmp = f.name
        subprocess.run(
            [objcopy, f"--dump-section=.sqtt_funcmap={tmp}", code_object],
            capture_output=True, timeout=10, check=True)
        with open(tmp, "r") as f:
            data = f.read()
        os.unlink(tmp)
        return data
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return ""


def demangle_name(name: str) -> str:
    """Demangle a single C++ symbol using c++filt."""
    return demangle_names([name])[0]


def demangle_names(names: list[str]) -> list[str]:
    """Batch-demangle C++ symbols through a single c++filt invocation."""
    if not names:
        return []
    try:
        r = subprocess.run(
            ["c++filt"], input="\n".join(names),
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            result = r.stdout.strip().split("\n")
            if len(result) == len(names):
                return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return list(names)


@dataclass
class FuncMap:
    """ID-to-name mapping from .sqtt_funcmap sections.

    The unified marker dict maps id -> (name, type) where type is one of:
      "function" -- enter/exit scope marker (F: prefix)
      "user"     -- user scope marker (U: prefix)
      "point"    -- point marker: barriers, memory ops, user points (P: prefix)

    source_locs holds the optional debug source location for F: / P: (addr
    trace) entries.  For functions it's the definition site (file:line).
    For address trace ops it's the call-site inline chain
    "<inner>:<line> -> <outer>:<line>" matching rocprofiler-sdk's format.
    """
    markers: dict[int, tuple[str, str]] = field(default_factory=dict)
    source_locs: dict[int, str] = field(default_factory=dict)
    kernels: list[str] = field(default_factory=list)
    kernel_source_locs: dict[str, str] = field(default_factory=dict)
    wave_size: int = 0  # from W: entry (32 or 64), 0 if unknown
    # from R:id:extra_payload_count=N entries; absent means zero
    extra_payload_counts: dict[int, int] = field(default_factory=dict)
    shader_clock_bits: int = 0
    shader_clock_shift: int = 0

    def resolve(self, marker_id: int) -> tuple[str, str]:
        """Returns (name, type). Type determines scope vs point behavior."""
        if marker_id in self.markers:
            return self.markers[marker_id]
        return (f"event#{marker_id}", "function")

    def extra_payload_count(self, marker_id: int) -> int:
        """Returns extra records after this marker header; default is zero."""
        return self.extra_payload_counts.get(marker_id, 0)

def merge_funcmaps(funcmaps: Iterable[FuncMap]) -> FuncMap:
    """Merge funcmaps while preserving all metadata used for marker decoding."""
    merged = FuncMap()
    for fm in funcmaps:
        merged.markers.update(fm.markers)
        merged.source_locs.update(fm.source_locs)
        merged.kernels.extend(fm.kernels)
        merged.kernel_source_locs.update(fm.kernel_source_locs)
        merged.extra_payload_counts.update(fm.extra_payload_counts)
        if fm.wave_size:
            merged.wave_size = fm.wave_size
        if fm.shader_clock_bits:
            merged.shader_clock_bits = fm.shader_clock_bits
            merged.shader_clock_shift = fm.shader_clock_shift
    return merged


def _split_source_loc(s: str) -> tuple[str, str]:
    """Split "name@file:line -> ..." into (name, source_loc).

    Returns (s, "") if no '@' present.  Source location may itself contain
    ' -> ' separators from inline chain expansion; those are preserved.
    """
    if "@" not in s:
        return s, ""
    name, loc = s.split("@", 1)
    return name, loc


def parse_funcmap(raw: str) -> FuncMap:
    """Parse raw .sqtt_funcmap content.

    Format v2 prefixes:
        F:id:name[@source_loc]  -- function (enter/exit scope)
        U:id:name               -- user scope marker
        P:id:name[@source_loc]  -- point marker; source_loc is set for
                                   addr_trace_* point markers
        K:name[@source_loc]     -- kernel (for vaddr lookup, not instrumented)
        W:N                     -- wave size
        R:id:extra_payload_count=N
                                -- number of payload records after the header
        M:shader_clock_bits=N;shader_clock_shift=S
                                -- marker ID shares the word with shader clock bits

    Source locations (when present) follow rocprofiler-sdk's inline chain
    format: "<inner_file>:<line> -> <outer_file>:<line>".
    """
    fm = FuncMap()
    for line in raw.splitlines():
        line = line.strip().rstrip("\x00")
        if not line or ":" not in line:
            continue
        prefix, rest = line.split(":", 1)
        prefix = prefix.strip()
        rest = rest.strip()
        if not rest:
            continue
        if prefix == "W":
            try:
                fm.wave_size = int(rest)
            except ValueError:
                pass
        elif prefix == "M":
            values = {}
            valid = True
            for item in rest.split(";"):
                key, sep, value = item.partition("=")
                key = key.strip()
                if key not in ("shader_clock_bits", "shader_clock_shift"):
                    continue
                if not sep or not value.strip().isdigit():
                    valid = False
                    break
                values[key] = int(value)
            bits = values.get("shader_clock_bits")
            shift = values.get("shader_clock_shift")
            if valid and bits is not None and (
                bits == 0 or (shift is not None and bits <= 29 and shift < 32 and bits <= 32 - shift)
            ):
                fm.shader_clock_bits = bits
                fm.shader_clock_shift = shift or 0
        elif prefix == "K":
            name, loc = _split_source_loc(rest)
            fm.kernels.append(name)
            if loc:
                fm.kernel_source_locs[name] = loc
        elif prefix == "R":
            if ":" not in rest:
                continue
            id_str, attrs = rest.split(":", 1)
            try:
                mid = int(id_str)
            except ValueError:
                continue
            for item in attrs.split(";"):
                key, sep, value = item.partition("=")
                if sep and key.strip() == "extra_payload_count":
                    try:
                        fm.extra_payload_counts[mid] = int(value)
                    except ValueError:
                        pass
        elif prefix in ("F", "U", "P"):
            if ":" not in rest:
                continue
            id_str, name = rest.split(":", 1)
            try:
                mid = int(id_str)
            except ValueError:
                continue
            type_map = {"F": "function", "U": "user", "P": "point"}
            name, loc = _split_source_loc(name)
            fm.markers[mid] = (name, type_map[prefix])
            if loc:
                fm.source_locs[mid] = loc
    return fm


def parse_code_object_id(path: str) -> Optional[int]:
    """Extract code_object_id from a filename like '*_code_object_id_2.out'."""
    import re
    m = re.search(r'code_object_id_(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


def load_funcmaps(code_objects: list[str], do_demangle: bool) -> dict[int, FuncMap]:
    """Load funcmaps per code object, keyed by code_object_id."""
    per_co: dict[int, FuncMap] = {}
    for co in code_objects:
        co_id = parse_code_object_id(co)
        if co_id is None:
            print(f"Warning: cannot extract code_object_id from {co}",
                  file=sys.stderr)
            continue
        raw = read_funcmap(co)
        if not raw:
            continue
        fm = parse_funcmap(raw)
        per_co[co_id] = fm

    if do_demangle:
        all_names: list[str] = []
        index: list[tuple[int, str, int]] = []
        for co_id, fm in per_co.items():
            for k, (name, mtype) in fm.markers.items():
                index.append((co_id, "marker", k))
                all_names.append(name)
            for i, k in enumerate(fm.kernels):
                index.append((co_id, "kernel", i))
                all_names.append(k)
        demangled = demangle_names(all_names)
        for (co_id, fld, key), name in zip(index, demangled):
            if fld == "marker":
                _, mtype = per_co[co_id].markers[key]
                per_co[co_id].markers[key] = (name, mtype)
            else:
                per_co[co_id].kernels[key] = name

    return per_co


def build_kernel_to_co(per_co: dict[int, FuncMap]) -> dict[str, int]:
    """Map kernel mangled names to code_object_id."""
    k2co: dict[str, int] = {}
    for co_id, fm in per_co.items():
        for k in fm.kernels:
            k2co[k] = co_id
    return k2co


# ---------------------------------------------------------------------------
# Shaderdata loading
# ---------------------------------------------------------------------------

@dataclass
class ShaderRecord:
    time: int
    value: int
    cu: int
    simd: int
    wave_id: int
    flags: int
    shader_engine: int = 0


def decode_marker(value: int, funcmap: FuncMap | None = None) -> tuple[int, bool, bool]:
    """Decode known packed headers/bare exits; keep numeric markers legacy."""
    raw = value & 0xFFFFFFFF
    exit_prev = bool(raw & 0x1)
    enter = bool(raw & 0x2)
    legacy_id = raw >> 2
    bits = funcmap.shader_clock_bits if funcmap else 0
    shift = funcmap.shader_clock_shift if funcmap else 0
    if not (funcmap and 0 < bits <= 29 and 0 <= shift < 32 and bits <= 32 - shift):
        return legacy_id, enter, exit_prev

    id_bits = 30 - bits
    packed_id = legacy_id & ((1 << id_bits) - 1)
    if packed_id in funcmap.markers or (packed_id == 0 and exit_prev and not enter):
        return packed_id, enter, exit_prev
    return legacy_id, enter, exit_prev


def _load_shaderdata_from_dir(trace_dir: str) -> list[ShaderRecord]:
    """Load shaderdata records from a single ui_* trace directory."""
    records = []

    filenames_path = os.path.join(trace_dir, "filenames.json")
    shaderdata_files: list[tuple[int, str]] = []

    if os.path.exists(filenames_path):
        with open(filenames_path) as f:
            meta = json.load(f)
        sd_filenames = meta.get("shaderdata_filenames", {})
        for se_id, file_list in sd_filenames.items():
            try:
                shader_engine = int(se_id)
            except (TypeError, ValueError):
                continue
            for entry in file_list:
                fname = entry[0] if isinstance(entry, list) else entry
                shaderdata_files.append((shader_engine, os.path.join(trace_dir, fname)))
    else:
        paths = sorted(
            glob(os.path.join(trace_dir, "shaderdata_*.json")),
            key=lambda p: [int(x) for x in os.path.splitext(os.path.basename(p))[0].split("_")[1:]],
        )
        shaderdata_files = [(int(os.path.basename(path).split("_")[1]), path) for path in paths]

    for shader_engine, path in shaderdata_files:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for rec in data.get("records", []):
            records.append(ShaderRecord(
                time=rec[0], value=rec[1],
                cu=rec[2], simd=rec[3],
                wave_id=rec[4], flags=rec[5],
                shader_engine=shader_engine,
            ))

    return records


def load_shaderdata(trace_dir: str) -> list[ShaderRecord]:
    """Load and time-sort shaderdata records from a single trace directory."""
    records = _load_shaderdata_from_dir(trace_dir)
    records.sort(key=lambda r: r.time)
    return records


# ---------------------------------------------------------------------------
# Occupancy loading -- wave launch/retire timeline
# ---------------------------------------------------------------------------

@dataclass
class WaveSpan:
    """A time range during which a wave was running a specific dispatch."""
    launch_time: int
    retire_time: int  # -1 if not yet retired
    dispatch_id: int


def load_occupancy(trace_dir: str) -> tuple[dict[str, str], dict[tuple, list[WaveSpan]]]:
    """
    Parse occupancy.json from a trace directory.

    Returns:
        dispatches: dict mapping dispatch_id (str) -> kernel name/description
        wave_spans: dict mapping (shader_engine, cu, simd, wave_slot) -> sorted list of WaveSpan
    """
    path = os.path.join(trace_dir, "occupancy.json")
    if not os.path.exists(path):
        return {}, {}

    with open(path) as f:
        data = json.load(f)

    dispatches = data.get("dispatches", {})

    events: dict[tuple, list[tuple]] = defaultdict(list)
    for key, val in data.items():
        try:
            shader_engine = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(val, list):
            continue
        for rec in val:
            if not isinstance(rec, list) or len(rec) < 6:
                continue
            time, cu, simd, wslot, is_launch, dispatch_id = rec[:6]
            events[(shader_engine, cu, simd, wslot)].append((time, is_launch, dispatch_id))

    wave_spans: dict[tuple, list[WaveSpan]] = {}
    for wave_key, evts in events.items():
        evts.sort(key=lambda e: e[0])
        spans: list[WaveSpan] = []
        pending: Optional[WaveSpan] = None
        for time, is_launch, dispatch_id in evts:
            if is_launch:
                if pending:
                    pending.retire_time = time
                    spans.append(pending)
                pending = WaveSpan(launch_time=time, retire_time=-1,
                                   dispatch_id=dispatch_id)
            else:
                if pending:
                    pending.retire_time = time
                    spans.append(pending)
                    pending = None
        if pending:
            pending.retire_time = 2**63
            spans.append(pending)
        wave_spans[wave_key] = spans

    return dispatches, wave_spans


def find_wave_span_at(spans: list[WaveSpan], time: int) -> Optional[WaveSpan]:
    """Find the most recent WaveSpan at or before a given time.

    Trace data records can arrive after a dispatch retires (the wave is
    still draining), so we match to the most recent span whose
    launch_time <= time rather than requiring strict [launch, retire]
    containment. Clock-corrected startup markers can also land just before
    the first occupancy launch; with no earlier wave in that slot, they
    belong to that first span.
    """
    lo, hi = 0, len(spans) - 1
    result = None
    while lo <= hi:
        mid = (lo + hi) // 2
        s = spans[mid]
        if s.launch_time <= time:
            result = s
            lo = mid + 1
        else:
            hi = mid - 1
    return result if result is not None or not spans else spans[0]


def make_funcmap_resolver(
    per_co: dict[int, FuncMap],
    dispatches: dict[str, str],
    kernel_to_co: dict[str, int],
    wave_spans: dict[tuple, list[WaveSpan]],
    fallback: FuncMap,
):
    """Resolve a late record to its latest launch until that slot relaunches."""
    def resolve(record: ShaderRecord) -> FuncMap:
        span = find_wave_span_at(
            wave_spans.get((record.shader_engine, record.cu, record.simd, record.wave_id), []), record.time
        )
        if span is None:
            return fallback
        codeobj = kernel_to_co.get(dispatches.get(str(span.dispatch_id), ""))
        return per_co.get(codeobj, fallback)

    return resolve


def merge_folded(all_folded: list[dict[str, tuple[int, int]]]) -> dict[str, tuple[int, int]]:
    """Merge folded stack counts from multiple independent time domains."""
    merged_cycles: dict[str, int] = defaultdict(int)
    merged_execs: dict[str, int] = defaultdict(int)
    for folded in all_folded:
        for stack, (cycles, execs) in folded.items():
            merged_cycles[stack] += cycles
            merged_execs[stack] += execs
    return {k: (merged_cycles[k], merged_execs[k]) for k in merged_cycles}


# ---------------------------------------------------------------------------
# Address trace preprocessing
# ---------------------------------------------------------------------------

@dataclass
class AddressTrace:
    """A single memory operation with per-lane addresses.

    Only active lanes (EXEC bit set) are stored in the addresses list.
    Inactive lane data is discarded during parsing since those VGPRs
    contain garbage.
    """
    kind: str           # "load", "store", "lds_load", "lds_store"
    time: int
    cu: int
    simd: int
    wave_id: int
    exec_mask: int      # 64-bit EXEC mask
    addresses: list[int]  # active lanes only; 64-bit for memory, 32-bit for LDS
    marker_id: int = 0  # unique per-op marker ID from funcmap
    source_loc: str = ""  # "file.hip:42" if compiled with -g
    shader_engine: int = 0

    def active_lane_count(self) -> int:
        """Return count of active lanes (same as len(addresses))."""
        return len(self.addresses)


# Map funcmap name prefix -> (kind, is_64bit)
# Funcmap entries may have @source:line suffix (e.g. "addr_trace_load@file.hip:42")
# Order matters: longer/more-specific prefixes first to avoid false matches.
_ADDR_TRACE_PREFIXES = [
    # LDS atomics (before lds_load/lds_store)
    ("addr_trace_lds_atomic",           "lds_atomic",           False),
    ("addr_trace_lds_load",             "lds_load",             False),
    ("addr_trace_lds_store",            "lds_store",            False),
    # Global atomics (before load/store)
    ("addr_trace_atomic",               "atomic",               True),
    ("addr_trace_load",                 "load",                 True),
    ("addr_trace_store",                "store",                True),
    # Buffer ops — component-based protocol (handled separately)
    ("addr_trace_struct_buffer_atomic", "struct_buffer_atomic", None),
    ("addr_trace_struct_buffer_load",   "struct_buffer_load",   None),
    ("addr_trace_struct_buffer_store",  "struct_buffer_store",  None),
    ("addr_trace_buffer_atomic",        "buffer_atomic",        None),
    ("addr_trace_buffer_load",          "buffer_load",          None),
    ("addr_trace_buffer_store",         "buffer_store",         None),
    # ds_permute/ds_bpermute — 32-bit index per lane
    ("addr_trace_ds_bpermute",          "ds_bpermute",          False),
    ("addr_trace_ds_permute",           "ds_permute",           False),
]


def _match_addr_trace(name: str) -> Optional[tuple[str, Optional[bool]]]:
    """Match a funcmap name against address trace prefixes.

    Returns (kind, is_64bit) or None if not an addr trace.
    is_64bit is None for buffer ops (component-based protocol).
    Source location lives separately on FuncMap.source_locs and is threaded
    through parse_address_block by callers.
    """
    for prefix, kind, is_64bit in _ADDR_TRACE_PREFIXES:
        if name == prefix:
            return kind, is_64bit
    return None


def _is_buffer_trace(kind: str) -> bool:
    """Check if trace kind uses the buffer component protocol."""
    return "buffer" in kind


def _is_struct_buffer(kind: str) -> bool:
    """Check if trace kind is a struct buffer (has vindex)."""
    return kind.startswith("struct_buffer")

def parse_address_block(
    records: list[ShaderRecord],
    start_idx: int,
    header_rec: ShaderRecord,
    name: str,
    marker_id: int,
    wave_size: int,
    source_loc: str = "",
) -> tuple[Optional[AddressTrace], int]:
    """Parse an address trace block starting at the header marker.

    Expected record sequence for memory/LDS/atomic/permute:
        [start_idx]   header marker  (already decoded as addr_trace_*)
        [start_idx+1] exec_lo
        [start_idx+2] exec_hi       (0 on wave32)
        [start_idx+3..] lane addresses (2 per lane for 64-bit, 1 for 32-bit)

    For buffer ops (component-based protocol):
        [start_idx]   header marker
        [start_idx+1] exec_lo
        [start_idx+2] exec_hi
        [start_idx+3] rsrc_lo
        [start_idx+4] rsrc_hi
        [start_idx+5] soffset
        [start_idx+6..] per-lane voffset (wave_size tokens)
        (struct only:) per-lane vindex (wave_size tokens)

    The wave_size (from W: funcmap entry) determines how many lane
    addresses to read.  No sentinel -- we trust the thread trace data.

    Returns (AddressTrace, next_index) or (None, next_index) on parse failure.
    """
    match = _match_addr_trace(name)
    if match is None:
        return None, start_idx + 1

    kind, is_64bit = match
    i = start_idx + 1  # skip header

    if wave_size not in (32, 64):
        return None, i

    # Need at least exec_lo + exec_hi
    if i + 1 >= len(records):
        return None, len(records)

    exec_lo = records[i].value & 0xFFFFFFFF
    exec_hi = records[i + 1].value & 0xFFFFFFFF
    exec_mask = exec_lo | (exec_hi << 32)
    i += 2

    # Buffer ops use component-based protocol
    if _is_buffer_trace(kind):
        return _parse_buffer_block(
            records, i, header_rec, kind, source_loc,
            marker_id, wave_size, exec_mask)

    # Standard per-lane address trace (memory, LDS, atomic, permute)
    tokens_per_lane = 2 if is_64bit else 1
    total_tokens = wave_size * tokens_per_lane
    if i + total_tokens > len(records):
        return None, len(records)

    addresses = []
    if is_64bit:
        for lane in range(wave_size):
            addr_lo = records[i].value & 0xFFFFFFFF
            addr_hi = records[i + 1].value & 0xFFFFFFFF
            i += 2
            if (exec_mask >> lane) & 1:
                addresses.append(addr_lo | (addr_hi << 32))
    else:
        for lane in range(wave_size):
            val = records[i].value & 0xFFFFFFFF
            i += 1
            if (exec_mask >> lane) & 1:
                addresses.append(val)

    trace = AddressTrace(
        kind=kind,
        time=header_rec.time,
        cu=header_rec.cu,
        simd=header_rec.simd,
        wave_id=header_rec.wave_id,
        exec_mask=exec_mask,
        addresses=addresses,
        marker_id=marker_id,
        source_loc=source_loc,
        shader_engine=header_rec.shader_engine,
    )
    return trace, i


def _parse_buffer_block(
    records: list[ShaderRecord],
    i: int,
    header_rec: ShaderRecord,
    kind: str,
    source_loc: str,
    marker_id: int,
    wave_size: int,
    exec_mask: int,
) -> tuple[Optional[AddressTrace], int]:
    """Parse buffer component block after exec mask has been read.

    Protocol: rsrc_lo, rsrc_hi, soffset, then per-lane voffset.
    Struct-buffer records contain vindex; consume them but do not guess an
    address without a verified descriptor-stride decoder.
    """
    is_struct = _is_struct_buffer(kind)

    # Fixed tokens: rsrc_lo, rsrc_hi, soffset
    if i + 3 > len(records):
        return None, len(records)

    rsrc_lo = records[i].value & 0xFFFFFFFF
    rsrc_hi = records[i + 1].value & 0xFFFFFFFF
    soffset = records[i + 2].value & 0xFFFFFFFF
    i += 3

    # Per-lane voffset
    if i + wave_size > len(records):
        return None, len(records)

    voffsets = []
    for lane in range(wave_size):
        voffsets.append(records[i].value & 0xFFFFFFFF)
        i += 1

    if is_struct:
        if i + wave_size > len(records):
            return None, len(records)
        return None, i + wave_size

    # Reconstruct addresses for active lanes
    # Base address from rsrc descriptor: bits [47:0]
    base_addr = (rsrc_lo | (rsrc_hi << 32)) & 0xFFFFFFFFFFFF

    addresses = []
    for lane in range(wave_size):
        if (exec_mask >> lane) & 1:
            addr = base_addr + soffset + voffsets[lane]
            addresses.append(addr)

    trace = AddressTrace(
        kind=kind,
        time=header_rec.time,
        cu=header_rec.cu,
        simd=header_rec.simd,
        wave_id=header_rec.wave_id,
        exec_mask=exec_mask,
        addresses=addresses,
        marker_id=marker_id,
        source_loc=source_loc,
        shader_engine=header_rec.shader_engine,
    )
    return trace, i


def preprocess_records(
    records: list[ShaderRecord],
    funcmap: FuncMap,
    resolve_funcmap=None,
) -> tuple[list[ShaderRecord], list[AddressTrace]]:
    """Extract address trace blocks from the record stream.

    Address trace blocks span multiple consecutive s_ttracedata records
    (header + exec + addresses), but the raw shaderdata stream interleaves
    records from all active waves.  We demultiplex by (SE, CU, SIMD, wave).

    Returns:
        markers: all non-address-trace records (for build_stacks)
        addr_traces: structured AddressTrace objects
    """
    markers: list[ShaderRecord] = []
    addr_traces: list[AddressTrace] = []

    # First pass: identify address-trace headers and split records
    # into per-wave streams for those waves that have address traces.
    # Non-address-trace records go directly to markers.
    #
    # A record is an address-trace header if decoding its value gives
    # a marker_id that resolves to an addr_trace_* funcmap entry.
    # All subsequent records from the same wave stream until
    # the next header belong to the same address trace block.

    # Group all records by wave identity, preserving time order
    per_wave: dict[tuple[int, int, int, int], list[ShaderRecord]] = defaultdict(list)
    wave_has_addr_trace: set[tuple[int, int, int, int]] = set()

    for rec in records:
        if rec.flags & 2:  # PRIV: generated by the trap handler
            continue
        key = (rec.shader_engine, rec.cu, rec.simd, rec.wave_id)
        per_wave[key].append(rec)

    def map_for(record: ShaderRecord) -> FuncMap:
        return resolve_funcmap(record) if resolve_funcmap is not None else funcmap

    for key, wave_records in per_wave.items():
        i = 0
        while i < len(wave_records):
            current_map = map_for(wave_records[i])
            marker_id, _enter, _exit_prev = decode_marker(wave_records[i].value, current_map)
            name, _mtype = current_map.resolve(marker_id)
            if _match_addr_trace(name) is not None:
                wave_has_addr_trace.add(key)
                break
            i += 1 + current_map.extra_payload_count(marker_id)

    # Process each wave's records
    for key, wave_records in per_wave.items():
        if key not in wave_has_addr_trace:
            # No address traces in this wave. Still honor generic payload
            # metadata so raw payload records are not decoded as headers.
            i = 0
            while i < len(wave_records):
                rec = wave_records[i]
                current_map = map_for(rec)
                marker_id, _enter, _exit_prev = decode_marker(rec.value, current_map)
                markers.append(rec)
                i += 1 + current_map.extra_payload_count(marker_id)
            continue

        # Parse this wave's contiguous record stream
        i = 0
        while i < len(wave_records):
            rec = wave_records[i]
            current_map = map_for(rec)
            marker_id, enter, exit_prev = decode_marker(rec.value, current_map)
            name, mtype = current_map.resolve(marker_id)

            if _match_addr_trace(name) is not None:
                trace, i = parse_address_block(
                    wave_records, i, rec, name, marker_id, current_map.wave_size,
                    current_map.source_locs.get(marker_id, ""))
                if trace:
                    addr_traces.append(trace)
            else:
                markers.append(rec)
                i += 1 + current_map.extra_payload_count(marker_id)

    # Re-sort markers by time since we processed per-wave
    markers.sort(key=lambda r: r.time)

    return markers, addr_traces


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def discover_base_dir(base: str) -> tuple[list[str], list[str]]:
    """
    Auto-discover trace directories and code objects under a base directory.

    Searches recursively for:
      - ui_*/shaderdata_*.json  -> trace directories
      - *code_object_id*.out   -> code objects
    """
    trace_dirs: list[str] = []
    code_objects: list[str] = []

    for root, dirs, files in os.walk(base):
        for f in files:
            if f.startswith("shaderdata_") and f.endswith(".json"):
                if root not in trace_dirs:
                    trace_dirs.append(root)
            if "code_object_id" in f and f.endswith(".out"):
                code_objects.append(os.path.join(root, f))

    return sorted(trace_dirs), sorted(code_objects)
