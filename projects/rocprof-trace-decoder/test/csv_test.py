#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
CSV integration test tool for rocprof-trace-decoder.

Loads the decoder shared library via ctypes, parses .att binary trace files,
and validates that accumulated per-instruction statistics match expected values
from .csv control files.
"""

import argparse
import ctypes
import csv
import glob
import os
import struct
import sys

# ── Status constants ─────────────────────────────────────────────────────────
_STATUS_SUCCESS              = 0
_STATUS_ERROR_OUT_OF_RESOURCES = 2
_STATUS_ERROR_INVALID_ARGUMENT = 3

# ── Record-type constants ────────────────────────────────────────────────────
_RECORD_WAVE = 3
_RECORD_INFO = 4


# ── ctypes struct mirrors ────────────────────────────────────────────────────

class _PcInfo(ctypes.Structure):
    """rocprofiler_thread_trace_decoder_pc_t"""
    _fields_ = [
        ("address",        ctypes.c_uint64),
        ("code_object_id", ctypes.c_uint64),
    ]


class _InstT(ctypes.Structure):
    """rocprofiler_thread_trace_decoder_inst_t  (bitfield packed in uint32)."""
    _fields_ = [
        ("_cat_stall", ctypes.c_uint32),  # category:8 | stall:24
        ("duration",   ctypes.c_int32),
        ("time",       ctypes.c_int64),
        ("pc",         _PcInfo),
    ]

    @property
    def stall(self):
        return (self._cat_stall >> 8) & 0xFF_FFFF


class _WaveStateT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int32), ("duration", ctypes.c_int32)]


class _WaveT(ctypes.Structure):
    """rocprofiler_thread_trace_decoder_wave_t"""
    _fields_ = [
        ("cu",                ctypes.c_uint8),
        ("simd",              ctypes.c_uint8),
        ("wave_id",           ctypes.c_uint8),
        ("contexts",          ctypes.c_uint8),
        ("_rsvd1",            ctypes.c_uint32),
        ("_rsvd2",            ctypes.c_uint32),
        ("_rsvd3",            ctypes.c_uint32),
        ("begin_time",        ctypes.c_int64),
        ("end_time",          ctypes.c_int64),
        ("timeline_size",     ctypes.c_uint64),
        ("instructions_size", ctypes.c_uint64),
        ("timeline_array",    ctypes.POINTER(_WaveStateT)),
        ("instructions_array", ctypes.POINTER(_InstT)),
    ]


# ── Callback signatures (CFUNCTYPE) ─────────────────────────────────────────

_SE_DATA_CB = ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),     # uint8_t** buffer
    ctypes.POINTER(ctypes.c_uint64),     # uint64_t* buffer_size
    ctypes.c_void_p,                     # void*     userdata
)

_TRACE_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,                        # record_type
    ctypes.c_void_p,                     # void*    trace_events
    ctypes.c_uint64,                     # uint64_t trace_size
    ctypes.c_void_p,                     # void*    userdata
)

_ISA_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,                     # char*     instruction  (writable)
    ctypes.POINTER(ctypes.c_uint64),     # uint64_t* memory_size
    ctypes.POINTER(ctypes.c_uint64),     # uint64_t* size
    _PcInfo,                             # pc        (by value)
    ctypes.c_void_p,                     # void*     userdata
)


# ── Instruction struct layout for bulk unpacking ────────────────────────────
# _cat_stall(u32), duration(i32), time(i64), address(u64), code_object_id(u64)
_INST_FMT  = "<IiqQQ"
_INST_SIZE = struct.calcsize(_INST_FMT)  # 32 bytes


# ── Library loading ──────────────────────────────────────────────────────────

def _load_library(libpath):
    lib = ctypes.CDLL(libpath)

    lib.rocprof_trace_decoder_parse_data.argtypes = [
        _SE_DATA_CB, _TRACE_CB, _ISA_CB, ctypes.c_void_p
    ]
    lib.rocprof_trace_decoder_parse_data.restype = ctypes.c_int

    lib.rocprof_trace_decoder_get_info_string.argtypes = [ctypes.c_int]
    lib.rocprof_trace_decoder_get_info_string.restype  = ctypes.c_char_p

    return lib


# ── CSV parsing ──────────────────────────────────────────────────────────────

def _parse_csv(filepath, instructions):
    """Read a control CSV and accumulate expected values into *instructions*."""
    with open(filepath, "r", newline="") as fh:
        reader = csv.reader(fh)
        headers = [h.strip() for h in next(reader)]

        col = {name: headers.index(name) for name in
               ("CodeObj", "Vaddr", "Instruction", "Hitcount", "Latency", "Stall", "Idle")}

        for row in reader:
            if not row:
                continue
            fields = [c.strip() for c in row]

            key = (int(fields[col["CodeObj"]], 0),
                   int(fields[col["Vaddr"]], 0))

            entry = instructions.setdefault(key, {
                "inst": "", "hitcount": 0, "stall": 0, "latency": 0, "idle": 0,
                "csv_hitcount": 0, "csv_stall": 0, "csv_latency": 0, "csv_idle": 0,
            })
            entry["inst"] = fields[col["Instruction"]]

            hc = fields[col["Hitcount"]]
            la = fields[col["Latency"]]
            st = fields[col["Stall"]]
            il = fields[col["Idle"]]

            if hc: entry["csv_hitcount"] += int(hc, 0)
            if la: entry["csv_latency"]  += int(la, 0)
            if st: entry["csv_stall"]    += int(st, 0)
            if il: entry["csv_idle"]     += int(il, 0)


def _build_next_addr(sorted_keys):
    """Map each key to the byte distance to the next instruction in the same code object."""
    mem = {}
    for i, k in enumerate(sorted_keys):
        if i + 1 < len(sorted_keys) and sorted_keys[i + 1][0] == k[0]:
            mem[k] = sorted_keys[i + 1][1] - k[1]
        else:
            mem[k] = 32
    return mem


# ── Core decode loop ─────────────────────────────────────────────────────────

def _process_shader(lib, file_path, instructions, next_addr, suppress, isa_cache):
    """Read one .att file and decode it through the library."""
    with open(file_path, "rb") as fh:
        raw = fh.read()
    if not raw:
        print(f"Warning: empty file {file_path}", file=sys.stderr)
        return

    data_buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
    consumed = [False]

    def _se_data(buf_pp, size_p, _ud):
        if consumed[0]:
            return 0
        consumed[0] = True
        buf_pp[0] = ctypes.addressof(data_buf)
        size_p[0] = len(data_buf)
        return len(data_buf)

    _unpack_from = struct.unpack_from  # local for speed
    _inst_fmt = _INST_FMT
    _inst_size = _INST_SIZE
    _instructions = instructions

    def _trace(record_type, events, size, _ud):
        if record_type == _RECORD_INFO and not suppress:
            infos = ctypes.cast(events, ctypes.POINTER(ctypes.c_int))
            for i in range(size):
                msg = lib.rocprof_trace_decoder_get_info_string(infos[i])
                if msg:
                    print(f"Warning: {msg.decode()}", file=sys.stderr)

        if record_type != _RECORD_WAVE:
            return _STATUS_SUCCESS

        waves = ctypes.cast(events, ctypes.POINTER(_WaveT))
        for wn in range(size):
            wave = waves[wn]
            n_inst = wave.instructions_size
            if n_inst == 0:
                continue

            prev_t = wave.begin_time
            # Bulk-read all instructions as raw bytes to avoid per-element ctypes overhead
            buf = ctypes.string_at(wave.instructions_array, n_inst * _inst_size)

            off = 0
            for _ in range(n_inst):
                cat_stall, duration, time, addr, coid = _unpack_from(_inst_fmt, buf, off)
                off += _inst_size

                if coid == 0 and addr == 0:
                    continue

                entry = _instructions.get((coid, addr))
                if entry is None:
                    print(f"PC not found: codeobj={coid} addr={addr}", file=sys.stderr)
                    sys.exit(1)

                entry["hitcount"] += 1
                entry["latency"]  += duration
                entry["stall"]    += (cat_stall >> 8) & 0xFF_FFFF
                idle = time - prev_t
                if idle > 0:
                    entry["idle"] += idle
                end = time + duration
                if end > prev_t:
                    prev_t = end

        return _STATUS_SUCCESS

    def _isa(instr_buf, mem_p, size_p, pc, _ud):
        key = (pc.code_object_id, pc.address)
        cached = isa_cache.get(key)
        if cached is None:
            entry = _instructions.get(key)
            if entry is None:
                return _STATUS_ERROR_INVALID_ARGUMENT
            cached = entry["inst"].encode()
            isa_cache[key] = cached

        avail = size_p[0]
        size_p[0] = len(cached)
        if len(cached) > avail:
            return _STATUS_ERROR_OUT_OF_RESOURCES
        ctypes.memmove(instr_buf, cached, len(cached))
        mem_p[0] = next_addr.get(key, 32)
        return _STATUS_SUCCESS

    # prevent GC of callback wrappers during the C call
    cb_se  = _SE_DATA_CB(_se_data)
    cb_tr  = _TRACE_CB(_trace)
    cb_isa = _ISA_CB(_isa)

    status = lib.rocprof_trace_decoder_parse_data(cb_se, cb_tr, cb_isa, None)
    if status != _STATUS_SUCCESS:
        print(f"parse_data returned error {status}", file=sys.stderr)
        sys.exit(1)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CSV integration test for rocprof-trace-decoder")
    parser.add_argument("--lib", required=True, help="Path to the decoder shared library")
    parser.add_argument("--suppress-warnings", action="store_true",
                        help="Suppress info/warning messages from the decoder")
    parser.add_argument("files", nargs="*", help=".att and .csv files (glob patterns supported)")
    args = parser.parse_args()

    if not args.files:
        sys.exit(0)

    att_files = []
    csv_files = []

    for arg in args.files:
        if "*" in arg or "?" in arg:
            expanded = sorted(glob.glob(arg))
            if not expanded:
                print(f"No matches for pattern: {arg}", file=sys.stderr)
                sys.exit(1)
        else:
            expanded = [arg]
        for p in expanded:
            if ".att" in p:
                att_files.append(os.path.abspath(p))
            elif ".csv" in p:
                csv_files.append(os.path.abspath(p))

    lib = _load_library(args.lib)
    suppress = args.suppress_warnings

    instructions = {}
    for cf in csv_files:
        _parse_csv(cf, instructions)

    sorted_keys = sorted(instructions.keys())
    next_addr   = _build_next_addr(sorted_keys)
    isa_cache   = {}

    for af in att_files:
        _process_shader(lib, af, instructions, next_addr, suppress, isa_cache)

    # Validate: accumulated stats must match CSV expected values
    for key, inst in instructions.items():
        #print(inst)
        if (inst["hitcount"] == inst["csv_hitcount"] and
            inst["stall"]   == inst["csv_stall"]    and
            inst["latency"] == inst["csv_latency"]  and
            inst["idle"]    == inst["csv_idle"]):
            continue

        print(f'{inst["inst"]} - PC: {key[0]},{key[1]}')
        print(f'Hitcount: {inst["hitcount"]}/{inst["csv_hitcount"]}, '
              f'Stall: {inst["stall"]}/{inst["csv_stall"]}, '
              f'Latency: {inst["latency"]}/{inst["csv_latency"]}, '
              f'Idle: {inst["idle"]}/{inst["csv_idle"]}')
        sys.exit(1)


if __name__ == "__main__":
    main()
