#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################
"""Instruction mix from LLVM's AMDGPU lowering.

This is the part COMGR metadata does not carry: how many VALU flops, MFMA
flops, and memory bytes one wave issues per trip through the machine code.
VGPR, SGPR, argument kinds, and segment sizes stay in ``metadata.py``.

Counts that scale with the wavefront (VALU, ordinary memory) are per lane.
MFMA and WMMA are per wave. ``scale`` multiplies the per-lane counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SUFFIXES = ("_e64", "_e32", "_dpp16", "_dpp8", "_dpp", "_sdwa")
_WIDTHS = (
    ("dwordx4", 16),
    ("dwordx3", 12),
    ("dwordx2", 8),
    ("b128", 16),
    ("b96", 12),
    ("b64", 8),
    ("b32", 4),
    ("b16", 2),
    ("b8", 1),
    ("dword", 4),
    ("ubyte", 1),
    ("sbyte", 1),
    ("byte", 1),
    ("ushort", 2),
    ("sshort", 2),
    ("short", 2),
)
_MFMA = re.compile(
    r"^v_(mfma|wmma|smfmac)_([a-z0-9]+)_(\d+)x(\d+)x(\d+)([a-z0-9]*)$",
    re.I,
)
_IR_MFMA = re.compile(
    r"llvm\.amdgcn\.(mfma|wmma)\.([a-z0-9]+)\.(\d+)x(\d+)x(\d+)([a-z0-9]*)",
    re.I,
)


@dataclass
class IsaCounts:
    """Per-lane VALU and memory counts, plus per-wave matrix and issue counts."""

    name: str = "kernel"
    valu_f32_per_lane: float = 0.0
    valu_f64_per_lane: float = 0.0
    valu_f16_per_lane: float = 0.0
    global_bytes_per_lane: float = 0.0
    lds_bytes_per_lane: float = 0.0
    scratch_bytes_per_lane: float = 0.0
    mfma_f32: float = 0.0
    mfma_f64: float = 0.0
    mfma_f16: float = 0.0
    atomic_ops: float = 0.0
    mem_ops: float = 0.0
    lds_ops: float = 0.0
    waitcnts: float = 0.0
    branches: float = 0.0
    sfu_ops: float = 0.0
    valu_issues: float = 0.0
    has_control_flow: bool = False
    lines: list[str] = field(default_factory=list, repr=False)


def _stem(opcode: str) -> str:
    stem = opcode.split(".")[0]
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def _width(stem: str) -> int:
    for key, nbytes in _WIDTHS:
        if key in stem:
            width = nbytes
            break
    else:
        width = 4
    if re.search(r"read2|write2|load2|store2", stem):
        width *= 2
    return width


def _matrix_bucket(out_type: str, in_type: str) -> str:
    src = (in_type or out_type).lower()
    if "f64" in src or "f64" in out_type.lower():
        return "f64"
    if any(token in src for token in ("f16", "bf16", "bf8", "fp8", "f8", "i8", "u8")):
        return "f16"
    return "f32"


def _add_mfma(
    counts: IsaCounts, out_type: str, m: int, n: int, k: int, in_type: str
) -> None:
    flops = float(2 * m * n * k)
    bucket = _matrix_bucket(out_type, in_type)
    if bucket == "f64":
        counts.mfma_f64 += flops
    elif bucket == "f16":
        counts.mfma_f16 += flops
    else:
        counts.mfma_f32 += flops


def _add_lane_flops(counts: IsaCounts, dtype: str, flops: float) -> None:
    if dtype == "f64":
        counts.valu_f64_per_lane += flops
    elif dtype == "f16":
        counts.valu_f16_per_lane += flops
    else:
        counts.valu_f32_per_lane += flops
    counts.valu_issues += 1


def accumulate_opcode(counts: IsaCounts, opcode: str) -> None:
    stem = _stem(opcode)
    mfma = _MFMA.match(stem)
    if mfma:
        _add_mfma(
            counts,
            mfma.group(2),
            int(mfma.group(3)),
            int(mfma.group(4)),
            int(mfma.group(5)),
            mfma.group(6),
        )
        return

    kind = _mem_kind(stem)
    if kind is not None:
        width = _width(stem)
        if kind == "atomic_global":
            counts.global_bytes_per_lane += 2 * width
            counts.atomic_ops += 1
            counts.mem_ops += 1
        elif kind == "scratch":
            counts.scratch_bytes_per_lane += width
            counts.mem_ops += 1
        elif kind == "global":
            counts.global_bytes_per_lane += width
            counts.mem_ops += 1
        elif kind == "lds_atomic":
            counts.lds_bytes_per_lane += 2 * width
            counts.lds_ops += 1
            counts.atomic_ops += 1
        else:
            counts.lds_bytes_per_lane += width
            counts.lds_ops += 1
        return

    if stem.startswith("s_wait") or stem.startswith("s_barrier"):
        counts.waitcnts += 1
        return
    if (
        stem.startswith("s_cbranch")
        or stem == "s_branch"
        or stem.startswith("s_branch")
    ):
        counts.branches += 1
        counts.has_control_flow = True
        return
    if not stem.startswith("v_"):
        return

    if stem.startswith(
        ("v_rcp", "v_sqrt", "v_rsq", "v_exp", "v_log", "v_sin", "v_cos", "v_div_")
    ):
        counts.sfu_ops += 1
        counts.valu_issues += 1
        return

    dot = re.match(r"v_dot(\d+)", stem)
    if dot:
        flops = 2 * int(dot.group(1))
        dtype = "f16" if ("f16" in stem or "bf16" in stem) else "f32"
        _add_lane_flops(counts, dtype, flops)
        return

    dtype = _float_dtype(stem)
    if dtype is not None:
        packed = 2 if stem.startswith("v_pk_") else 1
        if any(token in stem for token in ("fma", "fmac")) or re.search(
            r"v_(?:pk_)?mad_f", stem
        ):
            _add_lane_flops(counts, dtype, 2 * packed)
        elif any(token in stem for token in ("add", "sub", "mul", "max", "min")):
            _add_lane_flops(counts, dtype, packed)
        else:
            counts.valu_issues += 1
        return

    if re.search(r"v_(?:pk_)?(?:add|sub|mul|mad)(?:_|[0-9])", stem):
        flops = 2.0 if "mad" in stem else 1.0
        if stem.startswith("v_pk_"):
            flops *= 2
        _add_lane_flops(counts, "f32", flops)
        return
    counts.valu_issues += 1


def _float_dtype(stem: str) -> str | None:
    if "f64" in stem:
        return "f64"
    if "f16" in stem or "bf16" in stem:
        return "f16"
    if "f32" in stem:
        return "f32"
    return None


def _mem_kind(stem: str) -> str | None:
    if stem.startswith(("global_", "buffer_", "flat_", "scratch_")):
        if "atomic" in stem:
            return "atomic_global"
        if stem.startswith("scratch_"):
            return "scratch"
        if any(token in stem for token in ("load", "store")):
            return "global"
        return None
    if stem.startswith(("ds_read", "ds_write", "ds_load", "ds_store")):
        return "lds"
    if stem.startswith("ds_") and any(
        token in stem
        for token in (
            "add",
            "sub",
            "inc",
            "dec",
            "min",
            "max",
            "and",
            "or",
            "xor",
            "cmp",
        )
    ):
        return "lds_atomic"
    return None


def _asm_line(raw: str) -> str:
    line = raw.split(";", 1)[0].split("//", 1)[0].strip()
    line = re.sub(r"^[0-9a-fA-F]+:\s*", "", line)
    return line.strip()


def parse_assembly(text: str) -> list[IsaCounts]:
    """Split an ``llc`` or ``llvm-objdump -d`` listing into kernels."""

    kernels: list[IsaCounts] = []
    current: IsaCounts | None = None
    for raw in text.splitlines():
        line = _asm_line(raw)
        if not line or line.startswith("."):
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            label = re.sub(r"^[0-9a-fA-F]+\s+", "", label)
            label = label.strip("<>")
            if "." in label or label.startswith("LBB"):
                continue
            if current is not None and current.lines:
                kernels.append(current)
            current = IsaCounts(name=label)
            continue
        if current is None:
            continue
        opcode = line.split()[0].rstrip(",")
        if opcode.startswith("."):
            continue
        current.lines.append(line)
        accumulate_opcode(current, opcode)
        if opcode == "s_endpgm" or opcode.startswith("s_endpgm"):
            kernels.append(current)
            current = None
    if current is not None and current.lines:
        kernels.append(current)
    return kernels


_TYPE = r"(<\d+\s+x\s+[^>]+>|[^\s,]+)"
_ARITH = re.compile(rf"\b(fadd|fsub|fmul|fdiv|frem|add|sub|mul|udiv|sdiv)\s+{_TYPE}\b")
_LOAD = re.compile(
    rf"\bload\s+(?:atomic\s+)?{_TYPE}\s*,\s*ptr(?:\s+addrspace\((\d+)\))?"
)
_STORE = re.compile(
    rf"\bstore\s+(?:atomic\s+)?{_TYPE}\s+\S+\s*,\s*ptr(?:\s+addrspace\((\d+)\))?"
)
_LOAD_TYPED = re.compile(
    rf"\bload\s+(?:atomic\s+)?{_TYPE}\s*,\s*{_TYPE}\s+addrspace\((\d+)\)\*"
)
_STORE_TYPED = re.compile(
    rf"\bstore\s+(?:atomic\s+)?{_TYPE}\s+\S+\s*,\s*{_TYPE}\s+addrspace\((\d+)\)\*"
)
_FMA = re.compile(rf"\bcall\s+{_TYPE}\s+@(?:llvm\.fma\.|llvm\.fmuladd\.)")


def _type_lanes(type_token: str) -> tuple[int, str]:
    match = re.fullmatch(r"<(\d+)\s+x\s+(.+)>", type_token.strip())
    if match:
        return int(match.group(1)), match.group(2).strip()
    return 1, type_token.strip()


def _scalar_bytes(scalar: str) -> int:
    return {
        "double": 8,
        "float": 4,
        "half": 2,
        "bfloat": 2,
        "i64": 8,
        "i32": 4,
        "i16": 2,
        "i8": 1,
        "i1": 1,
        "ptr": 8,
    }.get(scalar, 4)


def _scalar_dtype(scalar: str) -> str:
    if scalar in ("double", "f64"):
        return "f64"
    if scalar in ("half", "bfloat") or "f16" in scalar or "bf16" in scalar:
        return "f16"
    if scalar in ("float", "f32"):
        return "f32"
    return "int"


def _accumulate_ir_line(counts: IsaCounts, line: str) -> None:
    mfma = _IR_MFMA.search(line)
    if mfma:
        _add_mfma(
            counts,
            mfma.group(2),
            int(mfma.group(3)),
            int(mfma.group(4)),
            int(mfma.group(5)),
            mfma.group(6),
        )
        return
    if "atomicrmw" in line or "cmpxchg" in line:
        counts.global_bytes_per_lane += 8
        counts.atomic_ops += 1
        counts.mem_ops += 1
        return

    loaded = (
        _LOAD.search(line)
        or _STORE.search(line)
        or _LOAD_TYPED.search(line)
        or _STORE_TYPED.search(line)
    )
    if loaded:
        lanes, scalar = _type_lanes(loaded.group(1))
        addrspace = loaded.groups()[-1] or "1"
        nbytes = lanes * _scalar_bytes(scalar)
        if addrspace == "3":
            counts.lds_bytes_per_lane += nbytes
            counts.lds_ops += 1
        elif addrspace == "5":
            counts.scratch_bytes_per_lane += nbytes
            counts.mem_ops += 1
        else:
            counts.global_bytes_per_lane += nbytes
            counts.mem_ops += 1
        return

    fma = _FMA.search(line)
    if fma:
        lanes, scalar = _type_lanes(fma.group(1))
        dtype = _scalar_dtype(scalar)
        if dtype == "int":
            dtype = "f32"
        _add_lane_flops(counts, dtype, 2 * lanes)
        return

    if re.search(
        r"\b(call\s+.*llvm\.(sqrt|amdgcn\.rcp|amdgcn\.rsq)|fdiv|frem|udiv|sdiv)\b", line
    ):
        counts.sfu_ops += 1
        counts.valu_issues += 1
        return

    arith = _ARITH.search(line)
    if arith:
        op = arith.group(1)
        lanes, scalar = _type_lanes(arith.group(2))
        if op in ("fdiv", "frem", "udiv", "sdiv"):
            counts.sfu_ops += 1
            counts.valu_issues += 1
            return
        dtype = _scalar_dtype(scalar)
        if dtype == "int":
            dtype = "f32"
        _add_lane_flops(counts, dtype, lanes)
        return

    if re.search(r"\bbr\b", line):
        counts.branches += 1
        counts.has_control_flow = True


def parse_llvm_ir(text: str) -> list[IsaCounts]:
    parts = re.split(r"(?=^\s*define\s)", text, flags=re.M)
    defines = []
    for part in parts:
        if not re.match(r"\s*define\s", part):
            continue
        header, _, rest = part.partition("{")
        defines.append((header, rest))
    chosen = [item for item in defines if "amdgpu_kernel" in item[0]] or defines
    kernels = []
    for header, rest in chosen:
        name_match = re.search(r"@([\w$.]+)\(", header)
        counts = IsaCounts(name=name_match.group(1) if name_match else "kernel")
        for line in rest.splitlines():
            if line.startswith("}"):
                break
            _accumulate_ir_line(counts, line)
        kernels.append(counts)
    return kernels


def looks_like_llvm_ir(text: str) -> bool:
    return (
        re.search(r"^\s*define\s+", text, re.M) is not None and "s_endpgm" not in text
    )


def parse_isa(text: str) -> list[IsaCounts]:
    if looks_like_llvm_ir(text):
        return parse_llvm_ir(text)
    return parse_assembly(text)
