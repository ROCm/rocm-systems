#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################
"""Catalog peaks for the roofline.

Peaks are published chip numbers, not measured clocks. A kernel on MI250X
runs on one GCD, and ROCm exposes each GCD as its own device, so gfx90a
peaks are half the OAM. Override any field when you have a measured clock.
"""

from __future__ import annotations

from dataclasses import dataclass

# (max VGPRs, waves per EU). Mirrors the occupancy tables used by PerfXpert.
_OCCUPANCY: dict[str, tuple[tuple[int, int], ...]] = {
    "gfx942": (
        (64, 8),
        (80, 6),
        (96, 5),
        (128, 4),
        (160, 3),
        (256, 2),
    ),
    "gfx950": (
        (64, 8),
        (80, 6),
        (96, 5),
        (128, 4),
        (160, 3),
        (256, 2),
    ),
    "gfx90a": (
        (64, 8),
        (80, 6),
        (96, 5),
        (128, 4),
        (160, 3),
        (256, 2),
    ),
    "gfx1100": (
        (24, 16),
        (32, 12),
        (48, 8),
        (96, 4),
        (1536, 1),
    ),
}


@dataclass(frozen=True)
class Device:
    """Peak rates the roofline divides by. Every rate must be positive."""

    name: str
    gfx: str
    cu_count: int
    simd_per_cu: int
    wavefront_size: int
    clock_hz: float
    fp32_vector_flops: float
    fp64_vector_flops: float
    fp16_vector_flops: float
    fp32_matrix_flops: float
    fp64_matrix_flops: float
    fp16_matrix_flops: float
    hbm_bytes_per_s: float
    l2_bytes: float
    hbm_latency_s: float
    lds_bytes_per_cu_per_clock: int
    # 800 on gfx9, matching rocdevice.cpp sgprsPerSimd_. 0 on gfx10+, where
    # SGPRs are not shared across waves and do not cap occupancy.
    sgprs_per_simd: int
    lds_bytes_per_cu: int
    notes: str

    def __post_init__(self) -> None:
        rates = (
            self.cu_count,
            self.simd_per_cu,
            self.wavefront_size,
            self.clock_hz,
            self.fp32_vector_flops,
            self.fp64_vector_flops,
            self.fp16_vector_flops,
            self.fp32_matrix_flops,
            self.fp64_matrix_flops,
            self.fp16_matrix_flops,
            self.hbm_bytes_per_s,
            self.l2_bytes,
            self.hbm_latency_s,
            self.lds_bytes_per_cu_per_clock,
            self.lds_bytes_per_cu,
        )
        if any(r <= 0 for r in rates) or self.sgprs_per_simd < 0:
            raise ValueError(f"device {self.gfx} has a non-positive rate")


def waves_per_eu(gfx: str, vgprs: int) -> int:
    """Resident waves per execution unit at this VGPR count.

    A missing VGPR count (``vgprs <= 0``) uses 64, which is the first
    occupancy bin on CDNA. Callers should say so in the hint notes.
    """

    table = _OCCUPANCY.get(gfx, _OCCUPANCY["gfx942"])
    count = vgprs if vgprs > 0 else 64
    for limit, waves in table:
        if count <= limit:
            return waves
    return 1


_GFX90A = Device(
    name="Instinct MI250X (one GCD)",
    gfx="gfx90a",
    cu_count=110,
    simd_per_cu=4,
    wavefront_size=64,
    clock_hz=1.7e9,
    # Half the published OAM peaks (95.7 / 47.9 / 383 TFLOP/s, 3.2 TB/s).
    fp32_vector_flops=47.85e12,
    fp64_vector_flops=23.95e12,
    # Vector FP16 is not broken out on the OAM sheet. Use the FP32 vector
    # rate so a packed-FP16 kernel is not silently called compute-bound.
    fp16_vector_flops=47.85e12,
    fp32_matrix_flops=47.85e12,
    fp64_matrix_flops=23.95e12,
    fp16_matrix_flops=191.5e12,
    hbm_bytes_per_s=1.6e12,
    l2_bytes=8 * 1024 * 1024,
    hbm_latency_s=4e-7,
    lds_bytes_per_cu_per_clock=128,
    sgprs_per_simd=800,
    lds_bytes_per_cu=64 * 1024,
    notes=(
        "MI250X OAM peaks halved because each GCD is a ROCm device. "
        "FP16 vector peak is unpublished and set equal to FP32 vector."
    ),
)

_GFX942 = Device(
    name="Instinct MI300X",
    gfx="gfx942",
    cu_count=304,
    simd_per_cu=4,
    wavefront_size=64,
    clock_hz=2.1e9,
    # AMD Instinct MI300X data sheet: FP32 vector 163.4, FP64 vector 81.7,
    # FP32/FP64 matrix 163.4 TFLOP/s, HBM3 5.3 TB/s, 304 CUs, 2100 MHz,
    # 256 MB last-level cache. FP16/BF16 matrix 1307.4 TFLOP/s dense.
    fp32_vector_flops=163.4e12,
    fp64_vector_flops=81.7e12,
    fp16_vector_flops=163.4e12,
    fp32_matrix_flops=163.4e12,
    fp64_matrix_flops=163.4e12,
    fp16_matrix_flops=1307.4e12,
    hbm_bytes_per_s=5.3e12,
    l2_bytes=256 * 1024 * 1024,
    hbm_latency_s=4e-7,
    lds_bytes_per_cu_per_clock=128,
    sgprs_per_simd=800,
    lds_bytes_per_cu=64 * 1024,
    notes=(
        "FP16 vector peak is unpublished on the MI300X sheet and is set "
        "equal to the FP32 vector peak. FP16 matrix peak is the dense "
        "1307.4 TFLOP/s figure, not the sparse 2614.9."
    ),
)

_GFX950 = Device(
    name="Instinct MI355X",
    gfx="gfx950",
    cu_count=256,
    simd_per_cu=4,
    wavefront_size=64,
    clock_hz=2.4e9,
    # AMD Instinct MI355X product page: FP32 vector 157.3, FP16 vector 157.3,
    # FP32 matrix 157.3, FP64 vector/matrix 78.6 TFLOP/s, FP16 matrix 2.5
    # PFLOP/s, HBM3E 8 TB/s, 256 CUs, 2400 MHz, 256 MB LLC.
    fp32_vector_flops=157.3e12,
    fp64_vector_flops=78.6e12,
    fp16_vector_flops=157.3e12,
    fp32_matrix_flops=157.3e12,
    fp64_matrix_flops=78.6e12,
    fp16_matrix_flops=2.5e15,
    hbm_bytes_per_s=8.0e12,
    l2_bytes=256 * 1024 * 1024,
    hbm_latency_s=4e-7,
    lds_bytes_per_cu_per_clock=128,
    sgprs_per_simd=800,
    lds_bytes_per_cu=64 * 1024,
    notes="Catalog peaks from the MI355X product page, dense matrix rates.",
)

_GFX1100 = Device(
    name="Radeon RX 7900 XTX",
    gfx="gfx1100",
    cu_count=96,
    simd_per_cu=2,
    wavefront_size=32,
    clock_hz=2.5e9,
    # Product page: up to 61 TFLOP/s FP32, 960 GB/s GDDR6, 96 MB Infinity
    # Cache, 96 CUs, boost up to 2.5 GHz. Wave32. FP64 is a placeholder
    # at 1/64 of FP32 so the rate stays positive.
    fp32_vector_flops=61.0e12,
    fp64_vector_flops=61.0e12 / 64.0,
    fp16_vector_flops=122.0e12,
    fp32_matrix_flops=61.0e12,
    fp64_matrix_flops=61.0e12 / 64.0,
    fp16_matrix_flops=122.0e12,
    hbm_bytes_per_s=960.0e9,
    l2_bytes=96 * 1024 * 1024,
    hbm_latency_s=4e-7,
    lds_bytes_per_cu_per_clock=128,
    sgprs_per_simd=0,
    lds_bytes_per_cu=64 * 1024,
    notes=(
        "FP16 vector is a 2x packed estimate. WMMA peak is not taken from "
        "the product page; matrix peaks equal the vector peaks until overridden. "
        "gfx11 does not share SGPRs across waves, so SGPR count is not an occupancy cap."
    ),
)

_DEVICES = {
    "gfx90a": _GFX90A,
    "gfx942": _GFX942,
    "gfx950": _GFX950,
    "gfx1100": _GFX1100,
}

_ALIASES = {
    "mi250x": "gfx90a",
    "mi300x": "gfx942",
    "mi355x": "gfx950",
    "7900xtx": "gfx1100",
    "rx7900xtx": "gfx1100",
}


def device(name: str) -> Device:
    """Look up a catalog device by gfx id or a short product alias."""

    key = name.strip().lower().replace("-", "").replace(" ", "")
    key = _ALIASES.get(key, key)
    try:
        return _DEVICES[key]
    except KeyError as exc:
        known = ", ".join(sorted(set(_DEVICES) | set(_ALIASES)))
        raise KeyError(f"unknown device {name!r}; known: {known}") from exc


def known_devices() -> list[str]:
    return sorted(_DEVICES)
