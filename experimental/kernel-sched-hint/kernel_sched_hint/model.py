#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################
"""Roofline duration and bound class.

The enqueue-time C++ header ``cxx/kernel_sched_hint.hpp`` implements the
same formulas. Keep the two in sync; ``tests/test_model.py`` checks them.

Per wave, one trip through the lowered ISA:

    t_valu   = passes * waves * sum(dtype_flops / vector_peak)
    t_matrix = passes * waves * sum(mfma_flops / matrix_peak)
    t_hbm    = passes * waves * (global_bytes + scratch_bytes) / hbm_bw
    t_lds    = passes * waves * lds_bytes / (lds_bytes_per_cu_per_clock * cus * clock)
    t_issue  = passes * waves * valu_issues / (cus * simds_per_cu * clock)

    t_throughput = max(t_valu, t_matrix, t_hbm, t_lds, t_issue)

    chain is the exposed dependency depth (waitcnt, else one memory
    round trip, plus special-function latency). It is hidden as the
    launch fills the machine:

    hide = min(1, waves / resident_waves)
    t_latency = chain * passes * (1 - hide)
    t_launch = 5e-6

    roofline = t_throughput + t_latency + t_launch
    duration = t_throughput * scale + t_latency + t_launch

``scale`` is 1/0.35 until a calibration or a fitted residual replaces it.
VALU, matrix, and HBM are overlapped with each other (the max), which is
the optimistic roofline. The scale is where real kernels miss that peak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from kernel_sched_hint.devices import Device, waves_per_eu

LAUNCH_OVERHEAD_S = 5e-6
DEFAULT_THROUGHPUT_SCALE = 1.0 / 0.35
LOW_EFFICIENCY = 0.10
LDS_LATENCY_CYCLES = 30.0
SFU_LATENCY_CYCLES = 16.0
MIXED_RATIO = 0.75
# Learned scales outside this range are clipped. The low end allows a
# kernel to beat the HBM roofline when the working set hits L2.
SCALE_MIN = 0.25
SCALE_MAX = 50.0

_ALU = frozenset({"valu", "matrix", "issue"})
_MEM = frozenset({"hbm", "lds"})


@dataclass
class KernelWork:
    """Static work of one wavefront for one trip through the ISA."""

    name: str = "kernel"
    valu_f32_flops: float = 0.0
    valu_f64_flops: float = 0.0
    valu_f16_flops: float = 0.0
    mfma_f32_flops: float = 0.0
    mfma_f64_flops: float = 0.0
    mfma_f16_flops: float = 0.0
    global_bytes: float = 0.0
    lds_bytes: float = 0.0
    scratch_bytes: float = 0.0
    atomic_ops: float = 0.0
    mem_ops: float = 0.0
    lds_ops: float = 0.0
    waitcnts: float = 0.0
    branches: float = 0.0
    sfu_ops: float = 0.0
    valu_issues: float = 0.0
    vgprs: int = 0
    sgprs: int = 0
    wavefront_size: int = 64
    has_control_flow: bool = False


@dataclass
class Terms:
    t_valu: float
    t_matrix: float
    t_hbm: float
    t_lds: float
    t_issue: float
    t_latency: float
    t_launch: float
    t_throughput: float
    flops: float
    bytes: float
    resident_waves: float
    hide: float
    waves_per_eu: int
    assumed_vgprs: int


@dataclass
class Prediction:
    name: str
    bound: str
    resource: str
    overlap_with: str
    roofline_s: float
    duration_s: float
    duration_lo_s: float
    duration_hi_s: float
    arithmetic_intensity: float
    flops: float
    bytes: float
    waves: int
    trip_count: int
    class_confidence: str
    time_confidence: str
    throughput_scale: float
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        data = dict(self.__dict__)
        if math.isinf(data["arithmetic_intensity"]):
            data["arithmetic_intensity"] = None
        return data


def wave_count(
    grid: tuple[int, int, int], block: tuple[int, int, int], wavefront_size: int
) -> int:
    threads = 1
    for n in (*grid, *block):
        if n <= 0:
            raise ValueError(f"launch dimension must be positive, got {n}")
        threads *= int(n)
    wf = max(1, int(wavefront_size))
    return max(1, math.ceil(threads / wf))


def evaluate(
    work: KernelWork,
    dev: Device,
    waves: float,
    passes: float,
    *,
    lds_bytes_per_workgroup: int = 0,
    workgroup_size: int = 0,
    global_byte_cap: float | None = None,
) -> Terms:
    waves = float(waves)
    passes = float(passes)
    if waves <= 0 or passes <= 0:
        raise ValueError("waves and passes must be positive")

    assumed = work.vgprs if work.vgprs > 0 else 64
    wpe = waves_per_eu(dev.gfx, assumed)
    if dev.sgprs_per_simd > 0 and work.sgprs > 0:
        wpe = min(wpe, max(1, dev.sgprs_per_simd // work.sgprs))
    resident = float(dev.cu_count * dev.simd_per_cu * wpe)
    if lds_bytes_per_workgroup > 0 and workgroup_size > 0 and dev.lds_bytes_per_cu > 0:
        wg_per_cu = max(1, dev.lds_bytes_per_cu // lds_bytes_per_workgroup)
        waves_per_wg = max(1, math.ceil(workgroup_size / max(1, work.wavefront_size)))
        resident = min(resident, float(dev.cu_count * wg_per_cu * waves_per_wg))
    if resident < 1.0:
        resident = 1.0
    hide = min(1.0, waves / resident)

    factor = passes * waves
    t_valu = factor * (
        work.valu_f32_flops / dev.fp32_vector_flops
        + work.valu_f64_flops / dev.fp64_vector_flops
        + work.valu_f16_flops / dev.fp16_vector_flops
    )
    t_matrix = factor * (
        work.mfma_f32_flops / dev.fp32_matrix_flops
        + work.mfma_f64_flops / dev.fp64_matrix_flops
        + work.mfma_f16_flops / dev.fp16_matrix_flops
    )
    global_bytes_total = factor * work.global_bytes
    if global_byte_cap is not None:
        global_bytes_total = min(global_bytes_total, float(global_byte_cap))
    scratch_bytes_total = factor * work.scratch_bytes
    t_hbm = (global_bytes_total + scratch_bytes_total) / dev.hbm_bytes_per_s
    lds_bw = dev.lds_bytes_per_cu_per_clock * dev.cu_count * dev.clock_hz
    t_lds = factor * work.lds_bytes / lds_bw
    issue_slots = dev.cu_count * dev.simd_per_cu * dev.clock_hz
    t_issue = factor * work.valu_issues / issue_slots
    t_throughput = max(t_valu, t_matrix, t_hbm, t_lds, t_issue)

    has_global = (work.mem_ops + work.atomic_ops) > 0 or (
        work.global_bytes + work.scratch_bytes
    ) > 0
    has_lds = work.lds_ops > 0 or work.lds_bytes > 0
    chain = 0.0
    if work.waitcnts > 0:
        if has_global or not has_lds:
            chain += work.waitcnts * dev.hbm_latency_s
        else:
            chain += work.waitcnts * (LDS_LATENCY_CYCLES / dev.clock_hz)
    elif has_global:
        chain += dev.hbm_latency_s
    elif has_lds:
        chain += LDS_LATENCY_CYCLES / dev.clock_hz
    if work.sfu_ops > 0:
        chain += work.sfu_ops * (SFU_LATENCY_CYCLES / dev.clock_hz)
    t_latency = chain * passes * (1.0 - hide)

    flops = factor * (
        work.valu_f32_flops
        + work.valu_f64_flops
        + work.valu_f16_flops
        + work.mfma_f32_flops
        + work.mfma_f64_flops
        + work.mfma_f16_flops
    )
    nbytes = global_bytes_total + scratch_bytes_total
    return Terms(
        t_valu=t_valu,
        t_matrix=t_matrix,
        t_hbm=t_hbm,
        t_lds=t_lds,
        t_issue=t_issue,
        t_latency=t_latency,
        t_launch=LAUNCH_OVERHEAD_S,
        t_throughput=t_throughput,
        flops=flops,
        bytes=nbytes,
        resident_waves=resident,
        hide=hide,
        waves_per_eu=wpe,
        assumed_vgprs=assumed,
    )


def _resource_class(resource: str) -> str:
    if resource in _ALU:
        return "alu"
    if resource in _MEM:
        return "memory"
    if resource == "latency":
        return "latency"
    return "unknown"


def classify(terms: Terms) -> tuple[str, str, float]:
    """Return ``(bound, resource, runner_up_ratio)``."""

    comps = {
        "valu": terms.t_valu,
        "matrix": terms.t_matrix,
        "hbm": terms.t_hbm,
        "lds": terms.t_lds,
        "issue": terms.t_issue,
        "latency": terms.t_latency,
    }
    ranked = sorted(comps.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_v = ranked[0]
    if top_v <= 0.0:
        return "unknown", "none", 0.0
    second_v = ranked[1][1]
    ratio = second_v / top_v
    bound = _resource_class(top_name)
    for name, val in ranked[1:]:
        if val <= 0.0:
            break
        other = _resource_class(name)
        if other != bound and val / top_v >= MIXED_RATIO:
            bound = "mixed"
            break
    return bound, top_name, ratio


def overlap_with(bound: str, resource: str) -> str:
    if bound in ("mixed", "unknown"):
        return "none"
    if resource == "lds":
        return "hbm"
    if bound == "memory":
        return "alu"
    if bound == "alu":
        return "memory"
    if bound == "latency":
        return "either"
    return "none"


def phi_vector(
    work: KernelWork, dev: Device, waves: float, terms: Terms
) -> list[float]:
    """Eight features for the optional log-scale residual. The last is bias."""

    if terms.bytes > 0.0:
        ai = terms.flops / terms.bytes
    else:
        ai = 1.0e6
    ridge = dev.fp32_vector_flops / dev.hbm_bytes_per_s
    return [
        math.log1p(terms.flops),
        math.log1p(terms.bytes),
        math.log1p(waves),
        ai / (ai + ridge),
        math.log1p(work.vgprs if work.vgprs > 0 else 64),
        1.0 if work.has_control_flow else 0.0,
        work.lds_bytes / (work.lds_bytes + work.global_bytes + 1.0),
        1.0,
    ]


def scale_from_weights(phi: list[float], weights: list[float]) -> float:
    if len(phi) != len(weights):
        raise ValueError(f"expected {len(phi)} weights, got {len(weights)}")
    dot = sum(a * b for a, b in zip(phi, weights))
    lo = math.log(SCALE_MIN)
    hi = math.log(SCALE_MAX)
    if dot < lo:
        dot = lo
    elif dot > hi:
        dot = hi
    return math.exp(dot)


def _class_confidence(bound: str, ratio: float, terms: Terms) -> str:
    if bound in ("unknown", "mixed"):
        return "low"
    if terms.t_throughput + terms.t_latency < terms.t_launch:
        return "low"
    if ratio >= 0.4:
        return "medium"
    return "high"


def _time_confidence(source: str, trip_known: bool) -> str:
    if not trip_known:
        return "low"
    if source == "calibrator":
        return "high"
    if source == "weights":
        return "medium"
    return "low"


def predict_work(
    work: KernelWork,
    dev: Device,
    waves: int,
    *,
    trip_count: int = 1,
    trip_count_known: bool = False,
    scale: float | None = None,
    weights: list[float] | None = None,
    calibrator: Calibrator | None = None,
    lds_bytes_per_workgroup: int = 0,
    workgroup_size: int = 0,
    global_byte_cap: float | None = None,
    metadata_only: bool = False,
    extra_notes: list[str] | None = None,
) -> Prediction:
    if trip_count <= 0:
        raise ValueError("trip_count must be positive")
    terms = evaluate(
        work,
        dev,
        waves,
        trip_count,
        lds_bytes_per_workgroup=lds_bytes_per_workgroup,
        workgroup_size=workgroup_size,
        global_byte_cap=global_byte_cap,
    )
    bound, resource, ratio = classify(terms)
    trip_known = trip_count_known or not work.has_control_flow

    if weights is not None:
        used = scale_from_weights(phi_vector(work, dev, waves, terms), weights)
        source = "weights"
    elif scale is not None:
        used = float(scale)
        source = "override"
    elif (
        calibrator is not None
        and bound not in ("unknown", "mixed")
        and calibrator.has(dev.gfx, bound)
    ):
        used = calibrator.scale_for(dev.gfx, bound)
        source = "calibrator"
    else:
        used = DEFAULT_THROUGHPUT_SCALE
        source = "prior"

    roofline = terms.t_throughput + terms.t_latency + terms.t_launch
    duration = terms.t_throughput * used + terms.t_latency + terms.t_launch
    hi = terms.t_throughput / LOW_EFFICIENCY + terms.t_latency + terms.t_launch
    if terms.bytes > 0.0:
        ai: float = terms.flops / terms.bytes
    else:
        ai = math.inf

    notes: list[str] = []
    if extra_notes:
        notes.extend(extra_notes)
    if metadata_only:
        notes.append(
            "Instruction mix was not counted. Traffic is a lower bound from "
            "COMGR argument kinds, the private segment, and ISA size."
        )
    if source == "prior":
        notes.append(
            "Duration scales the throughput term by 1/0.35. "
            "That prior is a stand-in until a one-shot measurement replaces it."
        )
    if work.has_control_flow and not trip_count_known:
        notes.append(
            "Trip count is unknown. ALU versus memory is unchanged by the "
            "trip count; the duration is not."
        )
    if work.vgprs <= 0:
        notes.append("VGPR count is missing. Occupancy assumes 64 VGPRs.")
    if terms.t_throughput + terms.t_latency < terms.t_launch:
        notes.append(
            "Estimated work is shorter than the dispatch floor. "
            "Reordering this kernel will not move the schedule."
        )
    if 0.0 < terms.bytes < dev.l2_bytes and resource == "hbm":
        notes.append(
            "Requested traffic fits in the last-level cache. "
            "HBM time is an overestimate if the working set actually hits."
        )
    notes.append(dev.notes)

    class_confidence = _class_confidence(bound, ratio, terms)
    if metadata_only:
        class_confidence = "low"

    return Prediction(
        name=work.name,
        bound=bound,
        resource=resource,
        overlap_with=overlap_with(bound, resource),
        roofline_s=roofline,
        duration_s=duration,
        duration_lo_s=min(roofline, duration),
        duration_hi_s=max(hi, duration),
        arithmetic_intensity=ai,
        flops=terms.flops,
        bytes=terms.bytes,
        waves=int(waves),
        trip_count=int(trip_count),
        class_confidence=class_confidence,
        time_confidence=_time_confidence(source, trip_known),
        throughput_scale=used,
        notes=notes,
    )


class Calibrator:
    """EMA of the throughput scale, keyed by device and bound class.

    ``scale`` multiplies ``t_throughput``. A measurement implies

        scale = (measured_s - t_latency - t_launch) / t_throughput
    """

    def __init__(self) -> None:
        self._scale: dict[tuple[str, str], tuple[float, int]] = {}

    def has(self, gfx: str, bound: str) -> bool:
        return (gfx, bound) in self._scale

    def scale_for(self, gfx: str, bound: str) -> float:
        return self._scale[(gfx, bound)][0]

    def samples(self, gfx: str, bound: str) -> int:
        return self._scale.get((gfx, bound), (0.0, 0))[1]

    def observe(
        self,
        gfx: str,
        bound: str,
        measured_s: float,
        terms: Terms,
        alpha: float = 0.2,
    ) -> float | None:
        variable = measured_s - terms.t_latency - terms.t_launch
        if terms.t_throughput <= 0.0 or variable <= 0.0:
            return None
        sample = variable / terms.t_throughput
        if sample < SCALE_MIN:
            sample = SCALE_MIN
        elif sample > SCALE_MAX:
            sample = SCALE_MAX
        key = (gfx, bound)
        if key not in self._scale:
            self._scale[key] = (sample, 1)
        else:
            prev, n = self._scale[key]
            self._scale[key] = ((1.0 - alpha) * prev + alpha * sample, n + 1)
        return self._scale[key][0]

    def to_dict(self) -> dict:
        return {
            f"{gfx}|{bound}": {"scale": scale, "n": n}
            for (gfx, bound), (scale, n) in sorted(self._scale.items())
        }

    def load_dict(self, data: dict) -> None:
        self._scale.clear()
        for key, rec in data.items():
            gfx, bound = key.split("|", 1)
            self._scale[(gfx, bound)] = (float(rec["scale"]), int(rec["n"]))
