#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################
"""Build a scheduling hint from COMGR metadata plus an optional ISA mix.

Enqueue time is ``VirtualGPU::submitKernelInternal``. By then the kernel
already carries the COMGR record, and the call has the grid and
``sharedMemBytes``. The instruction mix is computed once when the code
object is loaded and cached; this function only scales it.
"""

from __future__ import annotations

from kernel_sched_hint.devices import Device, device
from kernel_sched_hint.isa import IsaCounts, parse_isa
from kernel_sched_hint.metadata import KernelMetadata, parse_amdgpu_metadata
from kernel_sched_hint.model import (
    KernelWork,
    Prediction,
    predict_work,
    wave_count,
)


def counts_to_work(counts: IsaCounts, wavefront: int) -> KernelWork:
    wf = max(1, wavefront)
    return KernelWork(
        name=counts.name,
        valu_f32_flops=counts.valu_f32_per_lane * wf,
        valu_f64_flops=counts.valu_f64_per_lane * wf,
        valu_f16_flops=counts.valu_f16_per_lane * wf,
        mfma_f32_flops=counts.mfma_f32,
        mfma_f64_flops=counts.mfma_f64,
        mfma_f16_flops=counts.mfma_f16,
        global_bytes=counts.global_bytes_per_lane * wf,
        lds_bytes=counts.lds_bytes_per_lane * wf,
        scratch_bytes=counts.scratch_bytes_per_lane * wf,
        atomic_ops=counts.atomic_ops,
        mem_ops=counts.mem_ops,
        lds_ops=counts.lds_ops,
        waitcnts=counts.waitcnts,
        branches=counts.branches,
        sfu_ops=counts.sfu_ops,
        valu_issues=counts.valu_issues,
        wavefront_size=wf,
        has_control_flow=counts.has_control_flow,
    )


def _isa_is_empty(counts: IsaCounts | None) -> bool:
    if counts is None:
        return True
    return (
        counts.valu_issues == 0
        and counts.mem_ops == 0
        and counts.lds_ops == 0
        and counts.mfma_f32 == 0
        and counts.mfma_f64 == 0
        and counts.mfma_f16 == 0
        and counts.sfu_ops == 0
    )


def apply_metadata(
    work: KernelWork, meta: KernelMetadata, *, have_isa: bool
) -> list[str]:
    """Overlay the COMGR record. ISA traffic wins when the mix was counted."""

    notes: list[str] = []
    if meta.vgprs > 0:
        work.vgprs = meta.vgprs
    if meta.sgprs > 0:
        work.sgprs = meta.sgprs
    if meta.wavefront_size > 0:
        work.wavefront_size = meta.wavefront_size
    if meta.name and work.name in ("", "kernel"):
        work.name = meta.name

    wf = max(1, work.wavefront_size)
    private = meta.private_segment_fixed_size * wf
    if private > work.scratch_bytes:
        if work.scratch_bytes == 0 and private > 0:
            work.mem_ops += 1
        work.scratch_bytes = private
        notes.append(
            "Private segment from COMGR is scratch traffic per wave "
            f"({meta.private_segment_fixed_size} bytes/thread)."
        )
    if meta.vgpr_spill_count or meta.sgpr_spill_count:
        notes.append(
            f"Spills: {meta.vgpr_spill_count} VGPR, {meta.sgpr_spill_count} SGPR."
        )

    if not have_isa:
        buffers = meta.global_buffers()
        work.global_bytes = len(buffers) * 4 * wf
        if buffers:
            work.mem_ops += len(buffers)
        if meta.isa_size > 0:
            work.valu_issues = meta.isa_size / 4.0
            notes.append(
                f"ISA size {meta.isa_size} bytes is standing in for the instruction mix "
                "(about one issue per 4 bytes)."
            )
        notes.append(
            f"{len(meta.arguments)} arguments, {len(buffers)} global buffers, "
            f"{meta.vgprs} VGPRs, {meta.sgprs} SGPRs."
        )
    else:
        notes.append(
            f"COMGR record: {len(meta.arguments)} arguments, "
            f"{len(meta.global_buffers())} global buffers, "
            f"{meta.vgprs} VGPRs, {meta.sgprs} SGPRs."
        )
    if meta.code_object_size and not meta.isa_size:
        notes.append(
            "code_object_size is the whole ISA blob from amd_comgr_lookup_code_object, "
            "not this kernel's machine-code size."
        )
    return notes


def _pair(
    isas: list[IsaCounts], metas: list[KernelMetadata]
) -> list[tuple[IsaCounts | None, KernelMetadata | None]]:
    by_name = {meta.name: meta for meta in metas if meta.name}
    used: set[int] = set()
    paired: list[tuple[IsaCounts | None, KernelMetadata | None]] = []
    for counts in isas:
        meta = by_name.get(counts.name)
        if meta is None and len(isas) == 1 and len(metas) == 1:
            meta = metas[0]
        if meta is not None:
            used.add(id(meta))
        paired.append((counts, meta))
    if not isas:
        for meta in metas:
            paired.append((None, meta))
    else:
        for meta in metas:
            if (
                id(meta) not in used
                and meta.name
                and meta.name not in {c.name for c, _ in paired}
            ):
                paired.append((None, meta))
    return paired


def predict_metadata(
    meta: KernelMetadata,
    gfx: str | Device,
    grid: tuple[int, int, int] = (1, 1, 1),
    block: tuple[int, int, int] = (256, 1, 1),
    *,
    trip_count: int = 1,
    trip_count_known: bool = False,
    counts: IsaCounts | None = None,
    scale: float | None = None,
    weights: list[float] | None = None,
) -> Prediction:
    dev = device(gfx) if isinstance(gfx, str) else gfx
    wf = meta.wavefront_size or dev.wavefront_size
    have_isa = not _isa_is_empty(counts)
    if have_isa and counts is not None:
        work = counts_to_work(counts, wf)
    else:
        work = KernelWork(name=meta.name or "kernel", wavefront_size=wf)
    notes = apply_metadata(work, meta, have_isa=have_isa)
    waves = wave_count(grid, block, work.wavefront_size)
    block_threads = block[0] * block[1] * block[2]
    cap = None
    if meta.global_buffer_bytes:
        cap = float(sum(meta.global_buffer_bytes))
        notes.append("Global traffic is capped by the buffer sizes seen at dispatch.")
    return predict_work(
        work,
        dev,
        waves,
        trip_count=trip_count,
        trip_count_known=trip_count_known,
        scale=scale,
        weights=weights,
        lds_bytes_per_workgroup=meta.group_segment_fixed_size
        + meta.dynamic_shared_bytes,
        workgroup_size=block_threads,
        global_byte_cap=cap,
        metadata_only=not have_isa,
        extra_notes=notes,
    )


def predict_text(
    text: str,
    gfx: str | Device,
    grid: tuple[int, int, int] = (1, 1, 1),
    block: tuple[int, int, int] = (256, 1, 1),
    *,
    trip_count: int = 1,
    trip_count_known: bool = False,
    scale: float | None = None,
    weights: list[float] | None = None,
) -> list[Prediction]:
    """Predict every kernel in an assembly listing, LLVM IR, or metadata block."""

    dev = device(gfx) if isinstance(gfx, str) else gfx
    metas = parse_amdgpu_metadata(text)
    isas = parse_isa(text)
    paired = _pair(isas, metas)
    if not paired:
        raise ValueError("no kernel ISA or COMGR metadata found")
    predictions = []
    for counts, meta in paired:
        if meta is None:
            wf = dev.wavefront_size
            assert counts is not None
            work = counts_to_work(counts, wf)
            waves = wave_count(grid, block, wf)
            predictions.append(
                predict_work(
                    work,
                    dev,
                    waves,
                    trip_count=trip_count,
                    trip_count_known=trip_count_known,
                    scale=scale,
                    weights=weights,
                    workgroup_size=block[0] * block[1] * block[2],
                    extra_notes=[
                        "No COMGR metadata. VGPR/SGPR occupancy uses a default "
                        "unless the ISA listing carried .vgpr_count."
                    ],
                )
            )
            continue
        predictions.append(
            predict_metadata(
                meta,
                dev,
                grid,
                block,
                trip_count=trip_count,
                trip_count_known=trip_count_known,
                counts=counts,
                scale=scale,
                weights=weights,
            )
        )
    return predictions
