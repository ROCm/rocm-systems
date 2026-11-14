// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust ABCE frame mechanisms.

// These entry points project caller-owned C objects through raw pointers.
// Initialization/reconfiguration requires exclusive caller access; submission
// accesses individual fields and shared atomic cursors, never whole-object
// exclusive references. Pointer and buffer preconditions are specified by the
// corresponding public C header. The packet/ring tests run both providers.

// The C-facing projection keeps C integer conversion and alignment contracts.
// Safety requirements and callback preconditions are specified in the public
// header named by this module; bodies intentionally follow those algorithms.
#![allow(
    clippy::missing_safety_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_ptr_alignment,
    clippy::too_many_lines
)]

use super::abi::{
    ABCE_LINEAR_BATCH_MODE_AUTOMATIC, ABCE_MULTICAST_MODE_AUTOMATIC, ABCE_OP_KIND_BROADCAST,
    ABCE_OP_KIND_COPY_RECT, ABCE_OP_KIND_FILL, ABCE_OP_KIND_INDIRECT, ABCE_OP_KIND_LINEAR,
    ABCE_OP_KIND_MULTICAST, ABCE_OP_KIND_SWAP, SDMA_PKT_ATOMIC, SDMA_PKT_CONSTANT_FILL,
    SDMA_PKT_COPY_LINEAR, SDMA_PKT_COPY_LINEAR_BROADCAST, SDMA_PKT_COPY_LINEAR_RECT,
    SDMA_PKT_COPY_LINEAR_SWAP, SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX125PLUS, SDMA_PKT_FENCE,
    SDMA_PKT_FENCE_64B_GFX125PLUS, SDMA_PKT_GCR, SDMA_PKT_GCR_GFX125PLUS, SDMA_PKT_HDP_FLUSH,
    SDMA_PKT_POLL_MEM_64B_GFX125PLUS, SDMA_PKT_POLL_REGMEM, SDMA_PKT_TIMESTAMP, SDMA_PKT_TRAP,
    abce_body_sync_t, abce_builder_t, abce_coherency_control_t, abce_copy_metadata_t,
    abce_dep_signal_t, abce_engine_op_t, abce_frame_composer_t, abce_frame_job_t,
    abce_frame_rect_cursor_t, abce_platform_caps_t, abce_signal_ref_t, abce_timestamp_surface_t,
    abce_wait_signal_params_t,
};
use super::{
    abce_builder_build_atomic_add, abce_builder_build_atomic_decrement,
    abce_builder_build_broadcast_copy, abce_builder_build_copy, abce_builder_build_copy_rect,
    abce_builder_build_fence, abce_builder_build_fence_64b, abce_builder_build_fill,
    abce_builder_build_gcr, abce_builder_build_get_global_timestamp, abce_builder_build_hdp_flush,
    abce_builder_build_multicast_copy, abce_builder_build_multicast_wait_signal_copy,
    abce_builder_build_poll, abce_builder_build_poll_64b, abce_builder_build_swap_copy,
    abce_builder_build_trap, abce_builder_build_wait_signal_copy,
    abce_builder_build_wait_signal_indirect_copy, abce_builder_build_wait_signal_swap,
    abce_builder_is_gfx125plus, abce_builder_num_broadcast_packets, abce_builder_num_copy_packets,
    abce_builder_num_fill_packets, abce_builder_num_multicast_packets,
    abce_builder_num_rect_packets, abce_builder_num_swap_packets,
    abce_builder_num_wait_signal_swap_packets, abce_builder_requires_gcr, abce_status_is_ok,
    abce_wait_signal_none,
};

pub unsafe extern "C" fn abce_signal_ref_make(
    value: *mut core::ffi::c_void,
    completion_value: u64,
    event_mailbox: *mut core::ffi::c_void,
    event_id: u32,
    coordination_scratch: *mut core::ffi::c_void,
) -> abce_signal_ref_t {
    {
        let mut r#ref: abce_signal_ref_t = unsafe { core::mem::zeroed() };
        (r#ref).value = value;
        (r#ref).completion_value = completion_value;
        (r#ref).sole_update = false;
        (r#ref).event_mailbox = event_mailbox;
        (r#ref).event_id = event_id;
        (r#ref).coordination_scratch = coordination_scratch;
        (r#ref).host_store_readback = false;
        r#ref
    }
}

#[must_use]
pub unsafe extern "C" fn abce_timestamp_surface_enabled(
    surface: *const abce_timestamp_surface_t,
) -> bool {
    unsafe { !((*(surface)).start_value).is_null() && !((*(surface)).end_value).is_null() }
}

pub unsafe extern "C" fn abce_dep_signal_make(
    value: *mut core::ffi::c_void,
    observed: u64,
    reference: u64,
) -> abce_dep_signal_t {
    {
        let mut dep: abce_dep_signal_t = unsafe { core::mem::zeroed() };
        (dep).value = value;
        (dep).observed_value = observed;
        (dep).reference_value = reference;
        dep
    }
}

#[must_use]
pub unsafe extern "C" fn abce_coherency_control_default() -> abce_coherency_control_t {
    unsafe {
        let mut coherency: abce_coherency_control_t = { core::mem::zeroed() };
        (coherency).value = 0_u32;
        {
            let word = core::ptr::addr_of_mut!(((coherency).flags).word);
            *word = (*word & !(0x1u32 << 0)) | 0x1_u32;
        };
        {
            let word = core::ptr::addr_of_mut!(((coherency).flags).word);
            *word = (*word & !(0x1u32 << 1)) | (0x1_u32 << 1);
        };
        coherency
    }
}

pub unsafe extern "C" fn abce_copy_metadata_initialize(out_metadata: *mut abce_copy_metadata_t) {
    unsafe {
        let _ = libc::memset(
            (out_metadata).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<abce_copy_metadata_t>() as usize,
        );
        (*(out_metadata)).coherency = abce_coherency_control_default();
        (*(out_metadata)).linear_batch_mode = ABCE_LINEAR_BATCH_MODE_AUTOMATIC;
        (*(out_metadata)).multicast_mode = ABCE_MULTICAST_MODE_AUTOMATIC;
        (*(out_metadata)).prefer_fused = true;
    }
}

pub unsafe extern "C" fn abce_frame_composer_initialize(
    builder: *const abce_builder_t,
    caps: abce_platform_caps_t,
    out_composer: *mut abce_frame_composer_t,
) {
    unsafe {
        (*(out_composer)).builder = builder;
        (*(out_composer)).caps = caps;
        (*(out_composer)).is_gfx125plus = abce_builder_is_gfx125plus(builder);
    }
}

pub(super) unsafe extern "C" fn abce_frame_should_emit_gcr(
    composer: *const abce_frame_composer_t,
) -> bool {
    unsafe {
        abce_builder_requires_gcr((*(composer)).builder)
            && !(((*(composer)).caps).driver_manages_gcr)
    }
}

pub(super) unsafe extern "C" fn abce_frame_gcr_bytes(
    composer: *const abce_frame_composer_t,
) -> u32 {
    unsafe {
        if !(abce_frame_should_emit_gcr(composer)) {
            return 0_u32;
        }
        if (*(composer)).is_gfx125plus {
            (core::mem::size_of::<SDMA_PKT_GCR_GFX125PLUS>() as usize) as u32
        } else {
            (core::mem::size_of::<SDMA_PKT_GCR>() as usize) as u32
        }
    }
}

pub(super) unsafe extern "C" fn abce_frame_folded_dep(
    composer: *const abce_frame_composer_t,
    job: *const abce_frame_job_t,
) -> u32 {
    unsafe {
        let metadata: *const abce_copy_metadata_t = (*(job)).metadata;
        if !((*(composer)).is_gfx125plus) || (*(job)).multi {
            return 4_294_967_295_u32;
        }
        if (abce_timestamp_surface_enabled(core::ptr::addr_of!((*(metadata)).timestamps))
            || (((*(composer)).caps).emit_hdp_flush
                && ((((*(metadata)).coherency).flags).word & 0x1u32) != 0))
            || ((((((*(metadata)).coherency).flags).word >> 1) & 0x1u32) != 0
                && abce_frame_should_emit_gcr(composer))
        {
            return 4_294_967_295_u32;
        }
        if !((*((*(job)).ops).wrapping_offset((0i32) as isize)).fused) {
            return 4_294_967_295_u32;
        }
        {
            let mut dep_idx: u32 = 0_u32;
            let mut first_11 = true;
            loop {
                if !first_11 {
                    dep_idx = (dep_idx).wrapping_add(1u32);
                }
                first_11 = false;
                if (dep_idx) >= ((*(metadata)).num_deps) {
                    break;
                }
                {
                    if ((*((*(metadata)).deps).wrapping_offset((dep_idx) as isize)).observed_value)
                        != ((*((*(metadata)).deps).wrapping_offset((dep_idx) as isize))
                            .reference_value)
                    {
                        return dep_idx;
                    }
                }
            }
        }
        4_294_967_295_u32
    }
}

pub(super) unsafe extern "C" fn abce_body_sync_for(
    composer: *const abce_frame_composer_t,
    job: *const abce_frame_job_t,
    body_idx: u32,
) -> abce_body_sync_t {
    unsafe {
        let wait_for_start: bool =
            ((*(job)).multi && (*(job)).start_gate_required) && !((*(job)).coordinator);
        let signals_completion_directly: bool = !((*(job)).multi) && !((*(job)).epilogue_required);
        let contributes_to_fan_in: bool =
            (*(job)).multi && (!((*(job)).coordinator) || !((*(job)).start_gate_required));
        let mut sync: abce_body_sync_t = { core::mem::zeroed() };
        (sync).wait_addr = core::ptr::null_mut();
        (sync).wait_reference = 0_u64;
        (sync).wait_mask = (1_u64).wrapping_shl((62_u64) as u32);
        (sync).signal_addr = core::ptr::null_mut();
        (sync).wait = wait_for_start && ((body_idx) == (0_u32));
        (sync).inline_signal = (*(job)).frame_all_fused;
        (sync).signal = (signals_completion_directly || contributes_to_fan_in)
            && ((sync).inline_signal
                || ((body_idx) == (((*(job)).frame_num_ops).wrapping_sub(1_u32))));
        if (sync).wait {
            (sync).wait_addr = (*(job)).coordination_word;
        }
        if (sync).signal {
            (sync).signal_addr = (*(job)).coordination_word;
        }
        if (body_idx) == (0_u32) {
            let folded_dep: u32 = abce_frame_folded_dep(composer, job);
            if (folded_dep) != (4_294_967_295_u32) {
                let dep: *const abce_dep_signal_t = core::ptr::addr_of!(
                    (*((*((*(job)).metadata)).deps).wrapping_offset((folded_dep) as isize))
                );
                (sync).wait = true;
                (sync).wait_addr = (*(dep)).value;
                (sync).wait_reference = (*(dep)).reference_value;
                (sync).wait_mask = u64::MAX;
            }
        }
        sync
    }
}

pub(super) unsafe extern "C" fn abce_frame_poll_bytes(
    composer: *const abce_frame_composer_t,
) -> u32 {
    unsafe {
        if (*(composer)).is_gfx125plus {
            (core::mem::size_of::<SDMA_PKT_POLL_MEM_64B_GFX125PLUS>() as usize) as u32
        } else {
            (core::mem::size_of::<SDMA_PKT_POLL_REGMEM>() as usize) as u32
        }
    }
}

pub(super) unsafe extern "C" fn abce_frame_multicast_pkt_bytes(num_dsts: u32) -> u32 {
    {
        ((5u32).wrapping_add((2u32).wrapping_mul(num_dsts)))
            .wrapping_mul((core::mem::size_of::<u32>() as usize) as u32)
    }
}

pub(super) unsafe extern "C" fn abce_engine_op_size2(op: *const abce_engine_op_t) -> usize {
    unsafe {
        if ((*(op)).size2) != 0 {
            (*(op)).size2
        } else {
            (*(op)).size
        }
    }
}

pub(super) unsafe extern "C" fn abce_frame_dep_waits_bytes(
    composer: *const abce_frame_composer_t,
    deps: *const abce_dep_signal_t,
    num_deps: u32,
    folded_dep: u32,
) -> usize {
    unsafe {
        let mut bytes: usize = 0_usize;
        {
            let mut dep_idx: u32 = 0_u32;
            let mut first_12 = true;
            loop {
                if !first_12 {
                    dep_idx = (dep_idx).wrapping_add(1u32);
                }
                first_12 = false;
                if (dep_idx) >= (num_deps) {
                    break;
                }
                {
                    if ((dep_idx) == (folded_dep))
                        || (((*(deps).wrapping_offset((dep_idx) as isize)).observed_value)
                            == ((*(deps).wrapping_offset((dep_idx) as isize)).reference_value))
                    {
                        continue;
                    }
                    bytes = (bytes).wrapping_add(
                        (if (*(composer)).is_gfx125plus {
                            (core::mem::size_of::<SDMA_PKT_POLL_MEM_64B_GFX125PLUS>() as usize)
                                as u32
                        } else {
                            (core::mem::size_of::<SDMA_PKT_POLL_REGMEM>() as usize) as u32
                        }) as usize,
                    );
                    if !((*(composer)).is_gfx125plus)
                        && ((((*(deps).wrapping_offset((dep_idx) as isize)).observed_value)
                            .wrapping_shr((32_u64) as u32))
                            != (((*(deps).wrapping_offset((dep_idx) as isize)).reference_value)
                                .wrapping_shr((32_u64) as u32)))
                    {
                        bytes = (bytes).wrapping_add(
                            ((core::mem::size_of::<SDMA_PKT_POLL_REGMEM>() as usize) as u32)
                                as usize,
                        );
                    }
                }
            }
        }
        bytes
    }
}

pub(super) unsafe extern "C" fn abce_frame_payload_bytes(
    composer: *const abce_frame_composer_t,
    op: *const abce_engine_op_t,
) -> usize {
    unsafe {
        let builder: *const abce_builder_t = (*(composer)).builder;
        {
            match (*(op)).kind {
                value if value == ABCE_OP_KIND_LINEAR => {
                    return ((abce_builder_num_copy_packets(builder, (*(op)).size)) as usize)
                        .wrapping_mul(
                            ((core::mem::size_of::<SDMA_PKT_COPY_LINEAR>() as usize) as u32)
                                as usize,
                        );
                }
                value if value == ABCE_OP_KIND_MULTICAST => {
                    return ((abce_builder_num_multicast_packets(builder, (*(op)).size)) as usize)
                        .wrapping_mul((abce_frame_multicast_pkt_bytes((*(op)).num_dsts)) as usize);
                }
                value if value == ABCE_OP_KIND_BROADCAST => {
                    return (((abce_builder_num_broadcast_packets(builder, (*(op)).size)) as usize)
                        .wrapping_mul(
                            (core::mem::size_of::<SDMA_PKT_COPY_LINEAR_BROADCAST>() as usize)
                                as usize,
                        )) as usize;
                }
                value if value == ABCE_OP_KIND_SWAP => {
                    return (((abce_builder_num_swap_packets(builder, (*(op)).size)) as usize)
                        .wrapping_mul(
                            (core::mem::size_of::<SDMA_PKT_COPY_LINEAR_SWAP>() as usize) as usize,
                        )) as usize;
                }
                value if value == ABCE_OP_KIND_COPY_RECT => {
                    let mut num_packets: u32 = 0_u32;
                    if !(abce_status_is_ok(abce_builder_num_rect_packets(
                        builder,
                        core::ptr::addr_of!((*((*(op)).rect)).dst),
                        core::ptr::addr_of!((*((*(op)).rect)).dst_offset),
                        core::ptr::addr_of!((*((*(op)).rect)).src),
                        core::ptr::addr_of!((*((*(op)).rect)).src_offset),
                        core::ptr::addr_of!((*((*(op)).rect)).range),
                        core::ptr::addr_of_mut!(num_packets),
                    ))) {
                        return 0_usize;
                    }
                    return (((num_packets) as usize).wrapping_mul(
                        (core::mem::size_of::<SDMA_PKT_COPY_LINEAR_RECT>() as usize) as usize,
                    )) as usize;
                }
                value if value == ABCE_OP_KIND_INDIRECT => {
                    return core::mem::size_of::<SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX125PLUS>(
                    ) as usize;
                }
                value if value == ABCE_OP_KIND_FILL => {
                    return ((abce_builder_num_fill_packets(
                        builder,
                        (((*(op)).size) / (core::mem::size_of::<u32>() as usize) as usize) as usize,
                    )) as usize)
                        .wrapping_mul(
                            ((core::mem::size_of::<SDMA_PKT_CONSTANT_FILL>() as usize) as u32)
                                as usize,
                        );
                }
                _ => {}
            }
        }
        0_usize
    }
}

pub(super) unsafe extern "C" fn abce_frame_fused_packets(
    composer: *const abce_frame_composer_t,
    op: *const abce_engine_op_t,
) -> u64 {
    unsafe {
        let builder: *const abce_builder_t = (*(composer)).builder;
        {
            match (*(op)).kind {
                value if value == ABCE_OP_KIND_LINEAR => {
                    abce_builder_num_copy_packets(builder, (*(op)).size)
                }
                value if value == ABCE_OP_KIND_SWAP => abce_builder_num_wait_signal_swap_packets(
                    builder,
                    (*(op)).size,
                    abce_engine_op_size2(op),
                ),
                value if value == ABCE_OP_KIND_MULTICAST => {
                    abce_builder_num_multicast_packets(builder, (*(op)).size)
                }
                _ => 1_u64,
            }
        }
    }
}

pub(super) unsafe extern "C" fn abce_frame_fused_bytes(
    composer: *const abce_frame_composer_t,
    op: *const abce_engine_op_t,
    wait: bool,
    signal: bool,
) -> usize {
    unsafe {
        let num_packets: u64 = abce_frame_fused_packets(composer, op);
        let wait_dwords: u64 = u64::from(if wait { 7u32 } else { 0u32 });
        let signal_dwords: u64 = if signal {
            u64::from(5u32).wrapping_mul(num_packets)
        } else {
            u64::from(0u32)
        };

        let body_dwords: u64 = if ((((*(op)).kind) == ABCE_OP_KIND_INDIRECT)
            || (((*(op)).kind) == ABCE_OP_KIND_LINEAR))
            || (((*(op)).kind) == ABCE_OP_KIND_SWAP)
        {
            u64::from(6u32)
        } else if ((*(op)).kind) == ABCE_OP_KIND_MULTICAST {
            u64::from(4u32).wrapping_add(u64::from(2u32).wrapping_mul(u64::from((*(op)).num_dsts)))
        } else {
            return 0_usize;
        };
        ((((((num_packets).wrapping_mul(u64::from(1u32).wrapping_add(body_dwords)))
            .wrapping_add(wait_dwords))
        .wrapping_add(signal_dwords)) as usize)
            .wrapping_mul((core::mem::size_of::<u32>() as usize) as usize)) as usize
    }
}

pub(super) unsafe extern "C" fn abce_frame_body_bytes(
    composer: *const abce_frame_composer_t,
    op: *const abce_engine_op_t,
    sync: *const abce_body_sync_t,
) -> usize {
    unsafe {
        let standalone_signal: bool = (*(sync)).signal && !((*(sync)).inline_signal);
        if (*(op)).fused {
            return (abce_frame_fused_bytes(
                composer,
                op,
                (*(sync)).wait,
                (*(sync)).signal && (*(sync)).inline_signal,
            ))
            .wrapping_add(
                (if standalone_signal {
                    (core::mem::size_of::<SDMA_PKT_ATOMIC>() as usize) as u32
                } else {
                    0_u32
                }) as usize,
            );
        }
        let mut bytes: usize = 0_usize;
        if (*(sync)).wait {
            bytes = (bytes).wrapping_add((abce_frame_poll_bytes(composer)) as usize);
        }
        bytes = (bytes).wrapping_add(abce_frame_payload_bytes(composer, op));
        if (*(sync)).signal {
            bytes = (bytes)
                .wrapping_add(((core::mem::size_of::<SDMA_PKT_ATOMIC>() as usize) as u32) as usize);
        }
        bytes
    }
}

pub(super) unsafe extern "C" fn abce_frame_prologue_bytes(
    composer: *const abce_frame_composer_t,
    metadata: *const abce_copy_metadata_t,
    multi: bool,
    start_gate_required: bool,
    folded_dep: u32,
) -> usize {
    unsafe {
        let mut bytes: usize = abce_frame_dep_waits_bytes(
            composer,
            (*(metadata)).deps,
            (*(metadata)).num_deps,
            folded_dep,
        );
        if abce_timestamp_surface_enabled(core::ptr::addr_of!((*(metadata)).timestamps)) {
            bytes = (bytes).wrapping_add(
                ((core::mem::size_of::<SDMA_PKT_TIMESTAMP>() as usize) as u32) as usize,
            );
        }
        if ((*(composer)).caps).emit_hdp_flush
            && ((((*(metadata)).coherency).flags).word & 0x1u32) != 0
        {
            bytes = (bytes).wrapping_add(
                ((core::mem::size_of::<SDMA_PKT_HDP_FLUSH>() as usize) as u32) as usize,
            );
        }
        if (((((*(metadata)).coherency).flags).word >> 1) & 0x1u32) != 0 {
            bytes = (bytes).wrapping_add((abce_frame_gcr_bytes(composer)) as usize);
        }
        if multi && start_gate_required {
            bytes = (bytes)
                .wrapping_add(((core::mem::size_of::<SDMA_PKT_ATOMIC>() as usize) as u32) as usize);
        }
        bytes
    }
}

pub(super) unsafe extern "C" fn abce_frame_completion_decrements(
    composer: *const abce_frame_composer_t,
    target: *const abce_signal_ref_t,
) -> bool {
    unsafe { ((*(composer)).caps).device_atomic_support && !((*(target)).sole_update) }
}

pub(super) unsafe extern "C" fn abce_frame_completion_bytes(
    composer: *const abce_frame_composer_t,
    target: *const abce_signal_ref_t,
) -> u32 {
    unsafe {
        if abce_frame_completion_decrements(composer, target) {
            return (core::mem::size_of::<SDMA_PKT_ATOMIC>() as usize) as u32;
        }
        if (*(composer)).is_gfx125plus {
            return (core::mem::size_of::<SDMA_PKT_FENCE_64B_GFX125PLUS>() as usize) as u32;
        }
        if ((*(target)).completion_value) > u64::from(4_294_967_295_u32) {
            (2_u32).wrapping_mul((core::mem::size_of::<SDMA_PKT_FENCE>() as usize) as u32)
        } else {
            (core::mem::size_of::<SDMA_PKT_FENCE>() as usize) as u32
        }
    }
}

pub(super) unsafe extern "C" fn abce_frame_epilogue_bytes(
    composer: *const abce_frame_composer_t,
    metadata: *const abce_copy_metadata_t,
    multi: bool,
) -> usize {
    unsafe {
        let mut bytes: usize = 0_usize;
        if multi {
            bytes = (bytes).wrapping_add((abce_frame_poll_bytes(composer)) as usize);
        }
        if (((((*(metadata)).coherency).flags).word >> 1) & 0x1u32) != 0 {
            bytes = (bytes).wrapping_add((abce_frame_gcr_bytes(composer)) as usize);
        }
        if abce_timestamp_surface_enabled(core::ptr::addr_of!((*(metadata)).timestamps)) {
            bytes = (bytes).wrapping_add(
                ((core::mem::size_of::<SDMA_PKT_TIMESTAMP>() as usize) as u32) as usize,
            );
        }
        bytes = (bytes).wrapping_add(
            (abce_frame_completion_bytes(composer, core::ptr::addr_of!((*(metadata)).out)))
                as usize,
        );
        if !(((*(metadata)).out).event_mailbox).is_null() {
            bytes = (bytes).wrapping_add(
                (((core::mem::size_of::<SDMA_PKT_FENCE>() as usize) as u32)
                    .wrapping_add((core::mem::size_of::<SDMA_PKT_TRAP>() as usize) as u32))
                    as usize,
            );
        }
        bytes
    }
}

#[must_use]
pub unsafe extern "C" fn abce_frame_composer_frame_bytes(
    composer: *const abce_frame_composer_t,
    job: *const abce_frame_job_t,
) -> usize {
    unsafe {
        let metadata: *const abce_copy_metadata_t = (*(job)).metadata;
        let mut bytes: usize = 0_usize;
        if (*(job)).coordinator {
            bytes = (bytes).wrapping_add(abce_frame_prologue_bytes(
                composer,
                metadata,
                (*(job)).multi,
                (*(job)).start_gate_required,
                abce_frame_folded_dep(composer, job),
            ));
        }
        let mut body_idx: u32 = 0_u32;
        {
            let mut op_idx: u32 = 0_u32;
            let mut first_13 = true;
            loop {
                if !first_13 {
                    op_idx = (op_idx).wrapping_add(1u32);
                }
                first_13 = false;
                if (op_idx) >= ((*(job)).num_ops) {
                    break;
                }
                {
                    if ((*((*(job)).ops).wrapping_offset((op_idx) as isize)).engine)
                        != ((*(job)).engine)
                    {
                        continue;
                    }
                    let sync: abce_body_sync_t = abce_body_sync_for(composer, job, body_idx);
                    bytes = (bytes).wrapping_add(abce_frame_body_bytes(
                        composer,
                        core::ptr::addr_of!((*((*(job)).ops).wrapping_offset((op_idx) as isize))),
                        core::ptr::addr_of!(sync),
                    ));
                    body_idx = (body_idx).wrapping_add(1u32);
                }
            }
        }
        if (*(job)).coordinator && (*(job)).epilogue_required {
            bytes = (bytes).wrapping_add(abce_frame_epilogue_bytes(
                composer,
                metadata,
                (*(job)).multi,
            ));
        }
        bytes
    }
}

#[must_use]
pub unsafe extern "C" fn abce_frame_composer_signal_count(
    composer: *const abce_frame_composer_t,
    job: *const abce_frame_job_t,
) -> u64 {
    unsafe {
        let mut num_signals: u64 = 0_u64;
        let mut body_idx: u32 = 0_u32;
        {
            let mut op_idx: u32 = 0_u32;
            let mut first_14 = true;
            loop {
                if !first_14 {
                    op_idx = (op_idx).wrapping_add(1u32);
                }
                first_14 = false;
                if (op_idx) >= ((*(job)).num_ops) {
                    break;
                }
                {
                    if ((*((*(job)).ops).wrapping_offset((op_idx) as isize)).engine)
                        != ((*(job)).engine)
                    {
                        continue;
                    }
                    let sync: abce_body_sync_t = abce_body_sync_for(composer, job, body_idx);
                    if (sync).signal {
                        num_signals = (num_signals).wrapping_add(if (sync).inline_signal {
                            abce_frame_fused_packets(
                                composer,
                                core::ptr::addr_of!(
                                    (*((*(job)).ops).wrapping_offset((op_idx) as isize))
                                ),
                            )
                        } else {
                            1_u64
                        });
                    }
                    body_idx = (body_idx).wrapping_add(1u32);
                }
            }
        }
        num_signals
    }
}

pub(super) unsafe extern "C" fn abce_frame_emit_dep_waits(
    composer: *const abce_frame_composer_t,
    out: *mut core::ffi::c_char,
    deps: *const abce_dep_signal_t,
    num_deps: u32,
    folded_dep: u32,
) -> usize {
    unsafe {
        let mut cursor: *mut core::ffi::c_char = out;
        {
            let mut dep_idx: u32 = 0_u32;
            let mut first_15 = true;
            loop {
                if !first_15 {
                    dep_idx = (dep_idx).wrapping_add(1u32);
                }
                first_15 = false;
                if (dep_idx) >= (num_deps) {
                    break;
                }
                {
                    if ((dep_idx) == (folded_dep))
                        || (((*(deps).wrapping_offset((dep_idx) as isize)).observed_value)
                            == ((*(deps).wrapping_offset((dep_idx) as isize)).reference_value))
                    {
                        continue;
                    }
                    if (*(composer)).is_gfx125plus {
                        abce_builder_build_poll_64b(
                            (*(composer)).builder,
                            cursor,
                            (*(deps).wrapping_offset((dep_idx) as isize)).value,
                            (*(deps).wrapping_offset((dep_idx) as isize)).reference_value,
                            u64::MAX,
                        );
                        cursor = (cursor).wrapping_offset(
                            ((core::mem::size_of::<SDMA_PKT_POLL_MEM_64B_GFX125PLUS>() as usize)
                                as u32) as isize,
                        );
                    } else {
                        let words: *mut u32 =
                            ((*(deps).wrapping_offset((dep_idx) as isize)).value).cast::<u32>();
                        if (((*(deps).wrapping_offset((dep_idx) as isize)).observed_value)
                            .wrapping_shr((32_u64) as u32))
                            != (((*(deps).wrapping_offset((dep_idx) as isize)).reference_value)
                                .wrapping_shr((32_u64) as u32))
                        {
                            abce_builder_build_poll(
                                (*(composer)).builder,
                                cursor,
                                core::ptr::addr_of_mut!(
                                    (*(words).wrapping_offset((1i32) as isize))
                                )
                                .cast::<core::ffi::c_void>(),
                                (((*(deps).wrapping_offset((dep_idx) as isize)).reference_value)
                                    .wrapping_shr((32_u64) as u32))
                                    as u32,
                                4_294_967_295_u32,
                            );
                            cursor = (cursor).wrapping_offset(
                                ((core::mem::size_of::<SDMA_PKT_POLL_REGMEM>() as usize) as u32)
                                    as isize,
                            );
                        }
                        abce_builder_build_poll(
                            (*(composer)).builder,
                            cursor,
                            core::ptr::addr_of_mut!((*(words).wrapping_offset((0i32) as isize)))
                                .cast::<core::ffi::c_void>(),
                            ((*(deps).wrapping_offset((dep_idx) as isize)).reference_value) as u32,
                            4_294_967_295_u32,
                        );
                        cursor = (cursor).wrapping_offset(
                            ((core::mem::size_of::<SDMA_PKT_POLL_REGMEM>() as usize) as u32)
                                as isize,
                        );
                    }
                }
            }
        }
        (((cursor) as isize).wrapping_sub((out) as isize)
            / core::mem::size_of::<core::ffi::c_char>() as isize) as usize
    }
}

pub(super) unsafe extern "C" fn abce_frame_emit_prologue(
    composer: *const abce_frame_composer_t,
    out: *mut core::ffi::c_char,
    metadata: *const abce_copy_metadata_t,
    multi: bool,
    start_gate_required: bool,
    coordination_word: *mut core::ffi::c_void,
    folded_dep: u32,
) -> usize {
    unsafe {
        let mut cursor: *mut core::ffi::c_char = out;
        cursor = (cursor).wrapping_add(abce_frame_emit_dep_waits(
            composer,
            cursor,
            (*(metadata)).deps,
            (*(metadata)).num_deps,
            folded_dep,
        ));
        if abce_timestamp_surface_enabled(core::ptr::addr_of!((*(metadata)).timestamps)) {
            abce_builder_build_get_global_timestamp(
                (*(composer)).builder,
                cursor,
                ((*(metadata)).timestamps).start_value,
            );
            cursor = (cursor).wrapping_offset(
                ((core::mem::size_of::<SDMA_PKT_TIMESTAMP>() as usize) as u32) as isize,
            );
        }
        if ((*(composer)).caps).emit_hdp_flush
            && ((((*(metadata)).coherency).flags).word & 0x1u32) != 0
        {
            abce_builder_build_hdp_flush((*(composer)).builder, cursor);
            cursor = (cursor).wrapping_offset(
                ((core::mem::size_of::<SDMA_PKT_HDP_FLUSH>() as usize) as u32) as isize,
            );
        }
        if (((((*(metadata)).coherency).flags).word >> 1) & 0x1u32) != 0
            && abce_frame_should_emit_gcr(composer)
        {
            abce_builder_build_gcr((*(composer)).builder, cursor, true);
            cursor = (cursor).wrapping_offset((abce_frame_gcr_bytes(composer)) as isize);
        }
        if multi && start_gate_required {
            abce_builder_build_atomic_add(
                (*(composer)).builder,
                cursor,
                coordination_word,
                (0_u64).wrapping_sub((1_u64).wrapping_shl((62_u64) as u32)),
            );
            cursor = (cursor).wrapping_offset(
                ((core::mem::size_of::<SDMA_PKT_ATOMIC>() as usize) as u32) as isize,
            );
        }
        (((cursor) as isize).wrapping_sub((out) as isize)
            / core::mem::size_of::<core::ffi::c_char>() as isize) as usize
    }
}

pub(super) unsafe extern "C" fn abce_frame_write_start_gate_poll(
    composer: *const abce_frame_composer_t,
    out: *mut core::ffi::c_char,
    coordination_word: *mut core::ffi::c_void,
) -> usize {
    unsafe {
        if (*(composer)).is_gfx125plus {
            abce_builder_build_poll_64b(
                (*(composer)).builder,
                out,
                coordination_word,
                0_u64,
                (1_u64).wrapping_shl((62_u64) as u32),
            );
            return ((core::mem::size_of::<SDMA_PKT_POLL_MEM_64B_GFX125PLUS>() as usize) as u32)
                as usize;
        }
        let coordination_words: *mut u32 = (coordination_word).cast::<u32>();
        let start_gate_high_word: u32 = (1_u32).wrapping_shl(30_u32);
        abce_builder_build_poll(
            (*(composer)).builder,
            out,
            core::ptr::addr_of_mut!((*(coordination_words).wrapping_offset((1i32) as isize)))
                .cast::<core::ffi::c_void>(),
            0_u32,
            start_gate_high_word,
        );
        ((core::mem::size_of::<SDMA_PKT_POLL_REGMEM>() as usize) as u32) as usize
    }
}

pub(super) unsafe extern "C" fn abce_frame_rect_append(
    user_data: *mut core::ffi::c_void,
    packet_bytes: usize,
) -> *mut core::ffi::c_void {
    unsafe {
        let state: *mut abce_frame_rect_cursor_t = (user_data).cast::<abce_frame_rect_cursor_t>();
        let slot: *mut core::ffi::c_char = (*(state)).cursor;
        (*(state)).cursor = ((*(state)).cursor).wrapping_add(packet_bytes);
        (slot).cast::<core::ffi::c_void>()
    }
}

pub(super) unsafe extern "C" fn abce_frame_emit_body(
    composer: *const abce_frame_composer_t,
    out: *mut core::ffi::c_char,
    op: *const abce_engine_op_t,
    sync: *const abce_body_sync_t,
) -> usize {
    unsafe {
        let mut cursor: *mut core::ffi::c_char = out;
        let builder: *const abce_builder_t = (*(composer)).builder;
        if (*(op)).fused {
            let inline_signal: bool = (*(sync)).signal && (*(sync)).inline_signal;
            let mut params: abce_wait_signal_params_t = abce_wait_signal_none();
            (params).wait_addr = (*(sync)).wait_addr;
            (params).signal_addr = if inline_signal {
                (*(sync)).signal_addr
            } else {
                core::ptr::null_mut()
            };
            (params).wait_reference = (*(sync)).wait_reference;
            (params).wait_mask = (*(sync)).wait_mask;
            (params).wait_first_chunk_only = true;
            if ((*(op)).kind) == ABCE_OP_KIND_INDIRECT {
                abce_builder_build_wait_signal_indirect_copy(
                    builder,
                    cursor,
                    (*(op)).dst,
                    (*(op)).src,
                    (*(op)).size,
                    (*(op)).indirect_src,
                    (*(op)).indirect_dst,
                    core::ptr::addr_of_mut!(params).cast_const(),
                );
            } else if ((*(op)).kind) == ABCE_OP_KIND_LINEAR {
                abce_builder_build_wait_signal_copy(
                    builder,
                    cursor,
                    (*(op)).dst,
                    (*(op)).src,
                    (*(op)).size,
                    core::ptr::addr_of_mut!(params).cast_const(),
                );
            } else if ((*(op)).kind) == ABCE_OP_KIND_SWAP {
                abce_builder_build_wait_signal_swap(
                    builder,
                    cursor,
                    (*(op)).dst,
                    (((*(op)).src) as usize) as *mut core::ffi::c_void,
                    (*(op)).size,
                    abce_engine_op_size2(op),
                    core::ptr::addr_of_mut!(params).cast_const(),
                );
            } else if ((*(op)).kind) == ABCE_OP_KIND_MULTICAST {
                abce_builder_build_multicast_wait_signal_copy(
                    builder,
                    cursor,
                    (*(op)).dsts,
                    (*(op)).num_dsts,
                    (*(op)).src,
                    (*(op)).size,
                    core::ptr::addr_of_mut!(params).cast_const(),
                );
            }
            cursor = (cursor).wrapping_add(abce_frame_fused_bytes(
                composer,
                op,
                (*(sync)).wait,
                inline_signal,
            ));
            if (*(sync)).signal && !(inline_signal) {
                let () =
                    abce_builder_build_atomic_decrement(builder, cursor, (*(sync)).signal_addr);
                cursor = (cursor).wrapping_offset(
                    ((core::mem::size_of::<SDMA_PKT_ATOMIC>() as usize) as u32) as isize,
                );
            }
            return (((cursor) as isize).wrapping_sub((out) as isize)
                / core::mem::size_of::<core::ffi::c_char>() as isize) as usize;
        }
        if (*(sync)).wait {
            cursor = (cursor).wrapping_add(abce_frame_write_start_gate_poll(
                composer,
                cursor,
                (*(sync)).wait_addr,
            ));
        }
        'switch: {
            match (*(op)).kind {
                value if value == ABCE_OP_KIND_LINEAR => {
                    abce_builder_build_copy(
                        builder,
                        cursor,
                        (*(op)).dst,
                        (*(op)).src,
                        (*(op)).size,
                    );
                    break 'switch;
                }
                value if value == ABCE_OP_KIND_MULTICAST => {
                    abce_builder_build_multicast_copy(
                        builder,
                        cursor,
                        (*(op)).dsts,
                        (*(op)).num_dsts,
                        (*(op)).src,
                        (*(op)).size,
                    );
                    break 'switch;
                }
                value if value == ABCE_OP_KIND_BROADCAST => {
                    abce_builder_build_broadcast_copy(
                        builder,
                        cursor,
                        *((*(op)).dsts).wrapping_offset((0i32) as isize),
                        *((*(op)).dsts).wrapping_offset((1i32) as isize),
                        (*(op)).src,
                        (*(op)).size,
                    );
                    break 'switch;
                }
                value if value == ABCE_OP_KIND_SWAP => {
                    abce_builder_build_swap_copy(
                        builder,
                        cursor,
                        (*(op)).dst,
                        (((*(op)).src) as usize) as *mut core::ffi::c_void,
                        (*(op)).size,
                    );
                    break 'switch;
                }
                value if value == ABCE_OP_KIND_COPY_RECT => {
                    let mut rect_state: abce_frame_rect_cursor_t = { core::mem::zeroed() };
                    (rect_state).cursor = cursor;
                    let _ = abce_builder_build_copy_rect(
                        builder,
                        Some(abce_frame_rect_append),
                        core::ptr::addr_of_mut!(rect_state).cast::<core::ffi::c_void>(),
                        core::ptr::addr_of!((*((*(op)).rect)).dst),
                        core::ptr::addr_of!((*((*(op)).rect)).dst_offset),
                        core::ptr::addr_of!((*((*(op)).rect)).src),
                        core::ptr::addr_of!((*((*(op)).rect)).src_offset),
                        core::ptr::addr_of!((*((*(op)).rect)).range),
                    );
                    break 'switch;
                }
                value if value == ABCE_OP_KIND_INDIRECT => {
                    break 'switch;
                }
                value if value == ABCE_OP_KIND_FILL => {
                    abce_builder_build_fill(
                        builder,
                        cursor,
                        (*(op)).dst,
                        (*(op)).fill_value,
                        (((*(op)).size) / (core::mem::size_of::<u32>() as usize) as usize) as usize,
                    );
                    break 'switch;
                }
                _ => {}
            }
        }
        cursor = (cursor).wrapping_add(abce_frame_payload_bytes(composer, op));
        if (*(sync)).signal {
            abce_builder_build_atomic_decrement(builder, cursor, (*(sync)).signal_addr);
            cursor = (cursor).wrapping_offset(
                ((core::mem::size_of::<SDMA_PKT_ATOMIC>() as usize) as u32) as isize,
            );
        }
        (((cursor) as isize).wrapping_sub((out) as isize)
            / core::mem::size_of::<core::ffi::c_char>() as isize) as usize
    }
}

pub(super) unsafe extern "C" fn abce_frame_write_fan_in_poll(
    composer: *const abce_frame_composer_t,
    out: *mut core::ffi::c_char,
    coordination_word: *mut core::ffi::c_void,
) -> usize {
    unsafe {
        if (*(composer)).is_gfx125plus {
            abce_builder_build_poll_64b(
                (*(composer)).builder,
                out,
                coordination_word,
                0_u64,
                u64::from(4_294_967_295_u32),
            );
            return ((core::mem::size_of::<SDMA_PKT_POLL_MEM_64B_GFX125PLUS>() as usize) as u32)
                as usize;
        }
        abce_builder_build_poll(
            (*(composer)).builder,
            out,
            coordination_word,
            0_u32,
            4_294_967_295_u32,
        );
        ((core::mem::size_of::<SDMA_PKT_POLL_REGMEM>() as usize) as u32) as usize
    }
}

pub(super) unsafe extern "C" fn abce_frame_write_completion(
    composer: *const abce_frame_composer_t,
    out: *mut core::ffi::c_char,
    target: *const abce_signal_ref_t,
) -> usize {
    unsafe {
        if abce_frame_completion_decrements(composer, target) {
            let () =
                abce_builder_build_atomic_decrement((*(composer)).builder, out, (*(target)).value);
            return ((core::mem::size_of::<SDMA_PKT_ATOMIC>() as usize) as u32) as usize;
        }
        if (*(composer)).is_gfx125plus {
            abce_builder_build_fence_64b(
                (*(composer)).builder,
                out,
                (*(target)).value,
                (*(target)).completion_value,
            );
            return ((core::mem::size_of::<SDMA_PKT_FENCE_64B_GFX125PLUS>() as usize) as u32)
                as usize;
        }
        let mut cursor: *mut core::ffi::c_char = out;
        let words: *mut u32 = ((*(target)).value).cast::<u32>();
        if ((*(target)).completion_value) > u64::from(4_294_967_295_u32) {
            abce_builder_build_fence(
                (*(composer)).builder,
                cursor,
                core::ptr::addr_of_mut!((*(words).wrapping_offset((1i32) as isize))),
                (((*(target)).completion_value).wrapping_shr((32_u64) as u32)) as u32,
            );
            cursor = (cursor).wrapping_offset(
                ((core::mem::size_of::<SDMA_PKT_FENCE>() as usize) as u32) as isize,
            );
        }
        abce_builder_build_fence(
            (*(composer)).builder,
            cursor,
            core::ptr::addr_of_mut!((*(words).wrapping_offset((0i32) as isize))),
            ((*(target)).completion_value) as u32,
        );
        cursor = (cursor)
            .wrapping_offset(((core::mem::size_of::<SDMA_PKT_FENCE>() as usize) as u32) as isize);
        (((cursor) as isize).wrapping_sub((out) as isize)
            / core::mem::size_of::<core::ffi::c_char>() as isize) as usize
    }
}

pub(super) unsafe extern "C" fn abce_frame_write_mailbox(
    composer: *const abce_frame_composer_t,
    out: *mut core::ffi::c_char,
    target: *const abce_signal_ref_t,
) -> usize {
    unsafe {
        if ((*(target)).event_mailbox).is_null() {
            return 0_usize;
        }
        let mut cursor: *mut core::ffi::c_char = out;
        abce_builder_build_fence(
            (*(composer)).builder,
            cursor,
            ((*(target)).event_mailbox).cast::<u32>(),
            (*(target)).event_id,
        );
        cursor = (cursor)
            .wrapping_offset(((core::mem::size_of::<SDMA_PKT_FENCE>() as usize) as u32) as isize);
        abce_builder_build_trap((*(composer)).builder, cursor, (*(target)).event_id);
        cursor = (cursor)
            .wrapping_offset(((core::mem::size_of::<SDMA_PKT_TRAP>() as usize) as u32) as isize);
        (((cursor) as isize).wrapping_sub((out) as isize)
            / core::mem::size_of::<core::ffi::c_char>() as isize) as usize
    }
}

pub(super) unsafe extern "C" fn abce_frame_emit_epilogue(
    composer: *const abce_frame_composer_t,
    out: *mut core::ffi::c_char,
    metadata: *const abce_copy_metadata_t,
    multi: bool,
    coordination_word: *mut core::ffi::c_void,
) -> usize {
    unsafe {
        let mut cursor: *mut core::ffi::c_char = out;
        if multi {
            cursor = (cursor).wrapping_add(abce_frame_write_fan_in_poll(
                composer,
                cursor,
                coordination_word,
            ));
        }
        if (((((*(metadata)).coherency).flags).word >> 1) & 0x1u32) != 0
            && abce_frame_should_emit_gcr(composer)
        {
            abce_builder_build_gcr((*(composer)).builder, cursor, false);
            cursor = (cursor).wrapping_offset((abce_frame_gcr_bytes(composer)) as isize);
        }
        if abce_timestamp_surface_enabled(core::ptr::addr_of!((*(metadata)).timestamps)) {
            abce_builder_build_get_global_timestamp(
                (*(composer)).builder,
                cursor,
                ((*(metadata)).timestamps).end_value,
            );
            cursor = (cursor).wrapping_offset(
                ((core::mem::size_of::<SDMA_PKT_TIMESTAMP>() as usize) as u32) as isize,
            );
        }
        cursor = (cursor).wrapping_add(abce_frame_write_completion(
            composer,
            cursor,
            core::ptr::addr_of!((*(metadata)).out),
        ));
        cursor = (cursor).wrapping_add(abce_frame_write_mailbox(
            composer,
            cursor,
            core::ptr::addr_of!((*(metadata)).out),
        ));
        (((cursor) as isize).wrapping_sub((out) as isize)
            / core::mem::size_of::<core::ffi::c_char>() as isize) as usize
    }
}

pub unsafe extern "C" fn abce_frame_composer_emit_frame(
    composer: *const abce_frame_composer_t,
    buffer: *mut core::ffi::c_char,
    job: *const abce_frame_job_t,
) -> usize {
    unsafe {
        let metadata: *const abce_copy_metadata_t = (*(job)).metadata;
        let mut cursor: *mut core::ffi::c_char = buffer;
        if (*(job)).coordinator {
            cursor = (cursor).wrapping_add(abce_frame_emit_prologue(
                composer,
                cursor,
                metadata,
                (*(job)).multi,
                (*(job)).start_gate_required,
                (*(job)).coordination_word,
                abce_frame_folded_dep(composer, job),
            ));
        }
        let mut body_idx: u32 = 0_u32;
        {
            let mut op_idx: u32 = 0_u32;
            let mut first_16 = true;
            loop {
                if !first_16 {
                    op_idx = (op_idx).wrapping_add(1u32);
                }
                first_16 = false;
                if (op_idx) >= ((*(job)).num_ops) {
                    break;
                }
                {
                    if ((*((*(job)).ops).wrapping_offset((op_idx) as isize)).engine)
                        != ((*(job)).engine)
                    {
                        continue;
                    }
                    let sync: abce_body_sync_t = abce_body_sync_for(composer, job, body_idx);
                    cursor = (cursor).wrapping_add(abce_frame_emit_body(
                        composer,
                        cursor,
                        core::ptr::addr_of!((*((*(job)).ops).wrapping_offset((op_idx) as isize))),
                        core::ptr::addr_of!(sync),
                    ));
                    body_idx = (body_idx).wrapping_add(1u32);
                }
            }
        }
        if (*(job)).coordinator && (*(job)).epilogue_required {
            cursor = (cursor).wrapping_add(abce_frame_emit_epilogue(
                composer,
                cursor,
                metadata,
                (*(job)).multi,
                (*(job)).coordination_word,
            ));
        }
        (((cursor) as isize).wrapping_sub((buffer) as isize)
            / core::mem::size_of::<core::ffi::c_char>() as isize) as usize
    }
}
