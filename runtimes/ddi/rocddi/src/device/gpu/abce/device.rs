// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust ABCE device mechanisms.

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

use super::abce_builder_requires_gcr;
use super::abi::{
    ABCE_STATUS_OK, ABCE_STATUS_UNIMPLEMENTED, abce_builder_t, abce_device_engine_t,
    abce_device_handle_t, abce_device_ring_t, abce_status_t,
};

pub unsafe extern "C" fn abce_device_engine_initialize(
    builder: *const abce_builder_t,
    ring: *const abce_device_ring_t,
    engine_id: u32,
    device_atomic_support: bool,
    out_engine: *mut abce_device_engine_t,
) -> abce_status_t {
    unsafe {
        if abce_builder_requires_gcr(builder) {
            return ABCE_STATUS_UNIMPLEMENTED;
        }
        let _ = libc::memset(
            (out_engine).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<abce_device_engine_t>() as usize,
        );
        (*(out_engine)).abi_version = 1u32;
        (*(out_engine)).engine_id = engine_id;
        (*(out_engine)).device_atomic_support = device_atomic_support;
        (*(out_engine)).builder = *(builder);
        (*(out_engine)).ring = *(ring);
        ABCE_STATUS_OK
    }
}

#[must_use]
pub unsafe extern "C" fn abce_device_handle_valid(handle: *const abce_device_handle_t) -> bool {
    unsafe {
        ((!(handle).is_null() && !((*(handle)).engine).is_null())
            && (((*(handle)).abi_version) == (1u32)))
            && (((*((*(handle)).engine)).abi_version) == (1u32))
    }
}
