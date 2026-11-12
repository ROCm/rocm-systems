// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Optional profiler callbacks for scratch transitions.

use std::ffi::c_void;

use crate::ffi::HsaQueue;
#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
use crate::runtime::CallbackScope;

/// Tool table slot for scratch-allocation start events.
const ALLOC_START: usize = 0;
/// Tool table slot for scratch-allocation end events.
const ALLOC_END: usize = 1;
/// Tool table slot for scratch-free start events.
const FREE_START: usize = 2;
/// Tool table slot for scratch-free end events.
const FREE_END: usize = 3;

/// `ROCr` tool event emitted before scratch allocation begins.
#[repr(C)]
struct ScratchAllocStart {
    /// Tool ABI event discriminator.
    kind: i32,
    /// Public queue requesting scratch.
    queue: *const HsaQueue,
    /// Reserved tool ABI flags.
    flags: i32,
    /// Dispatch that triggered the allocation.
    dispatch_id: u64,
}

/// `ROCr` tool event emitted after scratch allocation completes.
#[repr(C)]
struct ScratchAllocEnd {
    /// Tool ABI event discriminator.
    kind: i32,
    /// Public queue receiving scratch.
    queue: *const HsaQueue,
    /// Reserved tool ABI flags.
    flags: i32,
    /// Dispatch that triggered the allocation.
    dispatch_id: u64,
    /// Scratch allocation byte size.
    size: usize,
    /// Number of scratch slots allocated.
    num_slots: usize,
}

/// `ROCr` tool event emitted before or after scratch release.
#[repr(C)]
struct ScratchFree {
    /// Tool ABI event discriminator.
    kind: i32,
    /// Public queue releasing scratch.
    queue: *const HsaQueue,
    /// Reserved tool ABI flags.
    flags: i32,
}

/// Invokes the selected tool event hook when a tool has installed one.
fn emit(slot: usize, event: *const c_void) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
    if let Some(callback) = crate::api_trace::scratch_callback(slot) {
        let _scope = CallbackScope::enter();
        // SAFETY: The stack event has the C layout required by this slot and
        // lives through the synchronous tool callback.
        let _ = unsafe { callback(event) };
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64", not(test))))]
    let _ = (slot, event);
}

/// Reports the start of a scratch allocation for a dispatch.
pub(crate) fn alloc_start(queue: *const HsaQueue, dispatch_id: u64) {
    let event = ScratchAllocStart {
        kind: 1,
        queue,
        flags: 0,
        dispatch_id,
    };
    emit(ALLOC_START, (&raw const event).cast());
}

/// Reports the size and slot count of a completed scratch allocation.
pub(crate) fn alloc_end(queue: *const HsaQueue, dispatch_id: u64, size: usize, slots: usize) {
    let event = ScratchAllocEnd {
        kind: 2,
        queue,
        flags: 0,
        dispatch_id,
        size,
        num_slots: slots,
    };
    emit(ALLOC_END, (&raw const event).cast());
}

/// Reports the start of scratch release to an attached tool.
pub(crate) fn free_start(queue: *const HsaQueue) {
    let event = ScratchFree {
        kind: 3,
        queue,
        flags: 0,
    };
    emit(FREE_START, (&raw const event).cast());
}

/// Reports completion of scratch release to an attached tool.
pub(crate) fn free_end(queue: *const HsaQueue) {
    let event = ScratchFree {
        kind: 4,
        queue,
        flags: 0,
    };
    emit(FREE_END, (&raw const event).cast());
}
