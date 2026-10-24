// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA signal creation, destruction, and IPC C entry points.

use crate::ffi::*;
use crate::memory::signal_batch_pool;
use crate::runtime::{boundary, initialized_mut, lock};
use crate::signal::{signal_uses_interrupt, valid_consumers};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates an HSA signal with validated consumers.
pub unsafe extern "C" fn hsa_signal_create(
    initial_value: SignalValue,
    num_consumers: u32,
    consumers: *const HsaAgent,
    signal: *mut HsaSignal,
) -> Status {
    boundary(|| {
        if signal.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        // SAFETY: The public ABI promises num_consumers readable handles when nonzero.
        let interrupt =
            match unsafe { signal_uses_interrupt(runtime.cpus.len(), num_consumers, consumers, 0) }
            {
                Ok(interrupt) => interrupt,
                Err(status) => return status,
            };
        let created = match runtime.create_signal(initial_value, interrupt) {
            Ok(signal) => signal,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { signal.write(created) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates an AMD signal with attributes and consumer selection.
pub unsafe extern "C" fn hsa_amd_signal_create(
    initial_value: SignalValue,
    num_consumers: u32,
    consumers: *const HsaAgent,
    attributes: u64,
    signal: *mut HsaSignal,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if signal.is_null() {
            return INVALID_ARGUMENT;
        }
        // Default/IPC signals do not use the consumer list. In particular, do
        // not inspect a pointer the caller is allowed to omit for these modes.
        // SAFETY: The public ABI promises num_consumers readable handles when
        // the selected signal kind consumes this list.
        let interrupt = match unsafe {
            signal_uses_interrupt(runtime.cpus.len(), num_consumers, consumers, attributes)
        } {
            Ok(interrupt) => interrupt,
            Err(status) => return status,
        };
        let created = match if attributes & AMD_SIGNAL_IPC != 0 {
            runtime.signal_state.create_ipc_signal(
                &runtime.gpus,
                runtime.host_page_size,
                initial_value,
            )
        } else {
            runtime.create_signal(initial_value, interrupt)
        } {
            Ok(signal) => signal,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { signal.write(created) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_batch_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates a batch of signals with shared native backing.
pub unsafe extern "C" fn hsa_amd_signal_batch_create(
    count: u32,
    initial_values: *const SignalValue,
    num_consumers: u32,
    consumers: *const HsaAgent,
    attributes: u64,
    pool: HsaMemoryPool,
    batch: *mut HsaAmdSignalBatch,
    signals: *mut HsaSignal,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if count == 0 || initial_values.is_null() || batch.is_null() || signals.is_null()
            || attributes & AMD_SIGNAL_IPC != 0
            // SAFETY: The ABI requires num_consumers readable handles.
            || !unsafe { valid_consumers(num_consumers, consumers) }
        {
            return INVALID_ARGUMENT;
        }
        let Some(pool_kind) = signal_batch_pool(runtime, pool) else {
            return INVALID_MEMORY_POOL;
        };
        // SAFETY: Validated non-null input with count readable values.
        let values = unsafe { std::slice::from_raw_parts(initial_values, count as usize) };
        // SAFETY: The consumer list was validated before this read.
        let consumer_slice = if num_consumers == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(consumers, num_consumers as usize) }
        };
        // SAFETY: The validated consumer list meets the signal selector's ABI.
        let interrupt = match unsafe {
            signal_uses_interrupt(runtime.cpus.len(), num_consumers, consumers, attributes)
        } {
            Ok(value) => value,
            Err(status) => return status,
        };
        let (created_batch, created_signals) =
            match runtime.create_signal_batch(values, interrupt, pool_kind, consumer_slice) {
                Ok(created) => created,
                Err(status) => return status,
            };
        // SAFETY: The caller supplies count writable handles and a batch slot.
        // Inputs have all been consumed before either output is written.
        unsafe {
            std::ptr::copy_nonoverlapping(created_signals.as_ptr(), signals, count as usize);
            batch.write(created_batch);
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_batch_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases a signal batch after all members retire.
pub extern "C" fn hsa_amd_signal_batch_destroy(batch: HsaAmdSignalBatch) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        runtime.signal_state.destroy_batch(batch)
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Invalidates a signal and retires its backing safely.
pub unsafe extern "C" fn hsa_signal_destroy(signal: HsaSignal) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        runtime.signal_state.destroy_signal(signal)
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ipc_signal_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates an IPC-capable signal in shared native storage.
pub unsafe extern "C" fn hsa_amd_ipc_signal_create(
    signal: HsaSignal,
    handle: *mut HsaAmdIpcSignal,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if handle.is_null() {
            return INVALID_ARGUMENT;
        }
        let words = match runtime.signal_state.export_ipc_signal(signal) {
            Ok(words) => words,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { handle.write(HsaAmdIpcSignal { handle: words }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ipc_signal_attach")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Attaches an exported IPC signal and returns its local HSA handle.
pub unsafe extern "C" fn hsa_amd_ipc_signal_attach(
    handle: *const HsaAmdIpcSignal,
    signal: *mut HsaSignal,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if handle.is_null() || signal.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable IPC handle.
        let words = unsafe { handle.read() }.handle;
        // SAFETY: attach_ipc_signal validates the imported mapping and shared header.
        let attached = match unsafe {
            runtime
                .signal_state
                .attach_ipc_signal(&runtime.session, &runtime.gpus, words)
        } {
            Ok(signal) => signal,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { signal.write(attached) };
        SUCCESS
    })
}
