// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA signal value and atomic C entry points.

use std::sync::atomic::{AtomicI64, Ordering, fence};

use crate::ffi::*;
use crate::runtime::{boundary, lock};
use crate::signal::{
    AmdSignal, batch_needs_flush, flush_device_store, flush_signal_store, signal_value_pointer,
    with_signal,
};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_load_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Loads an HSA signal value with relaxed memory ordering.
pub unsafe extern "C" fn hsa_signal_load_relaxed(signal: HsaSignal) -> SignalValue {
    // SAFETY: Callers must supply a live HSA signal handle.
    unsafe { with_signal(signal, |signal| signal.value.load(Ordering::Relaxed)) }.unwrap_or(0)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_load_scacquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Loads a signal value with the HSA SC-acquire contract.
pub unsafe extern "C" fn hsa_signal_load_scacquire(signal: HsaSignal) -> SignalValue {
    // SAFETY: Callers must supply a live HSA signal handle.
    unsafe { with_signal(signal, |signal| signal.value.load(Ordering::Acquire)) }.unwrap_or(0)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_load_acquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Uses the deprecated acquire spelling for the SC-acquire load.
pub unsafe extern "C" fn hsa_signal_load_acquire(signal: HsaSignal) -> SignalValue {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe { hsa_signal_load_scacquire(signal) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_store_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Stores an HSA signal value with relaxed memory ordering.
pub unsafe extern "C" fn hsa_signal_store_relaxed(signal: HsaSignal, value: SignalValue) {
    // SAFETY: Callers must supply a live HSA signal handle.
    let _ = unsafe {
        with_signal(signal, |signal| {
            match signal.kind.load(Ordering::Acquire) {
                AMD_SIGNAL_KIND_DOORBELL => {
                    let address = signal.value.load(Ordering::Relaxed) as usize;
                    // SAFETY: A queue doorbell signal contains its live MMIO mapping.
                    rocddi::device::gpu::queue::ring_doorbell(address, value as u64);
                }
                AMD_SIGNAL_KIND_LEGACY_DOORBELL => {
                    signal.value.store(value, Ordering::Relaxed);
                    // SAFETY: The HSA caller retains the owning queue through
                    // this store, including its stable intercept owner.
                    crate::queue_intercept::ring(signal.intercept_owner());
                }
                _ => {
                    signal.value.store(value, Ordering::Relaxed);
                    flush_signal_store(signal);
                }
            }
        })
    };
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_silent_store_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Uses the regular relaxed store path for polling-backed signals.
pub unsafe extern "C" fn hsa_signal_silent_store_relaxed(signal: HsaSignal, value: SignalValue) {
    // SAFETY: Same operation is sufficient for polling-backed signals.
    unsafe { hsa_signal_store_relaxed(signal, value) };
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_silent_store_screlease")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Uses the regular SC-release store path for polling-backed signals.
pub unsafe extern "C" fn hsa_signal_silent_store_screlease(signal: HsaSignal, value: SignalValue) {
    // SAFETY: Same operation is sufficient for polling-backed signals.
    unsafe { hsa_signal_store_screlease(signal, value) };
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_store_screlease")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Stores a signal value with the HSA SC-release contract.
pub unsafe extern "C" fn hsa_signal_store_screlease(signal: HsaSignal, value: SignalValue) {
    // SAFETY: Callers must supply a live HSA signal handle.
    let _ = unsafe {
        with_signal(signal, |signal| {
            match signal.kind.load(Ordering::Acquire) {
                AMD_SIGNAL_KIND_DOORBELL => {
                    fence(Ordering::Release);
                    let address = signal.value.load(Ordering::Relaxed) as usize;
                    // SAFETY: A queue doorbell signal contains its live MMIO mapping.
                    rocddi::device::gpu::queue::ring_doorbell(address, value as u64);
                }
                AMD_SIGNAL_KIND_LEGACY_DOORBELL => {
                    signal.value.store(value, Ordering::Release);
                    // SAFETY: The HSA caller retains the owning queue through
                    // this store, including its stable intercept owner.
                    crate::queue_intercept::ring(signal.intercept_owner());
                }
                _ => {
                    signal.value.store(value, Ordering::Release);
                    flush_signal_store(signal);
                }
            }
        })
    };
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_store_release")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Uses the deprecated release spelling for the SC-release store.
pub unsafe extern "C" fn hsa_signal_store_release(signal: HsaSignal, value: SignalValue) {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent release spelling.
    unsafe { hsa_signal_store_screlease(signal, value) };
}

/// Exchanges a signal value using the requested atomic ordering.
unsafe fn signal_exchange(signal: HsaSignal, value: SignalValue, order: Ordering) -> SignalValue {
    // SAFETY: Callers must supply a live HSA user signal.
    unsafe {
        with_signal(signal, |signal| {
            if batch_needs_flush(std::ptr::from_ref(signal) as usize) {
                let previous = signal.value.load(Ordering::Acquire);
                flush_device_store(&signal.value, value);
                previous
            } else {
                signal.value.swap(value, order)
            }
        })
    }
    .unwrap_or(0)
}

/// Compares and conditionally replaces a signal value atomically.
unsafe fn signal_compare_exchange(
    signal: HsaSignal,
    expected: SignalValue,
    value: SignalValue,
    success: Ordering,
    failure: Ordering,
) -> SignalValue {
    // SAFETY: Callers must supply a live HSA user signal.
    unsafe {
        with_signal(signal, |signal| {
            if batch_needs_flush(std::ptr::from_ref(signal) as usize) {
                let observed = signal.value.load(Ordering::Acquire);
                if observed == expected {
                    flush_device_store(&signal.value, value);
                }
                observed
            } else {
                signal
                    .value
                    .compare_exchange(expected, value, success, failure)
                    .unwrap_or_else(|observed| observed)
            }
        })
    }
    .unwrap_or(0)
}

/// Applies a fetch-update operation to a signal atomically.
unsafe fn signal_fetch_update(
    signal: HsaSignal,
    value: SignalValue,
    order: Ordering,
    operation: fn(&AtomicI64, SignalValue, Ordering),
    plain: fn(SignalValue, SignalValue) -> SignalValue,
) {
    // SAFETY: Callers must supply a live HSA user signal.
    let _ = unsafe {
        with_signal(signal, |signal| {
            if batch_needs_flush(std::ptr::from_ref(signal) as usize) {
                let previous = signal.value.load(Ordering::Acquire);
                flush_device_store(&signal.value, plain(previous, value));
            } else {
                operation(&signal.value, value, order);
            }
        })
    };
}

/// Selects wrapping arithmetic or bitwise updates for flush-backed signals.
macro_rules! signal_plain {
    (signal_add) => {
        |old: SignalValue, value: SignalValue| old.wrapping_add(value)
    };
    (signal_subtract) => {
        |old: SignalValue, value: SignalValue| old.wrapping_sub(value)
    };
    (signal_and) => {
        |old: SignalValue, value: SignalValue| old & value
    };
    (signal_or) => {
        |old: SignalValue, value: SignalValue| old | value
    };
    (signal_xor) => {
        |old: SignalValue, value: SignalValue| old ^ value
    };
}

/// Adds an operand to the signal using the requested ordering.
fn signal_add(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_add(operand, order);
}

/// Subtracts an operand from the signal using the requested ordering.
fn signal_subtract(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_sub(operand, order);
}

/// Bitwise-ANDs an operand into the signal value.
fn signal_and(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_and(operand, order);
}

/// Bitwise-ORs an operand into the signal value.
fn signal_or(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_or(operand, order);
}

/// Bitwise-XORs an operand into the signal value.
fn signal_xor(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_xor(operand, order);
}

/// Generates the signal exchange entry ABI entry points.
macro_rules! signal_exchange_entry {
    ($name:ident, $order:expr) => {
        #[doc = concat!("Atomically exchanges an HSA signal through `", stringify!($name), "`.")]
        #[cfg_attr(all(target_arch = "x86_64", not(test)), unsafe(export_name = concat!("__hsa_impl_", stringify!($name))))]
        #[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
        pub unsafe extern "C" fn $name(signal: HsaSignal, value: SignalValue) -> SignalValue {
            // SAFETY: The public entry point preserves the HSA signal contract.
            unsafe { signal_exchange(signal, value, $order) }
        }
    };
}

/// Generates the signal compare exchange entry ABI entry points.
macro_rules! signal_compare_exchange_entry {
    ($name:ident, $success:expr, $failure:expr) => {
        #[doc = concat!("Conditionally exchanges an HSA signal through `", stringify!($name), "`.")]
        #[cfg_attr(all(target_arch = "x86_64", not(test)), unsafe(export_name = concat!("__hsa_impl_", stringify!($name))))]
        #[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
        pub unsafe extern "C" fn $name(
            signal: HsaSignal,
            expected: SignalValue,
            value: SignalValue,
        ) -> SignalValue {
            // SAFETY: The public entry point preserves the HSA signal contract.
            unsafe { signal_compare_exchange(signal, expected, value, $success, $failure) }
        }
    };
}

/// Generates the signal fetch entry ABI entry points.
macro_rules! signal_fetch_entry {
    ($name:ident, $order:expr, $operation:ident) => {
        #[doc = concat!("Applies an atomic signal update through `", stringify!($name), "`.")]
        #[cfg_attr(all(target_arch = "x86_64", not(test)), unsafe(export_name = concat!("__hsa_impl_", stringify!($name))))]
        #[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
        pub unsafe extern "C" fn $name(signal: HsaSignal, value: SignalValue) {
            // SAFETY: The public entry point preserves the HSA signal contract.
            unsafe { signal_fetch_update(signal, value, $order, $operation, signal_plain!($operation)) };
        }
    };
}

signal_exchange_entry!(hsa_signal_exchange_scacq_screl, Ordering::AcqRel);
signal_exchange_entry!(hsa_signal_exchange_acq_rel, Ordering::AcqRel);
signal_exchange_entry!(hsa_signal_exchange_scacquire, Ordering::Acquire);
signal_exchange_entry!(hsa_signal_exchange_acquire, Ordering::Acquire);
signal_exchange_entry!(hsa_signal_exchange_relaxed, Ordering::Relaxed);
signal_exchange_entry!(hsa_signal_exchange_screlease, Ordering::Release);
signal_exchange_entry!(hsa_signal_exchange_release, Ordering::Release);

signal_compare_exchange_entry!(
    hsa_signal_cas_scacq_screl,
    Ordering::AcqRel,
    Ordering::Acquire
);
signal_compare_exchange_entry!(hsa_signal_cas_acq_rel, Ordering::AcqRel, Ordering::Acquire);
signal_compare_exchange_entry!(
    hsa_signal_cas_scacquire,
    Ordering::Acquire,
    Ordering::Acquire
);
signal_compare_exchange_entry!(hsa_signal_cas_acquire, Ordering::Acquire, Ordering::Acquire);
signal_compare_exchange_entry!(hsa_signal_cas_relaxed, Ordering::Relaxed, Ordering::Relaxed);
signal_compare_exchange_entry!(
    hsa_signal_cas_screlease,
    Ordering::Release,
    Ordering::Relaxed
);
signal_compare_exchange_entry!(hsa_signal_cas_release, Ordering::Release, Ordering::Relaxed);

signal_fetch_entry!(hsa_signal_add_scacq_screl, Ordering::AcqRel, signal_add);
signal_fetch_entry!(hsa_signal_add_acq_rel, Ordering::AcqRel, signal_add);
signal_fetch_entry!(hsa_signal_add_scacquire, Ordering::Acquire, signal_add);
signal_fetch_entry!(hsa_signal_add_acquire, Ordering::Acquire, signal_add);
signal_fetch_entry!(hsa_signal_add_relaxed, Ordering::Relaxed, signal_add);
signal_fetch_entry!(hsa_signal_add_screlease, Ordering::Release, signal_add);
signal_fetch_entry!(hsa_signal_add_release, Ordering::Release, signal_add);

signal_fetch_entry!(
    hsa_signal_subtract_scacq_screl,
    Ordering::AcqRel,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_acq_rel,
    Ordering::AcqRel,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_scacquire,
    Ordering::Acquire,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_acquire,
    Ordering::Acquire,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_relaxed,
    Ordering::Relaxed,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_screlease,
    Ordering::Release,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_release,
    Ordering::Release,
    signal_subtract
);

signal_fetch_entry!(hsa_signal_and_scacq_screl, Ordering::AcqRel, signal_and);
signal_fetch_entry!(hsa_signal_and_acq_rel, Ordering::AcqRel, signal_and);
signal_fetch_entry!(hsa_signal_and_scacquire, Ordering::Acquire, signal_and);
signal_fetch_entry!(hsa_signal_and_acquire, Ordering::Acquire, signal_and);
signal_fetch_entry!(hsa_signal_and_relaxed, Ordering::Relaxed, signal_and);
signal_fetch_entry!(hsa_signal_and_screlease, Ordering::Release, signal_and);
signal_fetch_entry!(hsa_signal_and_release, Ordering::Release, signal_and);

signal_fetch_entry!(hsa_signal_or_scacq_screl, Ordering::AcqRel, signal_or);
signal_fetch_entry!(hsa_signal_or_acq_rel, Ordering::AcqRel, signal_or);
signal_fetch_entry!(hsa_signal_or_scacquire, Ordering::Acquire, signal_or);
signal_fetch_entry!(hsa_signal_or_acquire, Ordering::Acquire, signal_or);
signal_fetch_entry!(hsa_signal_or_relaxed, Ordering::Relaxed, signal_or);
signal_fetch_entry!(hsa_signal_or_screlease, Ordering::Release, signal_or);
signal_fetch_entry!(hsa_signal_or_release, Ordering::Release, signal_or);

signal_fetch_entry!(hsa_signal_xor_scacq_screl, Ordering::AcqRel, signal_xor);
signal_fetch_entry!(hsa_signal_xor_acq_rel, Ordering::AcqRel, signal_xor);
signal_fetch_entry!(hsa_signal_xor_scacquire, Ordering::Acquire, signal_xor);
signal_fetch_entry!(hsa_signal_xor_acquire, Ordering::Acquire, signal_xor);
signal_fetch_entry!(hsa_signal_xor_relaxed, Ordering::Relaxed, signal_xor);
signal_fetch_entry!(hsa_signal_xor_screlease, Ordering::Release, signal_xor);
signal_fetch_entry!(hsa_signal_xor_release, Ordering::Release, signal_xor);

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_value_pointer")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Returns a stable host pointer to an eligible signal value.
pub unsafe extern "C" fn hsa_amd_signal_value_pointer(
    signal: HsaSignal,
    value: *mut *mut SignalValue,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        if !runtime.signal_state.owns_signal(signal) {
            return INVALID_SIGNAL;
        }
        let interrupt = runtime.signal_state.is_interrupt_signal(signal);
        // SAFETY: owns_signal validated the handle while the runtime lock
        // prevents destruction of its backing storage.
        let Some(pointer) =
            (unsafe { with_signal(signal, |signal| signal_value_pointer(signal, interrupt)) })
        else {
            return INVALID_SIGNAL;
        };
        let pointer = match pointer {
            Ok(pointer) => pointer,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { value.write(pointer) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_get_event_id")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the native event ID of an interrupt-backed signal.
pub unsafe extern "C" fn hsa_amd_signal_get_event_id(
    signal: HsaSignal,
    event_id: *mut u32,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if event_id.is_null() {
            return INVALID_ARGUMENT;
        }
        if !runtime.signal_state.owns_signal(signal) {
            return INVALID_SIGNAL;
        }
        // SAFETY: owns_signal validated this live slab slot and the caller
        // supplied writable output storage.
        unsafe { event_id.write((*((signal.handle as usize) as *const AmdSignal)).event_id()) };
        SUCCESS
    })
}
