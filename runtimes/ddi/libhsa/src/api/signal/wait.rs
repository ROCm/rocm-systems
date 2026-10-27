// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA signal wait and group C entry points.

use std::sync::atomic::{Ordering, fence};

use crate::ffi::*;
use crate::runtime::{boundary, initialized_mut, lock};
use crate::signal::wait::{
    prepare_multi_wait, signal_group_wait_any, signal_wait, signal_wait_all_open,
    signal_wait_any_open, signal_wait_inputs,
};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_wait_scacquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Waits for one signal with the HSA SC-acquire completion contract.
pub unsafe extern "C" fn hsa_signal_wait_scacquire(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
) -> SignalValue {
    // SAFETY: The public entry point preserves the HSA signal contract.
    unsafe {
        signal_wait(
            signal,
            condition,
            compare_value,
            timeout_hint,
            wait_state_hint,
            Ordering::Acquire,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_wait_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Waits for one signal without acquire synchronization.
pub unsafe extern "C" fn hsa_signal_wait_relaxed(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
) -> SignalValue {
    // SAFETY: The public entry point preserves the HSA signal contract.
    unsafe {
        signal_wait(
            signal,
            condition,
            compare_value,
            timeout_hint,
            wait_state_hint,
            Ordering::Relaxed,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_wait_acquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Uses the deprecated acquire spelling for a single-signal wait.
pub unsafe extern "C" fn hsa_signal_wait_acquire(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
) -> SignalValue {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe {
        hsa_signal_wait_scacquire(
            signal,
            condition,
            compare_value,
            timeout_hint,
            wait_state_hint,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_group_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates a group that retains the listed signals for waits.
pub unsafe extern "C" fn hsa_signal_group_create(
    signal_count: u32,
    signals: *const HsaSignal,
    consumer_count: u32,
    consumers: *const HsaAgent,
    group: *mut HsaSignalGroup,
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
        if signal_count == 0
            || signals.is_null()
            || group.is_null()
            || (consumer_count != 0 && consumers.is_null())
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplies signal_count readable handles.
        let signals = unsafe { std::slice::from_raw_parts(signals, signal_count as usize) };
        if signals
            .iter()
            .any(|signal| !runtime.signal_state.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        if consumer_count != 0 {
            // SAFETY: The caller supplies consumer_count readable handles.
            let consumers =
                unsafe { std::slice::from_raw_parts(consumers, consumer_count as usize) };
            if consumers.iter().any(|agent| !runtime.is_agent(*agent)) {
                return INVALID_AGENT;
            }
        }
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        let mut stored = Vec::new();
        if stored.try_reserve_exact(signals.len()).is_err() {
            return OUT_OF_RESOURCES;
        }
        stored.extend_from_slice(signals);
        runtime.signal_state.insert_group(handle, stored);
        // SAFETY: The caller supplied writable output storage.
        unsafe { group.write(HsaSignalGroup { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_group_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases a signal group and its retained signal references.
pub extern "C" fn hsa_signal_group_destroy(group: HsaSignalGroup) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.signal_state.remove_group(group.handle) {
            SUCCESS
        } else {
            INVALID_SIGNAL_GROUP
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_group_wait_any_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Returns when any group member satisfies its condition, without acquire.
pub unsafe extern "C" fn hsa_signal_group_wait_any_relaxed(
    group: HsaSignalGroup,
    conditions: *const u32,
    compare_values: *const SignalValue,
    wait_state_hint: u32,
    signal: *mut HsaSignal,
    value: *mut SignalValue,
) -> Status {
    // SAFETY: The public entry point forwards the complete signal-group contract.
    unsafe {
        signal_group_wait_any(
            group,
            conditions,
            compare_values,
            wait_state_hint,
            signal,
            value,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_group_wait_any_scacquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Returns when any group member satisfies its SC-acquire condition.
pub unsafe extern "C" fn hsa_signal_group_wait_any_scacquire(
    group: HsaSignalGroup,
    conditions: *const u32,
    compare_values: *const SignalValue,
    wait_state_hint: u32,
    signal: *mut HsaSignal,
    value: *mut SignalValue,
) -> Status {
    // SAFETY: The public entry point forwards the complete signal-group contract.
    let status = unsafe {
        signal_group_wait_any(
            group,
            conditions,
            compare_values,
            wait_state_hint,
            signal,
            value,
        )
    };
    if status == SUCCESS {
        fence(Ordering::Acquire);
    }
    status
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_wait_any")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Returns the first satisfied signal and its original input index.
pub unsafe extern "C" fn hsa_amd_signal_wait_any(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_value: *mut SignalValue,
) -> u32 {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(_) => return u32::MAX,
    };
    let Some(runtime) = guard.as_mut() else {
        return u32::MAX;
    };
    // SAFETY: The caller supplies signal_count readable entries in each array.
    let Some(inputs) = (unsafe { signal_wait_inputs(signal_count, signals, conditions, values) })
    else {
        return u32::MAX;
    };
    // SAFETY: The registry lock protects validation and retention; the returned
    // registration stays alive until this wait returns.
    let registration = match unsafe { prepare_multi_wait(runtime, &inputs.signals) } {
        Some(registration) => registration,
        None => return u32::MAX,
    };
    drop(guard);
    // SAFETY: The caller supplied optional writable output storage.
    unsafe {
        signal_wait_any_open(
            &registration.entries,
            &inputs.conditions,
            &inputs.values,
            timeout_hint,
            wait_state_hint,
            satisfying_value,
            &registration.stop,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_wait_all")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Waits until every signal condition holds or the timeout expires.
pub unsafe extern "C" fn hsa_amd_signal_wait_all(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_values: *mut SignalValue,
) -> u32 {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(_) => return u32::MAX,
    };
    let Some(runtime) = guard.as_mut() else {
        return u32::MAX;
    };
    // SAFETY: The caller supplies signal_count readable entries in each array.
    let Some(inputs) = (unsafe { signal_wait_inputs(signal_count, signals, conditions, values) })
    else {
        return u32::MAX;
    };
    // SAFETY: The registry lock protects validation and retention; the returned
    // registration stays alive until this wait returns.
    let mut registration = match unsafe { prepare_multi_wait(runtime, &inputs.signals) } {
        Some(registration) => registration,
        None => return u32::MAX,
    };
    drop(guard);
    // SAFETY: The caller supplied optional writable output storage.
    unsafe {
        signal_wait_all_open(
            &mut registration.entries,
            &inputs.conditions,
            &inputs.values,
            timeout_hint,
            wait_state_hint,
            satisfying_values,
            &registration.stop,
        )
    }
}
