// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMD agent, logging, event, profiling, and monitor extension entry points.

use std::ffi::c_void;

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::runtime::{InFlightToken, boundary, initialized_mut, lock, map_error};
use crate::{memory, runtime};

/// Flag that skips GPU linear-copy preload for an agent.
const AMD_AGENT_PRELOAD_SKIP_BLITS: u64 = 1 << 1;

/// # Safety
/// `flags` must address eight readable bytes for the duration of this call.
/// A non-null `file` must be an open C `FILE*` kept live while its logging
/// flags are enabled. Replacing the stream, disabling logging, or final HSA
/// shutdown waits for writes already using it to finish. A nonfinal
/// `hsa_shut_down` does not release the stream. Calls made recursively from a
/// custom stream callback are rejected. Null selects stderr.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_enable_logging")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_enable_logging(flags: *mut u8, file: *mut c_void) -> Status {
    boundary(|| {
        if runtime::LogWriteScope::active() {
            return INVALID_RUNTIME_STATE;
        }
        if flags.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The public ABI requires eight readable bytes for this call.
        let mut copied_flags = [0_u8; 8];
        copied_flags.copy_from_slice(unsafe { std::slice::from_raw_parts(flags, 8) });
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        let Some(_call) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        let logging = runtime.logging.clone();
        drop(guard);
        logging.set(copied_flags, file)
    })
}

/// # Safety
/// The callback must remain callable until runtime shutdown. `data` must stay
/// live and synchronized with event-worker access for that interval.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_register_system_event_handler")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_register_system_event_handler(
    callback: SystemEventCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        // SAFETY: The C registration contract keeps callback data live and
        // synchronized until the runtime stops delivering events.
        runtime.register_system_event_handler(callback, unsafe { CallbackArg::new(data) })
    })
}

/// # Safety
/// Any non-null `kind` must address writable `u32` storage.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_coherency_get_type")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_coherency_get_type(agent: HsaAgent, kind: *mut u32) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(index) = runtime.gpu_index(agent) else {
            return INVALID_AGENT;
        };
        if kind.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied writable output storage.
        unsafe { kind.write(runtime.gpus[index].coherency_type) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_coherency_set_type")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Sets the coherency policy exposed by a GPU agent.
pub extern "C" fn hsa_amd_coherency_set_type(agent: HsaAgent, kind: u32) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        if !(AMD_COHERENCY_TYPE_COHERENT..=AMD_COHERENCY_TYPE_NONCOHERENT).contains(&kind) {
            return INVALID_ARGUMENT;
        }
        let Some(index) = runtime.gpu_index(agent) else {
            return INVALID_AGENT;
        };
        runtime.gpus[index].coherency_type = kind;
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_agent_preload")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Requests that the selected agent preload its runtime state.
pub extern "C" fn hsa_amd_agent_preload(agent: HsaAgent, flags: u64) -> Status {
    boundary(|| {
        let (_call, device) = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(index) = runtime.gpu_index(agent) else {
                return INVALID_AGENT;
            };
            if flags & AMD_AGENT_PRELOAD_SKIP_BLITS != 0 {
                return SUCCESS;
            }
            let Some(token) = runtime.inflight.enter() else {
                return OUT_OF_RESOURCES;
            };
            (token, runtime.gpus[index].device.clone())
        };
        match device
            .gpu()
            .and_then(|gpu| gpu.preload_linear_copy().map_err(|failure| failure.error))
        {
            Ok(()) => SUCCESS,
            Err(error) => map_error(error),
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_agent_set_async_scratch_limit")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Rejects asynchronous scratch reclaim, which this backend does not expose.
pub extern "C" fn hsa_amd_agent_set_async_scratch_limit(
    agent: HsaAgent,
    _threshold: usize,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        // GFX1201 does not advertise asynchronous scratch reclaim.
        INVALID_ARGUMENT
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_profiling_async_copy_enable")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Enables or disables timestamps for asynchronous copies.
pub extern "C" fn hsa_amd_profiling_async_copy_enable(enable: bool) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_mut() else {
            return NOT_INITIALIZED;
        };
        runtime.memory_state.async_copy_profiling = if enable {
            memory::AsyncCopyProfiling::Enabled
        } else {
            memory::AsyncCopyProfiling::Disabled
        };
        SUCCESS
    })
}

/// # Safety
/// Any non-null `system_tick` must address writable `u64` storage.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_profiling_convert_tick_to_system_domain")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_profiling_convert_tick_to_system_domain(
    agent: HsaAgent,
    agent_tick: u64,
    system_tick: *mut u64,
) -> Status {
    boundary(|| {
        let (_call, device) = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            if system_tick.is_null() {
                return INVALID_ARGUMENT;
            }
            let Some(index) = runtime.gpu_index(agent) else {
                return INVALID_AGENT;
            };
            let Some(token) = runtime.inflight.enter() else {
                return OUT_OF_RESOURCES;
            };
            (token, runtime.gpus[index].device.clone())
        };
        let counters = match runtime::clock_counters(&device) {
            Ok(counters) => counters,
            Err(error) => return map_error(error),
        };
        let translated = match runtime::translate_gpu_tick(counters, agent_tick) {
            Ok(translated) => translated,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { system_tick.write(translated) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_spm_acquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Acquires the agent’s streaming performance monitor.
pub extern "C" fn hsa_amd_spm_acquire(agent: HsaAgent) -> Status {
    boundary(|| {
        let (_call, device) = match retained_gpu_device(agent) {
            Ok(retained) => retained,
            Err(status) => return status,
        };
        device
            .gpu()
            .and_then(|gpu| gpu.spm_acquire())
            .map_or_else(map_error, |()| SUCCESS)
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_spm_release")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases the SPM resource.
pub extern "C" fn hsa_amd_spm_release(agent: HsaAgent) -> Status {
    boundary(|| {
        let (_call, device) = match retained_gpu_device(agent) {
            Ok(retained) => retained,
            Err(status) => return status,
        };
        device
            .gpu()
            .and_then(|gpu| gpu.spm_release())
            .map_or_else(map_error, |()| SUCCESS)
    })
}

/// Retains the activated device through a native call and shutdown.
/// The token precedes the device so the device drops before shutdown can
/// observe an idle generation.
pub(crate) fn retained_gpu_device(
    agent: HsaAgent,
) -> Result<(InFlightToken, rocddi::device::Device), Status> {
    let guard = lock()?;
    let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
    let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
    let token = runtime.inflight.enter().ok_or(OUT_OF_RESOURCES)?;
    Ok((token, runtime.gpus[index].device.clone()))
}

/// # Safety
/// Scalar output pointers must be valid and writable, and `timeout` readable.
/// A non-null `destination` must hold `size` writable bytes accessible to
/// KFD. Retain both the old and new destinations through successful
/// replacement or unset, or conclusive teardown; a failed call may leave
/// either reachable.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_spm_set_dest_buffer")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_spm_set_dest_buffer(
    agent: HsaAgent,
    size: usize,
    timeout: *mut u32,
    bytes_copied: *mut u32,
    destination: *mut c_void,
    data_loss: *mut bool,
) -> Status {
    boundary(|| {
        let (_call, device) = match retained_gpu_device(agent) {
            Ok(retained) => retained,
            Err(status) => return status,
        };
        if timeout.is_null() || bytes_copied.is_null() || data_loss.is_null() {
            return INVALID_ARGUMENT;
        }
        let size = match u32::try_from(size) {
            Ok(size) => size,
            Err(_) => return INVALID_ARGUMENT,
        };
        // SAFETY: The caller supplied readable/writable scalar storage.
        let mut timeout_value = unsafe { timeout.read() };
        let mut copied_value = 0;
        let mut loss_value = false;
        let result = device.gpu().and_then(|gpu| {
            // SAFETY: The HSA C caller owns destination storage and must keep
            // both old and new buffers writable while KFD may retain them.
            // The raw core API exposes this retained-write obligation.
            unsafe {
                gpu.spm_set_destination(
                    size,
                    &mut timeout_value,
                    &mut copied_value,
                    (!destination.is_null()).then_some(destination as usize),
                    &mut loss_value,
                )
            }
        });
        // KFD returns these fields together with the ioctl result, including
        // partial progress when the native call fails.
        unsafe {
            timeout.write(timeout_value);
            bytes_copied.write(copied_value);
            data_loss.write(loss_value);
        }
        result.map_or_else(map_error, |()| SUCCESS)
    })
}
