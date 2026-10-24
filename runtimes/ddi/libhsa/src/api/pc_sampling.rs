// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! PC sampling C entry points and extension table policy.

use std::ffi::c_void;
use std::num::NonZeroU32;

use rocddi::device::gpu::profiling::{
    PcSamplingConfigurationSet, PcSamplingIntervalUnit, PcSamplingMethod,
};

use crate::ffi::*;
use crate::pc_sampling::session::SessionHandle;
use crate::pc_sampling::{
    CreateArgs, create_session, destroy_session, sampling_error, with_session,
};
use crate::runtime::{CallbackScope, boundary, lock};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_iterate_configuration")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Visits PC sampling configurations supported by an agent.
pub extern "C" fn hsa_ven_amd_pcs_iterate_configuration(
    agent: HsaAgent,
    configuration_callback: PcSamplingConfigurationCallback,
    callback_data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = configuration_callback else {
            return INVALID_ARGUMENT;
        };
        let (device, _token) = {
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
            if !runtime.gpus[index].pc_sampling_supported {
                return NOT_SUPPORTED;
            }
            if runtime.pc_sampling_state.agent_busy(agent) {
                return SUCCESS;
            }
            let Some(token) = runtime.inflight.enter() else {
                return OUT_OF_RESOURCES;
            };
            (runtime.gpus[index].device.clone(), token)
        };
        let configurations = match device
            .gpu()
            .and_then(|gpu| gpu.pc_sampling_configurations(PcSamplingConfigurationSet::Available))
        {
            Ok(configurations) => configurations,
            Err(error) => return sampling_error(error),
        };
        let _scope = CallbackScope::enter();
        for configuration in configurations {
            let (Ok(minimum_interval), Ok(maximum_interval)) = (
                usize::try_from(configuration.minimum_interval),
                usize::try_from(configuration.maximum_interval),
            ) else {
                continue;
            };
            let record = HsaPcSamplingConfiguration {
                method: match configuration.method {
                    PcSamplingMethod::HostTrap => 0,
                    PcSamplingMethod::Stochastic => 1,
                },
                units: match configuration.unit {
                    PcSamplingIntervalUnit::Microseconds => 0,
                    PcSamplingIntervalUnit::ClockCycles => 1,
                    PcSamplingIntervalUnit::Instructions => 2,
                },
                minimum_interval,
                maximum_interval,
                flags: configuration.flags,
            };
            // SAFETY: The callback is synchronous and the record stays live
            // through its return. The caller retains callback_data.
            match unsafe { callback(&raw const record, callback_data) } {
                SUCCESS => (),
                INFO_BREAK => return SUCCESS,
                status => return status,
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(clippy::too_many_arguments)]
/// Creates a PC sampling session for a selected configuration.
pub extern "C" fn hsa_ven_amd_pcs_create(
    agent: HsaAgent,
    method: u32,
    units: u32,
    interval: usize,
    _latency: usize,
    buffer_size: usize,
    data_ready_callback: PcSamplingDataReadyCallback,
    client_callback_data: *mut c_void,
    pc_sampling: *mut HsaPcSampling,
) -> Status {
    boundary(|| {
        create_session(&CreateArgs {
            trace_id: None,
            agent,
            method,
            units,
            interval,
            buffer_size,
            callback: data_ready_callback,
            callback_data: client_callback_data,
            output: pc_sampling,
        })
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_create_from_id")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(clippy::too_many_arguments)]
/// Creates a PC sampling session by configuration ID.
pub extern "C" fn hsa_ven_amd_pcs_create_from_id(
    trace_id: u32,
    agent: HsaAgent,
    method: u32,
    units: u32,
    interval: usize,
    _latency: usize,
    buffer_size: usize,
    data_ready_callback: PcSamplingDataReadyCallback,
    client_callback_data: *mut c_void,
    pc_sampling: *mut HsaPcSampling,
) -> Status {
    boundary(|| {
        let Some(trace_id) = NonZeroU32::new(trace_id) else {
            return INVALID_ARGUMENT;
        };
        create_session(&CreateArgs {
            trace_id: Some(trace_id),
            agent,
            method,
            units,
            interval,
            buffer_size,
            callback: data_ready_callback,
            callback_data: client_callback_data,
            output: pc_sampling,
        })
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_start")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Changes the pcs session through the HSA C ABI.
pub extern "C" fn hsa_ven_amd_pcs_start(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| with_session(pc_sampling, SessionHandle::start))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_stop")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Changes the pcs session through the HSA C ABI.
pub extern "C" fn hsa_ven_amd_pcs_stop(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| with_session(pc_sampling, SessionHandle::stop))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_flush")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Changes the pcs session through the HSA C ABI.
pub extern "C" fn hsa_ven_amd_pcs_flush(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| with_session(pc_sampling, SessionHandle::flush))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Stops and releases a PC sampling session.
pub extern "C" fn hsa_ven_amd_pcs_destroy(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| destroy_session(pc_sampling))
}

#[cfg(any(test, not(all(target_os = "linux", target_arch = "x86_64"))))]
/// Returns the PC sampling extension functions in public ABI table order.
pub(crate) fn extension_table() -> [usize; 7] {
    [
        hsa_ven_amd_pcs_iterate_configuration as *const () as usize,
        hsa_ven_amd_pcs_create as *const () as usize,
        hsa_ven_amd_pcs_create_from_id as *const () as usize,
        hsa_ven_amd_pcs_destroy as *const () as usize,
        hsa_ven_amd_pcs_start as *const () as usize,
        hsa_ven_amd_pcs_stop as *const () as usize,
        hsa_ven_amd_pcs_flush as *const () as usize,
    ]
}
