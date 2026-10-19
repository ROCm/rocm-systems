// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! PC-sampling ABI, driver capability discovery, and session ownership.

pub(crate) mod session;

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rocddi::device::gpu::profiling::{
    PcSamplingConfigurationSet, PcSamplingIntervalUnit, PcSamplingMethod, PcSamplingRequest,
};

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::runtime::{CallbackScope, boundary, lock, map_error};
use session::{Resources, RetiredAssets, SessionHandle};

/// HSA handles and uncertain cleanup for PC sampling sessions.
pub(crate) struct PcSamplingState {
    /// Live sessions indexed by public handle.
    sessions: HashMap<u64, Arc<SessionHandle>>,
    /// Sessions undergoing a native transition.
    busy: HashSet<u64>,
    /// Prevents reuse after uncertain native cleanup.
    quarantine: Arc<AtomicBool>,
    /// Trap assets retained until native users have stopped.
    retired: Vec<RetiredAssets>,
}

impl PcSamplingState {
    pub(crate) fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            busy: HashSet::new(),
            quarantine: Arc::new(AtomicBool::new(false)),
            retired: Vec::new(),
        }
    }

    /// Reports whether HSA already owns or is changing this agent's session.
    fn agent_busy(&self, agent: HsaAgent) -> bool {
        self.busy.contains(&agent.handle)
            || self.sessions.values().any(|owner| owner.agent == agent)
    }

    pub(crate) fn quarantine_all(&self) {
        for session in self.sessions.values() {
            session.quarantine();
        }
    }

    /// Releases every native sampling session before device teardown.
    /// Failure retains the remaining state with the whole HSA generation.
    pub(crate) fn shutdown(&mut self) -> bool {
        if self.quarantine.load(Ordering::Acquire) {
            self.quarantine_all();
            return false;
        }
        while let Some(handle) = self.sessions.keys().next().copied() {
            let Some(session) = self.sessions.remove(&handle) else {
                continue;
            };
            if let Ok(retired) = session.destroy() {
                self.retired.push(retired);
            } else {
                session.quarantine();
                self.sessions.insert(handle, session);
                return false;
            }
        }
        true
    }

    pub(crate) fn clear_retired(&mut self) {
        self.retired.clear();
    }
}

fn sampling_error(error: rocddi::Error) -> Status {
    match error.kind() {
        rocddi::ErrorKind::Unsupported => NOT_SUPPORTED,
        rocddi::ErrorKind::Busy => RESOURCE_BUSY,
        _ => map_error(error),
    }
}

fn with_session(
    session: HsaPcSampling,
    operation: impl FnOnce(&SessionHandle) -> Status,
) -> Status {
    if CallbackScope::active() {
        return INVALID_RUNTIME_STATE;
    }
    if session.handle == 0 {
        return INVALID_ARGUMENT;
    }
    let (owner, _token) = {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(owner) = runtime
            .pc_sampling_state
            .sessions
            .get(&session.handle)
            .cloned()
        else {
            return INVALID_ARGUMENT;
        };
        let Some(token) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        (owner, token)
    };
    operation(&owner)
}

/// Validated C arguments shared by both PC sampling creation entry points.
struct CreateArgs {
    /// Optional native trace ID supplied for adopting an existing trace.
    trace_id: Option<NonZeroU32>,
    /// HSA GPU agent that owns the session.
    agent: HsaAgent,
    /// Public sampling method selector.
    method: u32,
    /// Public interval unit selector.
    units: u32,
    /// Requested sampling interval in the selected units.
    interval: usize,
    /// Requested callback buffer size in bytes.
    buffer_size: usize,
    /// Caller callback for delivered sample batches.
    callback: PcSamplingDataReadyCallback,
    /// Caller-owned callback argument.
    callback_data: *mut c_void,
    /// Caller output slot for the new HSA session handle.
    output: *mut HsaPcSampling,
}

#[allow(unsafe_code)]
fn create_session(args: &CreateArgs) -> Status {
    if CallbackScope::active() {
        return INVALID_RUNTIME_STATE;
    }
    if args.output.is_null()
        || args.callback.is_none()
        || args.buffer_size == 0
        || args.buffer_size % 64 != 0
        || args.interval == 0
    {
        return INVALID_ARGUMENT;
    }
    let method = match args.method {
        0 => PcSamplingMethod::HostTrap,
        1 => PcSamplingMethod::Stochastic,
        _ => return INVALID_ARGUMENT,
    };
    let unit = match args.units {
        0 => PcSamplingIntervalUnit::Microseconds,
        1 => PcSamplingIntervalUnit::ClockCycles,
        2 => PcSamplingIntervalUnit::Instructions,
        _ => return INVALID_ARGUMENT,
    };
    let request = PcSamplingRequest {
        method,
        unit,
        interval: args.interval as u64,
    };
    let (handle, device, info, page_size, quarantine, _token) = {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_mut() else {
            return NOT_INITIALIZED;
        };
        let Some(index) = runtime.gpu_index(args.agent) else {
            return INVALID_AGENT;
        };
        let info = runtime.gpus[index].info;
        if !session::supported_gpu(info) {
            return NOT_SUPPORTED;
        }
        if runtime.pc_sampling_state.agent_busy(args.agent) {
            return RESOURCE_BUSY;
        }
        let Some(token) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.pc_sampling_state.busy.insert(args.agent.handle);
        (
            handle,
            runtime.gpus[index].device.clone(),
            info,
            runtime.host_page_size,
            runtime.pc_sampling_state.quarantine.clone(),
            token,
        )
    };
    // SAFETY: The C registration contract keeps callback data live until
    // session destruction has stopped callback delivery.
    let callback_data = unsafe { CallbackArg::new(args.callback_data) };
    let result = Resources::prepare(
        device,
        info,
        request,
        args.trace_id,
        args.buffer_size,
        page_size,
        args.callback,
        callback_data,
    )
    .and_then(|resources| SessionHandle::create(resources, args.agent, &quarantine));
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => {
            if let Ok(owner) = result {
                if let Ok(retired) = owner.destroy() {
                    std::mem::forget(retired);
                } else {
                    quarantine.store(true, Ordering::Release);
                    owner.quarantine();
                    std::mem::forget(owner);
                }
            }
            return status;
        }
    };
    let Some(runtime) = guard.as_mut() else {
        drop(guard);
        if let Ok(owner) = result {
            if let Ok(retired) = owner.destroy() {
                std::mem::forget(retired);
            } else {
                quarantine.store(true, Ordering::Release);
                owner.quarantine();
                std::mem::forget(owner);
            }
        }
        return NOT_INITIALIZED;
    };
    runtime.pc_sampling_state.busy.remove(&args.agent.handle);
    let owner = match result {
        Ok(owner) => owner,
        Err(status) => return status,
    };
    runtime.pc_sampling_state.sessions.insert(handle, owner);
    // SAFETY: The caller supplied one writable output handle.
    unsafe { args.output.write(HsaPcSampling { handle }) };
    SUCCESS
}

fn destroy_session(session: HsaPcSampling) -> Status {
    if CallbackScope::active() {
        return INVALID_RUNTIME_STATE;
    }
    if session.handle == 0 {
        return INVALID_ARGUMENT;
    }
    let (owner, quarantine, _token) = {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_mut() else {
            return NOT_INITIALIZED;
        };
        let Some(owner) = runtime
            .pc_sampling_state
            .sessions
            .get(&session.handle)
            .cloned()
        else {
            return INVALID_ARGUMENT;
        };
        if runtime.pc_sampling_state.busy.contains(&owner.agent.handle) {
            return RESOURCE_BUSY;
        }
        let Some(token) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        runtime.pc_sampling_state.busy.insert(owner.agent.handle);
        (owner, runtime.pc_sampling_state.quarantine.clone(), token)
    };
    let outcome = owner.destroy();
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => {
            quarantine.store(true, Ordering::Release);
            owner.quarantine();
            if let Ok(retired) = outcome {
                std::mem::forget(retired);
            }
            return status;
        }
    };
    let Some(runtime) = guard.as_mut() else {
        if let Ok(retired) = outcome {
            std::mem::forget(retired);
            return SUCCESS;
        }
        quarantine.store(true, Ordering::Release);
        owner.quarantine();
        return INVALID_RUNTIME_STATE;
    };
    runtime.pc_sampling_state.busy.remove(&owner.agent.handle);
    match outcome {
        Ok(retired) => {
            if runtime
                .pc_sampling_state
                .sessions
                .get(&session.handle)
                .is_some_and(|current| Arc::ptr_eq(current, &owner))
            {
                runtime.pc_sampling_state.sessions.remove(&session.handle);
            }
            runtime.pc_sampling_state.retired.push(retired);
            SUCCESS
        }
        Err(status) => status,
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_iterate_configuration")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
pub extern "C" fn hsa_ven_amd_pcs_start(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| with_session(pc_sampling, SessionHandle::start))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_stop")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_ven_amd_pcs_stop(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| with_session(pc_sampling, SessionHandle::stop))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_flush")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_ven_amd_pcs_flush(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| with_session(pc_sampling, SessionHandle::flush))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_pcs_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_ven_amd_pcs_destroy(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| destroy_session(pc_sampling))
}

#[cfg(any(test, not(all(target_os = "linux", target_arch = "x86_64"))))]
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn entry_points_and_records_match_the_public_abi() {
        type Iterate =
            extern "C" fn(HsaAgent, PcSamplingConfigurationCallback, *mut c_void) -> Status;
        type Create = extern "C" fn(
            HsaAgent,
            u32,
            u32,
            usize,
            usize,
            usize,
            PcSamplingDataReadyCallback,
            *mut c_void,
            *mut HsaPcSampling,
        ) -> Status;
        type CreateFromId = extern "C" fn(
            u32,
            HsaAgent,
            u32,
            u32,
            usize,
            usize,
            usize,
            PcSamplingDataReadyCallback,
            *mut c_void,
            *mut HsaPcSampling,
        ) -> Status;
        type Control = extern "C" fn(HsaPcSampling) -> Status;

        let _: Iterate = hsa_ven_amd_pcs_iterate_configuration;
        let _: Create = hsa_ven_amd_pcs_create;
        let _: CreateFromId = hsa_ven_amd_pcs_create_from_id;
        let _: Control = hsa_ven_amd_pcs_destroy;
        let _: Control = hsa_ven_amd_pcs_start;
        let _: Control = hsa_ven_amd_pcs_stop;
        let _: Control = hsa_ven_amd_pcs_flush;
        assert_eq!(size_of::<HsaPcSampling>(), 8);
        assert_eq!(size_of::<HsaPcSamplingConfiguration>(), 32);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, method), 0);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, units), 4);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, minimum_interval), 8);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, maximum_interval), 16);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, flags), 24);
    }
}
