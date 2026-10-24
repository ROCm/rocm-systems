// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! PC-sampling ABI, driver capability discovery, and session ownership.

pub(crate) mod session;

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rocddi::device::gpu::profiling::{PcSamplingIntervalUnit, PcSamplingMethod, PcSamplingRequest};

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::runtime::{CallbackScope, lock, map_error};
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
    /// Initializes an empty PC sampling session registry.
    pub(crate) fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            busy: HashSet::new(),
            quarantine: Arc::new(AtomicBool::new(false)),
            retired: Vec::new(),
        }
    }

    /// Reports whether HSA already owns or is changing this agent's session.
    pub(crate) fn agent_busy(&self, agent: HsaAgent) -> bool {
        self.busy.contains(&agent.handle)
            || self.sessions.values().any(|owner| owner.agent == agent)
    }

    /// Quarantines all sessions whose native handler may still be installed.
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

    /// Drops assets whose native sampling sessions have retired.
    pub(crate) fn clear_retired(&mut self) {
        self.retired.clear();
    }
}

/// Translates a native sampling error into an HSA status.
pub(crate) fn sampling_error(error: rocddi::Error) -> Status {
    match error.kind() {
        rocddi::ErrorKind::Unsupported => NOT_SUPPORTED,
        _ => map_error(error),
    }
}

/// Runs a session operation while retaining its frontend owner.
pub(crate) fn with_session(
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
pub(crate) struct CreateArgs {
    /// Optional native trace ID supplied for adopting an existing trace.
    pub(crate) trace_id: Option<NonZeroU32>,
    /// HSA GPU agent that owns the session.
    pub(crate) agent: HsaAgent,
    /// Public sampling method selector.
    pub(crate) method: u32,
    /// Public interval unit selector.
    pub(crate) units: u32,
    /// Requested sampling interval in the selected units.
    pub(crate) interval: usize,
    /// Requested callback buffer size in bytes.
    pub(crate) buffer_size: usize,
    /// Caller callback for delivered sample batches.
    pub(crate) callback: PcSamplingDataReadyCallback,
    /// Caller-owned callback argument.
    pub(crate) callback_data: *mut c_void,
    /// Caller output slot for the new HSA session handle.
    pub(crate) output: *mut HsaPcSampling,
}

#[allow(unsafe_code)]
/// Creates session and records its ownership.
pub(crate) fn create_session(args: &CreateArgs) -> Status {
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

/// Destroys session and releases its owned resources.
pub(crate) fn destroy_session(session: HsaPcSampling) -> Status {
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

#[cfg(test)]
/// PC sampling status and C ABI checks.
mod tests {
    use super::*;
    use crate::api::pc_sampling::*;
    use std::mem::{offset_of, size_of};

    #[test]
    /// Checks that native contention becomes the public resource-busy status.
    fn sampling_busy_uses_the_resource_busy_status() {
        let error = rocddi::Error::Operation {
            kind: rocddi::ErrorKind::Busy,
            detail: "injected active sampling resource",
        };
        assert_eq!(sampling_error(error), RESOURCE_BUSY);
    }

    #[test]
    /// Checks PC sampling entry-point signatures and record field offsets.
    fn entry_points_and_records_match_the_public_abi() {
        /// Configuration-iteration entry-point signature.
        type Iterate =
            extern "C" fn(HsaAgent, PcSamplingConfigurationCallback, *mut c_void) -> Status;
        /// Session-creation entry-point signature.
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
        /// ID-based session-creation entry-point signature.
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
        /// Start, stop, flush, and destroy entry-point signature.
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
