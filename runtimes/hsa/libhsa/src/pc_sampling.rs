// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! PC-sampling ABI symbols retained while the capability is disabled.
//!
//! Sampling is unqualified. These entry points preserve the exported C ABI
//! and reject creation without publishing a session.

use std::ffi::c_void;

use crate::ffi::*;
use crate::runtime::{CallbackScope, boundary, lock};

fn unavailable_agent(agent: HsaAgent) -> Status {
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
    NOT_SUPPORTED
}

fn absent_session(session: HsaPcSampling) -> Status {
    if CallbackScope::active() {
        return INVALID_RUNTIME_STATE;
    }
    if session.handle == 0 {
        return INVALID_ARGUMENT;
    }
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    if guard.as_ref().is_none() {
        NOT_INITIALIZED
    } else {
        INVALID_ARGUMENT
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ven_amd_pcs_iterate_configuration(
    agent: HsaAgent,
    configuration_callback: PcSamplingConfigurationCallback,
    _callback_data: *mut c_void,
) -> Status {
    boundary(|| {
        if configuration_callback.is_none() {
            return INVALID_ARGUMENT;
        }
        unavailable_agent(agent)
    })
}

#[allow(clippy::too_many_arguments)]
fn reject_create(
    trace_id: Option<u32>,
    agent: HsaAgent,
    buffer_size: usize,
    data_ready_callback: PcSamplingDataReadyCallback,
    output: *mut HsaPcSampling,
) -> Status {
    if output.is_null()
        || data_ready_callback.is_none()
        || buffer_size == 0
        || buffer_size % 128 != 0
        || trace_id == Some(0)
    {
        return INVALID_ARGUMENT;
    }
    unavailable_agent(agent)
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn hsa_ven_amd_pcs_create(
    agent: HsaAgent,
    _method: u32,
    _units: u32,
    _interval: usize,
    _latency: usize,
    buffer_size: usize,
    data_ready_callback: PcSamplingDataReadyCallback,
    _client_callback_data: *mut c_void,
    pc_sampling: *mut HsaPcSampling,
) -> Status {
    boundary(|| reject_create(None, agent, buffer_size, data_ready_callback, pc_sampling))
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn hsa_ven_amd_pcs_create_from_id(
    trace_id: u32,
    agent: HsaAgent,
    _method: u32,
    _units: u32,
    _interval: usize,
    _latency: usize,
    buffer_size: usize,
    data_ready_callback: PcSamplingDataReadyCallback,
    _client_callback_data: *mut c_void,
    pc_sampling: *mut HsaPcSampling,
) -> Status {
    boundary(|| {
        reject_create(
            Some(trace_id),
            agent,
            buffer_size,
            data_ready_callback,
            pc_sampling,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ven_amd_pcs_start(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| absent_session(pc_sampling))
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ven_amd_pcs_stop(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| absent_session(pc_sampling))
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ven_amd_pcs_flush(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| absent_session(pc_sampling))
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ven_amd_pcs_destroy(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| absent_session(pc_sampling))
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
