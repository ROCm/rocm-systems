// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust implementation of the pinned AMDF native C ABI.
//!
//! This frontend is early-access runtime software. The imported AMDF headers
//! define its C contract, but packaging, deployment, platform qualification,
//! and the underlying private rocddi integration may still change.
//!
//! Negotiation returns immutable tables without creating provider state.
//! PROCESS instances share a KFD connection; resources still follow the
//! borrowing and output-publication contracts in the imported headers.

#![allow(clippy::wildcard_imports)]

use std::ffi::c_void;

mod generated;
mod gpu_clock;
mod gpu_event;
mod instance;
mod kernel_queue;
mod memory;
mod observation;
mod platform;
mod queue;
mod reservation;
mod support;

use generated::amdf::*;
use support::{INVALID, UNSUPPORTED, VERSION, boundary};

/// Negotiates an immutable AMDF API table without creating provider state.
///
/// # Safety
/// `out_api` is null or points to writable, properly aligned pointer storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn amdf_query_api(
    minimum: u32,
    maximum: u32,
    out_api: *mut *const amdf_api_t,
) -> u64 {
    boundary(|| {
        support::output_pointer(out_api)?;
        if minimum > maximum {
            return Err(INVALID);
        }
        if minimum > AMDF_ABI_VERSION_11 || maximum < AMDF_ABI_VERSION_11 {
            return Err(VERSION);
        }
        // SAFETY: The checked output slot receives a library-lifetime table.
        unsafe { out_api.write(&raw const API) };
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn query_extension(
    extension: u32,
    minimum: u32,
    maximum: u32,
    out: *mut *const c_void,
) -> u64 {
    crate::support::boundary(|| unsafe {
        support::output_pointer(out)?;
        if minimum > maximum {
            return Err(INVALID);
        }
        let table = match extension {
            AMDF_EXTENSION_GPU if minimum <= 3 && maximum >= 3 => (&raw const GPU_API_V3).cast(),
            AMDF_EXTENSION_GPU if minimum <= 2 && maximum >= 2 => (&raw const GPU_API_V2).cast(),
            AMDF_EXTENSION_GPU if minimum <= 1 && maximum >= 1 => (&raw const GPU_API_V1).cast(),
            AMDF_EXTENSION_OBSERVATION if minimum <= 1 && maximum >= 1 => {
                (&raw const OBSERVATION_API).cast()
            }
            AMDF_EXTENSION_GPU | AMDF_EXTENSION_OBSERVATION => return Err(VERSION),
            _ => return Err(UNSUPPORTED),
        };
        out.write(table);
        Ok(())
    })
}

// Imported table sizes fit the u32 ABI extent field.
#[allow(clippy::cast_possible_truncation)]
static API: amdf_api_t = amdf_api_t {
    structure_size: size_of::<amdf_api_t>() as u32,
    abi_version: AMDF_ABI_VERSION_11,
    instance_create: Some(instance::create),
    instance_destroy: Some(instance::destroy),
    endpoint_enumerate: Some(instance::enumerate),
    endpoint_open: Some(instance::open),
    endpoint_query_info: Some(instance::endpoint_info),
    endpoint_close: Some(instance::close),
    query_extension: Some(query_extension),
    endpoint_query_queue_family_info: Some(instance::queue_family_info),
    device_destroy: Some(instance::device_destroy),
    instance_enumerate_memory_scopes: Some(memory::instance_scopes),
    device_enumerate_memory_scopes: Some(memory::device_scopes),
    memory_scope_query_info: Some(memory::scope_info),
    memory_scope_query_device_profile: Some(memory::device_profile),
    memory_create: Some(memory::create),
    memory_import: Some(memory::import),
    memory_query_info: Some(memory::info),
    memory_query_access_info: Some(memory::access_info),
    memory_export: Some(memory::export),
    external_memory_release: Some(memory::external_release),
    memory_query_pair_info: Some(memory::pair_info),
    memory_map: Some(memory::map),
    host_mapping_query_info: Some(memory::mapping_info),
    host_mapping_cache_control: Some(memory::cache_control),
    host_mapping_destroy: Some(memory::unmap),
    memory_destroy: Some(memory::destroy),
    kernel_queue_query_info: Some(kernel_queue::info),
    kernel_queue_query_status: Some(kernel_queue::status),
    kernel_queue_wait: Some(kernel_queue::wait),
    kernel_queue_destroy: Some(kernel_queue::destroy),
    user_queue_query_info: Some(queue::info),
    user_queue_map: Some(queue::map),
    user_queue_mapping_query_info: Some(queue::mapping_info),
    user_queue_mapping_destroy: Some(queue::unmap),
    user_queue_query_status: Some(queue::status),
    user_queue_wait_consumed: Some(queue::wait),
    user_queue_destroy: Some(queue::destroy),
    memory_query_address: Some(memory::address),
    memory_scope_query_pair_info: Some(memory::scope_pair_info),
    kernel_queue_refresh_status: Some(kernel_queue::refresh_status),
    kernel_queue_request_notification: Some(kernel_queue::request_notification),
    device_query_memory_reservation_capabilities: Some(reservation::device_capabilities),
    instance_query_host_memory_reservation_capabilities: Some(reservation::host_capabilities),
    memory_reservation_create: Some(reservation::create),
    memory_reservation_map: Some(reservation::map),
    memory_reservation_query_access_info: Some(reservation::query_access_info),
    memory_reservation_update: Some(reservation::update),
    memory_reservation_detach: Some(reservation::detach),
    memory_reservation_destroy: Some(reservation::destroy),
    kernel_queue_update_memory: Some(reservation::queue_update),
};

#[allow(clippy::cast_possible_truncation)]
const fn gpu_table(version: u32) -> amdf_gpu_api_t {
    amdf_gpu_api_t {
        structure_size: if version == 1 {
            std::mem::offset_of!(amdf_gpu_api_t, device_query_queue_clock_info) as u32
        } else if version == 2 {
            std::mem::offset_of!(amdf_gpu_api_t, event_create) as u32
        } else {
            size_of::<amdf_gpu_api_t>() as u32
        },
        extension_version: version,
        endpoint_query_info: Some(instance::gpu_endpoint_info),
        device_create: Some(instance::device_create),
        device_query_info: Some(instance::device_info),
        kernel_queue_create: Some(kernel_queue::create),
        kernel_queue_submit: Some(kernel_queue::submit),
        user_queue_create: Some(queue::create),
        device_query_queue_clock_info: if version >= 2 {
            Some(gpu_clock::queue_clock_info)
        } else {
            None
        },
        device_query_clock_sample: if version >= 2 {
            Some(gpu_clock::clock_sample)
        } else {
            None
        },
        event_create: if version >= 3 {
            Some(gpu_event::create)
        } else {
            None
        },
        event_query_info: if version >= 3 {
            Some(gpu_event::info)
        } else {
            None
        },
        event_signal: if version >= 3 {
            Some(gpu_event::signal)
        } else {
            None
        },
        event_wait: if version >= 3 {
            Some(gpu_event::wait)
        } else {
            None
        },
        event_destroy: if version >= 3 {
            Some(gpu_event::destroy)
        } else {
            None
        },
    }
}

static GPU_API_V1: amdf_gpu_api_t = gpu_table(1);
static GPU_API_V2: amdf_gpu_api_t = gpu_table(2);
static GPU_API_V3: amdf_gpu_api_t = gpu_table(3);

#[allow(clippy::cast_possible_truncation)]
static OBSERVATION_API: amdf_observation_api_t = amdf_observation_api_t {
    structure_size: size_of::<amdf_observation_api_t>() as u32,
    extension_version: AMDF_OBSERVATION_EXTENSION_VERSION_1,
    query_host_clock_info: Some(observation::host_clock_info),
    query_host_clock: Some(observation::host_clock),
};

#[cfg(test)]
mod tests;
