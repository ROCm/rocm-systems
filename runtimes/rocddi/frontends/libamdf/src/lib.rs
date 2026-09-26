//! Rust implementation of the pinned AMDF native C ABI.
//!
//! This frontend is early-access runtime software. The imported AMDF headers
//! define its C contract, but packaging, deployment, platform qualification,
//! and the underlying private rocddi integration may still change.
//!
//! Negotiation returns immutable tables without creating provider state. Native
//! connections belong to explicit instances, and each resource follows the
//! borrowing and output-publication contracts in the imported headers.

#![allow(clippy::wildcard_imports)]

use std::ffi::c_void;

mod generated;
mod instance;
mod kernel_queue;
mod memory;
mod queue;
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
        if minimum > AMDF_ABI_VERSION_3 || maximum < AMDF_ABI_VERSION_3 {
            return Err(VERSION);
        }
        // SAFETY: The checked output slot receives a library-lifetime table.
        unsafe { out_api.write(&raw const API) };
        Ok(())
    })
}

#[allow(unused_unsafe)]
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
        if extension != AMDF_EXTENSION_GPU {
            return Err(UNSUPPORTED);
        }
        if minimum > 1 || maximum < 1 {
            return Err(VERSION);
        }
        out.write((&raw const GPU_API).cast());
        Ok(())
    })
}

// Both imported v3 table sizes fit the u32 ABI extent field.
#[allow(clippy::cast_possible_truncation)]
static API: amdf_api_t = amdf_api_t {
    structure_size: size_of::<amdf_api_t>() as u32,
    abi_version: AMDF_ABI_VERSION_3,
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
};

#[allow(clippy::cast_possible_truncation)]
static GPU_API: amdf_gpu_api_t = amdf_gpu_api_t {
    structure_size: size_of::<amdf_gpu_api_t>() as u32,
    extension_version: 1,
    endpoint_query_info: Some(instance::gpu_endpoint_info),
    device_create: Some(instance::device_create),
    device_query_info: Some(instance::device_info),
    kernel_queue_create: Some(kernel_queue::create),
    kernel_queue_submit: Some(kernel_queue::submit),
    user_queue_create: Some(queue::create),
};

#[cfg(test)]
mod tests;
