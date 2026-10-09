// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA runtime ABI frontend backed directly by `rocddi`.
//!
//! This frontend is early-access runtime software, not a drop-in replacement
//! for the production `ROCr` HSA runtime. Its packaging, symbol-versioning,
//! coexistence, platform coverage, and hardware qualification remain subject
//! to change. Callers must select it explicitly.
//!
//! The crate owns HSA-visible process state and translates each C entry point
//! into the private rocddi mechanisms. Public handles are integer or pointer
//! encodings into runtime-owned objects; the individual modules document the
//! lifetime and synchronization rules for those objects.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_ptr_alignment,
    clippy::cast_sign_loss,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::semicolon_if_nothing_returned,
    clippy::too_many_lines,
    clippy::wildcard_imports
)]

use std::ffi::{CStr, c_char, c_void};

// Unit-test executables call the Rust entry points directly. They do not
// export the shared library's versioned ABI symbols.
#[cfg(all(target_os = "linux", not(test)))]
core::arch::global_asm!(include_str!("exports_linux.S"));

mod callback_arg;
mod ffi;
mod finalizer;
mod image_abi;
mod loader;
mod memory;
mod pc_sampling;
mod platform;
mod queue;
mod runtime;
mod signal;

use callback_arg::CallbackArg;
use ffi::*;
use runtime::{
    InFlightToken, LifecycleTransition, Runtime, boundary, initialized_mut, lock, map_error,
};

fn complete_deferred_shutdown(runtime: Runtime, generation: u64) {
    let mut transition = LifecycleTransition::stopping(generation);
    let status = runtime.stop();
    if let Ok(mut guard) = lock() {
        if guard.finish_shutdown(generation, status).is_ok() {
            transition.disarm();
        }
    }
    signal::set_system_frequency(0);
}

const HSA_RUNTIME_VERSION_MINOR: u16 = 21;
const AMD_AGENT_PRELOAD_SKIP_BLITS: u64 = 1 << 1;

unsafe fn write_value<T: Copy>(output: *mut c_void, value: T) -> Status {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: The HSA ABI requires output to point to writable storage for T.
    unsafe { output.cast::<T>().write(value) };
    SUCCESS
}

unsafe fn write_bytes(output: *mut c_void, bytes: &[u8], extent: usize) -> Status {
    if output.is_null() || bytes.len() > extent {
        return INVALID_ARGUMENT;
    }
    // SAFETY: The HSA ABI requires output to provide the fixed attribute extent.
    unsafe {
        std::ptr::write_bytes(output.cast::<u8>(), 0, extent);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.cast::<u8>(), bytes.len());
    }
    SUCCESS
}

fn isa_handle(gpu_index: usize, variant: u64) -> HsaIsa {
    HsaIsa {
        handle: ISA_BASE + gpu_index as u64 * ISA_COUNT_PER_GPU + variant,
    }
}

fn isa_name(gfx_major: u32, gfx_minor: u32, gfx_stepping: u32, variant: u64) -> Option<String> {
    match variant {
        0 => Some(format!(
            "amdgcn-amd-amdhsa--gfx{gfx_major}{gfx_minor}{gfx_stepping}"
        )),
        1 => Some(format!("amdgcn-amd-amdhsa--gfx{gfx_major}-generic")),
        _ => None,
    }
}

fn wavefront_handle(isa: HsaIsa) -> HsaWavefront {
    HsaWavefront {
        handle: WAVEFRONT_BASE + isa.handle.saturating_sub(ISA_BASE),
    }
}

const KERNEL_CLUSTER_MAX_DIM: HsaAmdDim3 = HsaAmdDim3 {
    x: i32::MAX as u64,
    y: 65_535,
    z: 65_535,
};
const CLUSTER_MAX_DIM: HsaAmdDim3 = HsaAmdDim3 { x: 1, y: 1, z: 1 };
const KERNEL_CLUSTER_MAX_SIZE: u64 =
    KERNEL_CLUSTER_MAX_DIM.x * KERNEL_CLUSTER_MAX_DIM.y * KERNEL_CLUSTER_MAX_DIM.z;

fn agent_uuid(gpu: Option<&rocddi::topology::GpuInfo>) -> String {
    match gpu {
        None => "CPU-XX".to_owned(),
        Some(gpu) => gpu.unique_id.map_or_else(
            || "GPU-XX".to_owned(),
            |unique_id| format!("GPU-{unique_id:016x}"),
        ),
    }
}

fn host_alloc_dmabuf_supported(gpu_count: usize) -> bool {
    gpu_count != 0
}

fn cpu_rejects_amd_agent_info(attribute: u32) -> bool {
    matches!(
        attribute,
        AMD_AGENT_INFO_MEMORY_WIDTH
            | AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY
            | AMD_AGENT_INFO_COOPERATIVE_QUEUES
            | AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT
            | AMD_AGENT_INFO_MEMORY_AVAIL
            | AMD_AGENT_INFO_PM4_EMULATION
            | AMD_AGENT_INFO_LUID
            | AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE
            | AMD_AGENT_INFO_CUID
            | AMD_AGENT_INFO_KERNEL_WG_MAX_SIZE
            | AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM
            | AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_SIZE
            | AMD_AGENT_INFO_CLUSTER_MAX_DIM
            | AMD_AGENT_INFO_CLUSTER_MAX_SIZE
            | AMD_AGENT_INFO_KERNEL_WG_MAX_DIM
            | AMD_AGENT_INFO_REQUEST_PERSISTING_L2_CACHE_SIZE
            | AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE
    )
}

fn nearest_cpu_agent(gpu_only: bool) -> HsaAgent {
    HsaAgent {
        handle: if gpu_only { CPU_AGENT } else { 0 },
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_init() -> Status {
    boundary(|| {
        let generation = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            match guard.begin_init() {
                Ok(None) => return SUCCESS,
                Ok(Some(generation)) => generation,
                Err(status) => return status,
            }
        };
        let mut transition = LifecycleTransition::starting(generation);
        let result = Runtime::create();
        let frequency = result
            .as_ref()
            .ok()
            .and_then(|runtime| runtime.gpus.first())
            .and_then(|gpu| gpu.device.gpu().ok())
            .and_then(|gpu| gpu.clock_counters().ok())
            .map_or(0, |counters| counters.system_frequency);
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let (status, transitioned) = match result {
            Ok(runtime) => match guard.publish_init(generation, runtime) {
                Ok(()) => {
                    signal::set_system_frequency(frequency);
                    (SUCCESS, true)
                }
                Err(status) => (status, false),
            },
            Err(failure) => match guard.cancel_init(generation, failure.cleanup_failed) {
                Ok(()) => (failure.status, true),
                Err(error) => (error, false),
            },
        };
        drop(guard);
        if transitioned {
            transition.disarm();
        }
        status
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_shut_down() -> Status {
    boundary(|| {
        if runtime::LogWriteScope::active() {
            return INVALID_RUNTIME_STATE;
        }
        let (runtime, generation) = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            match guard.begin_shutdown() {
                Ok(None) => return SUCCESS,
                Ok(Some(runtime)) => runtime,
                Err(status) => return status,
            }
        };
        let mut transition = LifecycleTransition::stopping(generation);
        runtime.request_stop();
        if runtime::CallbackScope::active() || runtime.running_on_worker() {
            match runtime::defer_cleanup(runtime, move |runtime| {
                complete_deferred_shutdown(runtime, generation);
            }) {
                Ok(()) => {
                    transition.disarm();
                    return SUCCESS;
                }
                Err(runtime) => {
                    // A callback may still be executing against signal storage.
                    if let Some(runtime) = runtime {
                        std::mem::forget(runtime);
                    }
                    return OUT_OF_RESOURCES;
                }
            }
        }
        let status = runtime.stop();
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(error) => return error,
        };
        let result = guard.finish_shutdown(generation, status);
        drop(guard);
        if result.is_ok() {
            transition.disarm();
        }
        signal::set_system_frequency(0);
        result.map_or_else(|error| error, |()| status)
    })
}

/// # Safety
/// Any non-null `value` must address aligned, writable storage for the type
/// selected by `attribute`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_system_get_info(attribute: u32, value: *mut c_void) -> Status {
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
        if matches!(
            attribute,
            SYSTEM_INFO_TIMESTAMP | SYSTEM_INFO_TIMESTAMP_FREQUENCY
        ) {
            let Some(gpu) = runtime.gpus.first() else {
                return ERROR;
            };
            let Some(_call) = runtime.inflight.enter() else {
                return OUT_OF_RESOURCES;
            };
            let device = gpu.device.clone();
            drop(guard);
            return match device.gpu().and_then(|gpu| gpu.clock_counters()) {
                // SAFETY: The caller supplied writable storage for this attribute.
                Ok(counters) if attribute == SYSTEM_INFO_TIMESTAMP => unsafe {
                    write_value(value, counters.system)
                },
                // SAFETY: The caller supplied writable storage for this attribute.
                Ok(counters) => unsafe { write_value(value, counters.system_frequency) },
                Err(error) => map_error(error),
            };
        }
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                SYSTEM_INFO_VERSION_MAJOR => write_value(value, 1_u16),
                SYSTEM_INFO_VERSION_MINOR => write_value(value, HSA_RUNTIME_VERSION_MINOR),
                SYSTEM_INFO_SIGNAL_MAX_WAIT => write_value(value, u64::MAX),
                SYSTEM_INFO_ENDIANNESS => write_value(value, 0_u32),
                SYSTEM_INFO_MACHINE_MODEL => write_value(value, 1_u32),
                SYSTEM_INFO_EXTENSIONS => write_value(value, extension_mask()),
                AMD_SYSTEM_INFO_SVM_SUPPORTED => write_value(value, true),
                AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT => write_value(value, false),
                AMD_SYSTEM_INFO_MWAITX_ENABLED => write_value(value, false),
                AMD_SYSTEM_INFO_DMABUF_SUPPORTED => write_value(value, true),
                AMD_SYSTEM_INFO_EXT_VERSION_MAJOR => write_value(value, 1_u16),
                AMD_SYSTEM_INFO_EXT_VERSION_MINOR => write_value(value, 33_u16),
                AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED => write_value(value, true),
                AMD_SYSTEM_INFO_XNACK_ENABLED | AMD_SYSTEM_INFO_FABRIC_HANDLES_SUPPORTED => {
                    write_value(value, false)
                }
                AMD_SYSTEM_INFO_HOST_ALLOC_DMABUF_SUPPORTED => {
                    write_value(value, host_alloc_dmabuf_supported(runtime.gpus.len()))
                }
                _ => INVALID_ARGUMENT,
            }
        }
    })
}

fn extension_name(extension: u16) -> Option<&'static [u8]> {
    match extension {
        EXTENSION_FINALIZER => Some(b"HSA_EXTENSION_FINALIZER\0"),
        EXTENSION_IMAGES => Some(b"HSA_EXTENSION_IMAGES\0"),
        EXTENSION_PERFORMANCE_COUNTERS => Some(b"HSA_EXTENSION_PERFORMANCE_COUNTERS\0"),
        EXTENSION_PROFILING_EVENTS => Some(b"HSA_EXTENSION_PROFILING_EVENTS\0"),
        EXTENSION_AMD_PROFILER => Some(b"HSA_EXTENSION_AMD_PROFILER\0"),
        EXTENSION_AMD_LOADER => Some(b"HSA_EXTENSION_AMD_LOADER\0"),
        EXTENSION_AMD_AQLPROFILE => Some(b"HSA_EXTENSION_AMD_AQLPROFILE\0"),
        _ => None,
    }
}

fn extension_mask() -> [u8; 128] {
    let mut extensions = [0_u8; 128];
    extensions[usize::from(EXTENSION_AMD_LOADER / 8)] |= 1 << (EXTENSION_AMD_LOADER % 8);
    extensions
}

fn supported_extension_minor(extension: u16, version_major: u16) -> Option<u16> {
    (version_major == 1 && extension == EXTENSION_AMD_LOADER).then_some(0)
}

fn legacy_extension_supported(extension: u16, version_major: u16, version_minor: u16) -> bool {
    matches!(version_major, 0 | 1) && version_minor == 0 && extension == EXTENSION_AMD_LOADER
}

fn legacy_agent_extension_supported(
    extension: u16,
    gpu: bool,
    version_major: u16,
    version_minor: u16,
) -> bool {
    gpu && legacy_extension_supported(extension, version_major, version_minor)
}

fn major_agent_extension_supported(extension: u16, gpu: bool, version_major: u16) -> bool {
    gpu && supported_extension_minor(extension, version_major).is_some()
}

fn valid_extension(extension: u16) -> bool {
    extension <= EXTENSION_PROFILING_EVENTS
        || (EXTENSION_AMD_PROFILER..=EXTENSION_AMD_PC_SAMPLING).contains(&extension)
}

/// # Safety
/// Any non-null `name` must address writable storage for one C string
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_extension_get_name(
    extension: u16,
    name: *mut *const c_char,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.is_none() {
            return NOT_INITIALIZED;
        }
        if name.is_null() {
            return INVALID_ARGUMENT;
        }
        let Some(value) = extension_name(extension) else {
            // SAFETY: The caller supplied writable output storage.
            unsafe { name.write(c"HSA_EXTENSION_INVALID".as_ptr()) };
            return INVALID_ARGUMENT;
        };
        // SAFETY: The caller supplied writable output storage and the selected
        // byte string has static lifetime and a trailing NUL.
        unsafe { name.write(value.as_ptr().cast()) };
        SUCCESS
    })
}

/// # Safety
/// Any non-null `result` must address writable `bool` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_system_extension_supported(
    extension: u16,
    version_major: u16,
    version_minor: u16,
    result: *mut bool,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        if !valid_extension(extension) || result.is_null() {
            return INVALID_ARGUMENT;
        }
        let supported = legacy_extension_supported(extension, version_major, version_minor);
        // SAFETY: The caller supplied writable output storage.
        unsafe { result.write(supported) };
        SUCCESS
    })
}

/// # Safety
/// Any non-null `version_minor` and `result` must address writable `u16` and
/// `bool` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_system_major_extension_supported(
    extension: u16,
    version_major: u16,
    version_minor: *mut u16,
    result: *mut bool,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        if !valid_extension(extension) || version_minor.is_null() || result.is_null() {
            return INVALID_ARGUMENT;
        }
        let minor = supported_extension_minor(extension, version_major);
        // ROCr leaves version_minor untouched when the extension is unsupported.
        unsafe {
            if let Some(minor) = minor {
                version_minor.write(minor);
            }
            result.write(minor.is_some());
        }
        SUCCESS
    })
}

/// # Safety
/// Any non-null `table` must address the writable table extent required by
/// the selected extension.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_system_get_extension_table(
    extension: u16,
    version_major: u16,
    version_minor: u16,
    table: *mut c_void,
) -> Status {
    let table_length = match (extension, version_major, version_minor) {
        (EXTENSION_IMAGES, 1, 0) => 10 * size_of::<usize>(),
        (EXTENSION_AMD_LOADER, 1, 0) => 3 * size_of::<usize>(),
        (EXTENSION_AMD_LOADER, 1, 1) => 5 * size_of::<usize>(),
        (EXTENSION_AMD_LOADER, 1, 2) => 6 * size_of::<usize>(),
        (EXTENSION_AMD_LOADER, 1, 3) => 7 * size_of::<usize>(),
        (EXTENSION_AMD_PC_SAMPLING, 1, 0) => 7 * size_of::<usize>(),
        _ => 0,
    };
    if table_length == 0 {
        return ERROR;
    }
    // SAFETY: This deprecated entry point selects the historical table length
    // and delegates the pointer contract to the major-version API.
    unsafe { hsa_system_get_major_extension_table(extension, version_major, table_length, table) }
}

/// # Safety
/// The callback must be callable with the C ABI; `data` must remain valid for
/// any access the callback performs during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_iterate_agents(callback: AgentCallback, data: *mut c_void) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let gpu_count = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            runtime.gpus.len()
        };
        // SAFETY: Traversal callbacks are synchronous and data remains live.
        let status = unsafe { callback(HsaAgent { handle: CPU_AGENT }, data) };
        if status != SUCCESS {
            return status;
        }
        for index in 0..gpu_count {
            // SAFETY: Traversal callbacks are synchronous and data remains live.
            let status = unsafe {
                callback(
                    HsaAgent {
                        handle: GPU_AGENT_BASE + index as u64,
                    },
                    data,
                )
            };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

/// # Safety
/// Any non-null `value` must address aligned, writable storage for the type
/// selected by `attribute`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_agent_get_info(
    agent: HsaAgent,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let (
            gpu_index,
            endpoint,
            name,
            product_name,
            host_compute_units,
            native_device,
            host_alloc_dmabuf,
            cache_sizes,
            asic_family_id,
            hdp_flush,
            persisting_l2_cache_size,
            timestamp_frequency_hz,
        ) = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            if agent.handle == CPU_AGENT {
                (
                    None,
                    None,
                    runtime.host_name.clone(),
                    runtime.host_name.clone(),
                    runtime.host_compute_units,
                    None,
                    host_alloc_dmabuf_supported(runtime.gpus.len()),
                    runtime.agent_cache_sizes(agent),
                    0,
                    [0; 2],
                    None,
                    1_000_000_000,
                )
            } else if let Some(index) = runtime.gpu_index(agent) {
                let native_device = if matches!(
                    attribute,
                    AMD_AGENT_INFO_CLOCK_COUNTERS
                        | AMD_AGENT_INFO_MEMORY_AVAIL
                        | AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE
                ) {
                    let Some(token) = runtime.inflight.enter() else {
                        return OUT_OF_RESOURCES;
                    };
                    Some((runtime.gpus[index].device.clone(), token))
                } else {
                    None
                };
                (
                    Some(index),
                    Some(runtime.gpus[index].endpoint.clone()),
                    runtime.gpus[index].name.clone(),
                    runtime.gpus[index].product_name.clone(),
                    runtime.host_compute_units,
                    native_device,
                    host_alloc_dmabuf_supported(runtime.gpus.len()),
                    runtime.agent_cache_sizes(agent),
                    runtime.gpus[index].asic_family_id,
                    runtime.gpus[index].hdp_flush,
                    Some(runtime.gpus[index].persisting_l2_cache_size.clone()),
                    runtime.gpus[index].timestamp_frequency_hz,
                )
            } else {
                return INVALID_AGENT;
            }
        };
        let clock_counters = if attribute == AMD_AGENT_INFO_CLOCK_COUNTERS {
            native_device
                .as_ref()
                .map(|(device, _)| device.gpu().and_then(|gpu| gpu.clock_counters()))
        } else {
            None
        };
        let available_memory = if attribute == AMD_AGENT_INFO_MEMORY_AVAIL {
            native_device
                .as_ref()
                .map(|(device, _)| device.available_memory())
        } else {
            None
        };
        let expert_scheduling = if attribute == AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE {
            native_device.as_ref().map(|(device, _)| {
                device
                    .gpu()
                    .and_then(|gpu| gpu.supports_expert_scheduling())
            })
        } else {
            None
        };
        let gpu = endpoint.as_ref().and_then(|endpoint| endpoint.gpu());
        let gpu_only = gpu.is_some();
        if !gpu_only && cpu_rejects_amd_agent_info(attribute) {
            return INVALID_ARGUMENT;
        }
        let node = platform::node_id(endpoint.as_ref());
        let pci = endpoint.as_ref().and_then(|endpoint| endpoint.pci);
        let bdf = pci.map_or(0, |pci| (pci.bus << 8) | (pci.device << 3) | pci.function);
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                AGENT_INFO_NAME => write_bytes(value, name.as_bytes(), 64),
                AGENT_INFO_VENDOR_NAME => {
                    write_bytes(value, if gpu_only { b"AMD" } else { b"CPU" }, 64)
                }
                AGENT_INFO_FEATURE => write_value(
                    value,
                    if gpu_only {
                        AGENT_FEATURE_KERNEL_DISPATCH
                    } else {
                        0_u32
                    },
                ),
                AGENT_INFO_MACHINE_MODEL => write_value(value, 1_u32),
                AGENT_INFO_PROFILE => {
                    write_value(value, if gpu_only { PROFILE_BASE } else { PROFILE_FULL })
                }
                AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE => write_value(value, 2_u32),
                AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES => {
                    write_value(value, (1_u32 << 1) | (1_u32 << 2))
                }
                AGENT_INFO_FAST_F16_OPERATION => write_value(value, gpu_only),
                AGENT_INFO_WAVEFRONT_SIZE => {
                    write_value(value, gpu.map_or(0, |info| info.wavefront_size))
                }
                AGENT_INFO_WORKGROUP_MAX_DIM => write_value(
                    value,
                    if gpu_only {
                        [1024_u16, 1024, 1024]
                    } else {
                        [0_u16; 3]
                    },
                ),
                AGENT_INFO_WORKGROUP_MAX_SIZE => {
                    write_value(value, if gpu_only { 1024 } else { 0 })
                }
                AGENT_INFO_GRID_MAX_DIM => write_value(
                    value,
                    if gpu_only {
                        HsaDim3 {
                            x: u32::MAX,
                            y: u16::MAX.into(),
                            z: u16::MAX.into(),
                        }
                    } else {
                        HsaDim3 { x: 0, y: 0, z: 0 }
                    },
                ),
                AGENT_INFO_GRID_MAX_SIZE => write_value(value, if gpu_only { u32::MAX } else { 0 }),
                AGENT_INFO_FBARRIER_MAX_SIZE => {
                    write_value(value, if gpu_only { 32_u32 } else { 0 })
                }
                AGENT_INFO_QUEUES_MAX => write_value(value, if gpu_only { 128_u32 } else { 0 }),
                AGENT_INFO_QUEUE_MIN_SIZE => write_value(value, if gpu_only { 64_u32 } else { 0 }),
                AGENT_INFO_QUEUE_MAX_SIZE => {
                    write_value(value, if gpu_only { 131_072_u32 } else { 0 })
                }
                AGENT_INFO_QUEUE_TYPE => write_value(value, QUEUE_TYPE_MULTI),
                AGENT_INFO_NODE => write_value(value, node),
                AGENT_INFO_DEVICE => {
                    write_value(value, if gpu_only { DEVICE_GPU } else { DEVICE_CPU })
                }
                AGENT_INFO_CACHE_SIZE => write_value(value, cache_sizes),
                AGENT_INFO_ISA => write_value(
                    value,
                    gpu_index.map_or(HsaIsa { handle: 0 }, |index| isa_handle(index, 0)),
                ),
                AGENT_INFO_EXTENSIONS => {
                    write_value(value, if gpu_only { extension_mask() } else { [0; 128] })
                }
                AGENT_INFO_VERSION_MAJOR | AGENT_INFO_VERSION_MINOR => write_value(value, 1_u16),
                AMD_AGENT_INFO_CHIP_ID => write_value(value, pci.map_or(0, |info| info.device_id)),
                AMD_AGENT_INFO_CACHELINE_SIZE => write_value(
                    value,
                    endpoint.as_ref().map_or(64, |endpoint| {
                        endpoint
                            .caches()
                            .iter()
                            .find(|cache| cache.level() == 2 && cache.line_size_bytes() != 0)
                            .map_or(256, rocddi::topology::CacheInfo::line_size_bytes)
                    }),
                ),
                AMD_AGENT_INFO_COMPUTE_UNIT_COUNT
                | AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT => write_value(
                    value,
                    gpu.map_or(host_compute_units, |info| info.compute_unit_count),
                ),
                AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY => write_value(
                    value,
                    gpu.map_or(5476, |info| info.maximum_engine_clock_mhz),
                ),
                AMD_AGENT_INFO_DRIVER_NODE_ID => write_value(value, node),
                AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS => write_value(
                    value,
                    gpu.map_or(1, |info| info.maximum_address_watch_point_count),
                ),
                AMD_AGENT_INFO_BDFID => write_value(value, bdf),
                AMD_AGENT_INFO_MEMORY_WIDTH => {
                    write_value(value, gpu.map_or(0, |info| info.memory_bus_width_bits))
                }
                AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY => {
                    write_value(value, gpu.map_or(0, |info| info.maximum_memory_clock_mhz))
                }
                AMD_AGENT_INFO_PRODUCT_NAME => write_bytes(value, product_name.as_bytes(), 64),
                AMD_AGENT_INFO_MAX_WAVES_PER_CU => write_value(
                    value,
                    gpu.map_or(0, |info| info.maximum_wave_count_per_compute_unit),
                ),
                AMD_AGENT_INFO_NUM_SIMDS_PER_CU => write_value(
                    value,
                    gpu.map_or(0, |info| info.simd_count_per_compute_unit),
                ),
                AMD_AGENT_INFO_NUM_SHADER_ENGINES => write_value(
                    value,
                    gpu.map_or(0, |info| {
                        info.shader_engine_count_per_xcc
                            .saturating_mul(info.xcc_count)
                    }),
                ),
                AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE => write_value(
                    value,
                    gpu.map_or(0, |info| info.shader_array_count_per_engine),
                ),
                AMD_AGENT_INFO_HDP_FLUSH => write_value(value, hdp_flush),
                AMD_AGENT_INFO_DOMAIN => write_value(value, pci.map_or(0, |info| info.domain)),
                AMD_AGENT_INFO_COOPERATIVE_QUEUES => {
                    write_value(value, gpu.is_some_and(|info| info.gws_count != 0))
                }
                AMD_AGENT_INFO_UUID => {
                    let uuid = agent_uuid(gpu);
                    write_bytes(value, uuid.as_bytes(), 21)
                }
                AMD_AGENT_INFO_ASIC_REVISION => {
                    write_value(value, gpu.map_or(0, |info| info.asic_revision))
                }
                AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS => {
                    write_value(value, gpu.is_none_or(|info| info.coherent_host_access))
                }
                AMD_AGENT_INFO_MEMORY_AVAIL => match available_memory {
                    Some(Ok(bytes)) => write_value(value, bytes),
                    Some(Err(_)) | None => INVALID_ARGUMENT,
                },
                AMD_AGENT_INFO_TIMESTAMP_FREQUENCY => write_value(value, timestamp_frequency_hz),
                AMD_AGENT_INFO_ASIC_FAMILY_ID => write_value(value, asic_family_id),
                AMD_AGENT_INFO_UCODE_VERSION => write_value(
                    value,
                    gpu.map_or(0, |info| info.packet_processor_firmware_version),
                ),
                AMD_AGENT_INFO_SDMA_UCODE_VERSION => {
                    write_value(value, gpu.map_or(0, |info| info.sdma_firmware_version))
                }
                AMD_AGENT_INFO_NUM_SDMA_ENG => {
                    write_value(value, gpu.map_or(0, |info| info.queues.sdma_engine_count))
                }
                AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG => {
                    write_value(value, gpu.map_or(0, |info| info.sdma_xgmi_engine_count))
                }
                AMD_AGENT_INFO_IOMMU_SUPPORT => write_value(
                    value,
                    u32::from(gpu.is_some_and(|info| info.iommu_v2_supported)),
                ),
                AMD_AGENT_INFO_DRIVER_UID => {
                    write_value(value, platform::driver_uid(endpoint.as_ref()))
                }
                AMD_AGENT_INFO_MAX_DATA_PREFETCH_REGIONS => write_value(value, 0_u32),
                AMD_AGENT_INFO_NUM_XCC => write_value(value, gpu.map_or(0, |info| info.xcc_count)),
                AMD_AGENT_INFO_NEAREST_CPU => write_value(value, nearest_cpu_agent(gpu_only)),
                AMD_AGENT_INFO_MEMORY_PROPERTIES | AMD_AGENT_INFO_AQL_EXTENSIONS => {
                    write_value(value, [0_u8; 8])
                }
                AMD_AGENT_INFO_SCRATCH_LIMIT_MAX => write_value(
                    value,
                    gpu.map_or(0, rocddi::topology::GpuInfo::maximum_scratch_aperture_bytes),
                ),
                AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT => write_value(value, 0_u64),
                AMD_AGENT_INFO_CLOCK_COUNTERS => {
                    if gpu.is_some() {
                        let Some(clock_counters) = clock_counters else {
                            return ERROR;
                        };
                        match clock_counters {
                            Ok(counters) => write_value(
                                value,
                                HsaAmdClockCounters {
                                    gpu_clock_counter: counters.gpu,
                                    cpu_clock_counter: counters.host,
                                    system_clock_counter: counters.system,
                                    system_clock_frequency: counters.system_frequency,
                                },
                            ),
                            Err(error) => map_error(error),
                        }
                    } else {
                        write_value(
                            value,
                            HsaAmdClockCounters {
                                gpu_clock_counter: 0,
                                cpu_clock_counter: 0,
                                system_clock_counter: 0,
                                system_clock_frequency: 0,
                            },
                        )
                    }
                }
                AMD_AGENT_INFO_PM4_EMULATION => write_value(value, false),
                AMD_AGENT_INFO_LUID if gpu_only => write_value(value, HsaLuid::default()),
                AMD_AGENT_INFO_HOST_ALLOC_DMABUF_SUPPORTED => write_value(value, host_alloc_dmabuf),
                AMD_AGENT_INFO_REQUEST_PERSISTING_L2_CACHE_SIZE => {
                    let Some(state) = persisting_l2_cache_size else {
                        return INVALID_ARGUMENT;
                    };
                    let requested = match state.lock() {
                        Ok(requested) => *requested,
                        Err(_) => return ERROR,
                    };
                    write_value(value, requested)
                }
                AMD_AGENT_INFO_MAX_PERSISTING_L2_CACHE_SIZE => write_value(
                    value,
                    gpu.map_or(0, |info| info.persisting_l2_cache_size_max as usize),
                ),
                AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE => match expert_scheduling {
                    Some(Ok(supported)) => write_value(value, supported),
                    Some(Err(error)) => map_error(error),
                    None => INVALID_ARGUMENT,
                },
                AMD_AGENT_INFO_CUID => write_value(value, [0_u8; 16]),
                AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM | AMD_AGENT_INFO_KERNEL_WG_MAX_DIM
                    if gpu_only =>
                {
                    write_value(value, KERNEL_CLUSTER_MAX_DIM)
                }
                AMD_AGENT_INFO_KERNEL_WG_MAX_SIZE | AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_SIZE
                    if gpu_only =>
                {
                    write_value(value, KERNEL_CLUSTER_MAX_SIZE)
                }
                AMD_AGENT_INFO_CLUSTER_MAX_DIM if gpu_only => write_value(value, CLUSTER_MAX_DIM),
                AMD_AGENT_INFO_CLUSTER_MAX_SIZE if gpu_only => {
                    write_value(value, CLUSTER_MAX_DIM.x)
                }
                EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS => write_value(value, 0_u32),
                EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS => write_value(value, [0_u32; 2]),
                EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS => write_value(value, [0_u32; 3]),
                EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS
                | EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES
                | EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES
                | EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS => write_value(value, 0_u32),
                EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT => write_value(value, 0_usize),
                EXT_AGENT_INFO_IMAGE_SUPPORT => write_value(value, false),
                _ => INVALID_ARGUMENT,
            }
        }
    })
}

/// # Safety
/// A non-null `value` must point to readable, aligned `size_t` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_agent_set_attribute(
    agent: HsaAgent,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        let (device, maximum, state, _inflight) = {
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
            let Some(index) = runtime.gpu_index(agent) else {
                return INVALID_AGENT;
            };
            if attribute != AMD_AGENT_ATTRIBUTE_REQUEST_PERSISTING_L2_CACHE_SIZE {
                return INVALID_ARGUMENT;
            }
            let Some(token) = runtime.inflight.enter() else {
                return OUT_OF_RESOURCES;
            };
            let gpu = &runtime.gpus[index];
            (
                gpu.device.clone(),
                gpu.info.persisting_l2_cache_size_max,
                gpu.persisting_l2_cache_size.clone(),
                token,
            )
        };
        // SAFETY: The HSA caller supplies aligned, readable size_t storage.
        let requested = unsafe { value.cast::<usize>().read() };
        let Ok(native_size) = u32::try_from(requested) else {
            return INVALID_ARGUMENT;
        };
        if native_size > maximum {
            return INVALID_ARGUMENT;
        }
        let mut current = match state.lock() {
            Ok(current) => current,
            Err(_) => return ERROR,
        };
        let gpu = match device.gpu() {
            Ok(gpu) => gpu,
            Err(error) => return map_error(error),
        };
        if let Err(error) = gpu.set_persisting_l2_cache_size(native_size) {
            return if error.kind() == rocddi::ErrorKind::Unsupported {
                NOT_SUPPORTED
            } else {
                map_error(error)
            };
        }
        *current = requested;
        SUCCESS
    })
}

/// # Safety
/// Any non-null `mask` must address writable `u16` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_agent_get_exception_policies(
    agent: HsaAgent,
    profile: u32,
    mask: *mut u16,
) -> Status {
    boundary(|| {
        if mask.is_null() || !matches!(profile, PROFILE_BASE | PROFILE_FULL) {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        // SAFETY: The caller supplied writable output storage.
        unsafe { mask.write(0) };
        SUCCESS
    })
}

/// # Safety
/// Any non-null `result` must address writable `bool` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_agent_extension_supported(
    extension: u16,
    agent: HsaAgent,
    version_major: u16,
    version_minor: u16,
    result: *mut bool,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !valid_extension(extension) || result.is_null() {
            return INVALID_ARGUMENT;
        }
        // ROCr clears the result before validating the agent handle.
        unsafe { result.write(false) };
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        let gpu_index = runtime.gpu_index(agent);
        let supported = legacy_agent_extension_supported(
            extension,
            gpu_index.is_some(),
            version_major,
            version_minor,
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { result.write(supported) };
        SUCCESS
    })
}

/// # Safety
/// Any non-null `version_minor` and `result` must address writable `u16` and
/// `bool` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_agent_major_extension_supported(
    extension: u16,
    agent: HsaAgent,
    version_major: u16,
    version_minor: *mut u16,
    result: *mut bool,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !valid_extension(extension) || result.is_null() {
            return INVALID_ARGUMENT;
        }
        // ROCr clears the result before validating the agent handle.
        unsafe { result.write(false) };
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        let gpu_index = runtime.gpu_index(agent);
        let supported =
            major_agent_extension_supported(extension, gpu_index.is_some(), version_major);
        if supported && version_minor.is_null() {
            return INVALID_ARGUMENT;
        }
        // ROCr leaves version_minor untouched when the extension is unsupported.
        unsafe {
            if supported {
                version_minor.write(0);
            }
            result.write(supported);
        }
        SUCCESS
    })
}

/// # Safety
/// The callback must be callable with the C ABI; `data` must remain valid for
/// any access the callback performs during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_agent_iterate_isas(
    agent: HsaAgent,
    callback: IsaCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let index = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(index) = runtime.gpu_index(agent) else {
                return if agent.handle == CPU_AGENT {
                    SUCCESS
                } else {
                    INVALID_AGENT
                };
            };
            index
        };
        for variant in 0..ISA_COUNT_PER_GPU {
            // SAFETY: The callback and data remain valid for this synchronous call.
            let status = unsafe { callback(isa_handle(index, variant), data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

/// # Safety
/// Any non-null `name` must point to a readable NUL-terminated C string; any
/// non-null `isa` must address writable `HsaIsa` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_isa_from_name(name: *const c_char, isa: *mut HsaIsa) -> Status {
    boundary(|| {
        if name.is_null() || isa.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The public contract requires a NUL-terminated input string.
        let requested = unsafe { CStr::from_ptr(name) }.to_bytes();
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        for (index, gpu) in runtime.gpus.iter().enumerate() {
            for variant in 0..ISA_COUNT_PER_GPU {
                let Some(candidate) = isa_name(
                    gpu.info.gfx_major,
                    gpu.info.gfx_minor,
                    gpu.info.gfx_stepping,
                    variant,
                ) else {
                    continue;
                };
                if requested == candidate.as_bytes() {
                    // SAFETY: The caller supplied writable output storage.
                    unsafe { isa.write(isa_handle(index, variant)) };
                    return SUCCESS;
                }
            }
        }
        INVALID_ISA_NAME
    })
}

/// # Safety
/// Any non-null `value` must address aligned, writable storage for the type
/// selected by `attribute`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_isa_get_info_alt(
    isa: HsaIsa,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let (info, variant) = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some((index, variant)) = runtime.isa_parts(isa) else {
                return INVALID_ISA;
            };
            (runtime.gpus[index].info, variant)
        };
        let Some(name) = isa_name(info.gfx_major, info.gfx_minor, info.gfx_stepping, variant)
        else {
            return INVALID_ISA;
        };
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                ISA_INFO_NAME_LENGTH => {
                    write_value(value, u32::try_from(name.len() + 1).unwrap_or(u32::MAX))
                }
                ISA_INFO_NAME => {
                    std::ptr::copy_nonoverlapping(name.as_ptr(), value.cast::<u8>(), name.len());
                    value.cast::<u8>().add(name.len()).write(0);
                    SUCCESS
                }
                ISA_INFO_CALL_CONVENTION_COUNT => write_value(value, 1_u32),
                ISA_INFO_MACHINE_MODELS => write_value(value, [false, true]),
                ISA_INFO_PROFILES => write_value(value, [true, false]),
                ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES => write_value(value, [false, false, true]),
                ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES => {
                    write_value(value, [false, false, true])
                }
                ISA_INFO_FAST_F16_OPERATION => write_value(value, true),
                ISA_INFO_WORKGROUP_MAX_DIM => write_value(value, [1024_u16, 1024, 1024]),
                ISA_INFO_WORKGROUP_MAX_SIZE => write_value(value, 1024_u32),
                ISA_INFO_GRID_MAX_DIM => write_value(
                    value,
                    HsaDim3 {
                        x: i32::MAX as u32,
                        y: u16::MAX.into(),
                        z: u16::MAX.into(),
                    },
                ),
                ISA_INFO_GRID_MAX_SIZE => write_value(value, u64::MAX),
                ISA_INFO_FBARRIER_MAX_SIZE => write_value(value, 32_u32),
                _ => INVALID_ARGUMENT,
            }
        }
    })
}

/// # Safety
/// Any non-null `value` must address aligned, writable storage for the type
/// selected by `attribute` and `index`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_isa_get_info(
    isa: HsaIsa,
    attribute: u32,
    index: u32,
    value: *mut c_void,
) -> Status {
    if index != 0 {
        return INVALID_INDEX;
    }
    if !matches!(attribute, 3 | 4) {
        // SAFETY: This deprecated entry point shares the public output contract
        // of hsa_isa_get_info_alt for non-call-convention attributes.
        return unsafe { hsa_isa_get_info_alt(isa, attribute, value) };
    }
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.isa_parts(isa).is_none() {
            return INVALID_ISA;
        }
        // SAFETY: The caller supplied writable output storage.
        unsafe {
            match attribute {
                3 => write_value(value, 64_u32),
                4 => write_value(value, 40_u32),
                _ => INVALID_ARGUMENT,
            }
        }
    })
}

/// # Safety
/// Any non-null `mask` must address writable `u16` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_isa_get_exception_policies(
    isa: HsaIsa,
    profile: u32,
    mask: *mut u16,
) -> Status {
    boundary(|| {
        if mask.is_null() || !matches!(profile, PROFILE_BASE | PROFILE_FULL) {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.isa_parts(isa).is_none() {
            return INVALID_ISA;
        }
        // SAFETY: The caller supplied writable output storage.
        unsafe { mask.write(0) };
        SUCCESS
    })
}

/// # Safety
/// Any non-null `round_method` must address writable `u32` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_isa_get_round_method(
    isa: HsaIsa,
    fp_type: u32,
    flush_mode: u32,
    round_method: *mut u32,
) -> Status {
    boundary(|| {
        if round_method.is_null() || !matches!(fp_type, 1 | 2 | 4) || !matches!(flush_mode, 1 | 2) {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.isa_parts(isa).is_none() {
            return INVALID_ISA;
        }
        // GFX1201 implements MAD with one rounding step for every supported
        // floating-point width and flush mode.
        unsafe { round_method.write(1) };
        SUCCESS
    })
}

/// # Safety
/// The callback must be callable with the C ABI; `data` must remain valid for
/// any access the callback performs during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_isa_iterate_wavefronts(
    isa: HsaIsa,
    callback: WavefrontCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.isa_parts(isa).is_none() {
            return INVALID_ISA;
        }
        drop(guard);
        // SAFETY: The callback and data remain valid for this synchronous call.
        unsafe { callback(wavefront_handle(isa), data) }
    })
}

/// # Safety
/// Any non-null `value` must address aligned, writable storage for the type
/// selected by `attribute`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_wavefront_get_info(
    wavefront: HsaWavefront,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let wavefront_size = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(isa) = runtime.wavefront_isa(wavefront) else {
                return INVALID_WAVEFRONT;
            };
            let Some(index) = runtime.isa_index(isa) else {
                return INVALID_WAVEFRONT;
            };
            runtime.gpus[index].info.wavefront_size
        };
        if attribute != 0 {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied writable output storage.
        unsafe { write_value(value, wavefront_size) }
    })
}

/// # Safety
/// Any non-null `result` must address writable `bool` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_isa_compatible(
    code_object_isa: HsaIsa,
    agent_isa: HsaIsa,
    result: *mut bool,
) -> Status {
    boundary(|| {
        if result.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if runtime.isa_parts(code_object_isa).is_none() || runtime.isa_parts(agent_isa).is_none() {
            return INVALID_ISA;
        }
        // SAFETY: The caller supplied writable output storage.
        unsafe { result.write(code_object_isa == agent_isa) };
        SUCCESS
    })
}

/// # Safety
/// Any non-null `table` must address at least `table_length` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_system_get_major_extension_table(
    extension: u16,
    version_major: u16,
    table_length: usize,
    table: *mut c_void,
) -> Status {
    boundary(|| {
        if table.is_null() || table_length == 0 {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        if matches!(
            extension,
            EXTENSION_IMAGES
                | EXTENSION_AMD_PROFILER
                | EXTENSION_AMD_AQLPROFILE
                | EXTENSION_AMD_PC_SAMPLING
        ) {
            return NOT_SUPPORTED;
        }
        match (extension, version_major) {
            (EXTENSION_AMD_LOADER, 1) => {
                let functions = loader::loader_extension_table();
                let count = table_length.min(size_of_val(&functions));
                // SAFETY: The caller promises table_length writable bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (&raw const functions).cast::<u8>(),
                        table.cast::<u8>(),
                        count,
                    );
                }
            }
            _ => return ERROR,
        }
        SUCCESS
    })
}

/// # Safety
/// Any non-null `output` must address writable storage for one C string
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_status_string(status: Status, output: *mut *const c_char) -> Status {
    boundary(|| {
        if output.is_null() {
            return INVALID_ARGUMENT;
        }
        let value = match status {
            SUCCESS => STATUS_SUCCESS,
            INFO_BREAK => STATUS_INFO_BREAK,
            ERROR => STATUS_ERROR,
            INVALID_ARGUMENT => STATUS_INVALID_ARGUMENT,
            INVALID_QUEUE_CREATION => STATUS_INVALID_QUEUE_CREATION,
            INVALID_ALLOCATION => STATUS_INVALID_ALLOCATION,
            INVALID_AGENT => STATUS_INVALID_AGENT,
            INVALID_REGION => STATUS_INVALID_REGION,
            INVALID_SIGNAL => STATUS_INVALID_SIGNAL,
            INVALID_QUEUE => STATUS_INVALID_QUEUE,
            OUT_OF_RESOURCES => STATUS_OUT_OF_RESOURCES,
            INVALID_PACKET_FORMAT => STATUS_INVALID_PACKET_FORMAT,
            RESOURCE_FREE => STATUS_RESOURCE_FREE,
            NOT_INITIALIZED => STATUS_NOT_INITIALIZED,
            REFCOUNT_OVERFLOW => STATUS_REFCOUNT_OVERFLOW,
            INCOMPATIBLE_ARGUMENTS => STATUS_INCOMPATIBLE_ARGUMENTS,
            INVALID_INDEX => STATUS_INVALID_INDEX,
            INVALID_ISA => STATUS_INVALID_ISA,
            INVALID_CODE_OBJECT => STATUS_INVALID_CODE_OBJECT,
            INVALID_EXECUTABLE => STATUS_INVALID_EXECUTABLE,
            FROZEN_EXECUTABLE => STATUS_FROZEN_EXECUTABLE,
            INVALID_SYMBOL_NAME => STATUS_INVALID_SYMBOL_NAME,
            VARIABLE_ALREADY_DEFINED => STATUS_VARIABLE_ALREADY_DEFINED,
            VARIABLE_UNDEFINED => STATUS_VARIABLE_UNDEFINED,
            EXCEPTION => STATUS_EXCEPTION,
            INVALID_ISA_NAME => STATUS_INVALID_ISA_NAME,
            INVALID_CODE_SYMBOL => STATUS_INVALID_CODE_SYMBOL,
            INVALID_EXECUTABLE_SYMBOL => STATUS_INVALID_EXECUTABLE_SYMBOL,
            INVALID_FILE => STATUS_INVALID_FILE,
            INVALID_CODE_OBJECT_READER => STATUS_INVALID_CODE_OBJECT_READER,
            INVALID_CACHE => STATUS_INVALID_CACHE,
            INVALID_WAVEFRONT => STATUS_INVALID_WAVEFRONT,
            INVALID_SIGNAL_GROUP => STATUS_INVALID_SIGNAL_GROUP,
            INVALID_RUNTIME_STATE => STATUS_INVALID_RUNTIME_STATE,
            FATAL => STATUS_FATAL,
            INVALID_MEMORY_POOL => STATUS_INVALID_MEMORY_POOL,
            MEMORY_APERTURE_VIOLATION => STATUS_MEMORY_APERTURE_VIOLATION,
            ILLEGAL_INSTRUCTION => STATUS_ILLEGAL_INSTRUCTION,
            MEMORY_FAULT => STATUS_MEMORY_FAULT,
            CU_MASK_REDUCED => STATUS_CU_MASK_REDUCED,
            OUT_OF_REGISTERS => STATUS_OUT_OF_REGISTERS,
            RESOURCE_BUSY => STATUS_RESOURCE_BUSY,
            NOT_SUPPORTED => STATUS_NOT_SUPPORTED,
            XNACK_DISABLED => STATUS_XNACK_DISABLED,
            INVALID_DISPATCH_PARAMETERS => STATUS_INVALID_DISPATCH_PARAMETERS,
            RESOURCE_NOT_READY => STATUS_RESOURCE_NOT_READY,
            IMAGE_FORMAT_UNSUPPORTED => STATUS_IMAGE_FORMAT_UNSUPPORTED,
            IMAGE_SIZE_UNSUPPORTED => STATUS_IMAGE_SIZE_UNSUPPORTED,
            IMAGE_PITCH_UNSUPPORTED => STATUS_IMAGE_PITCH_UNSUPPORTED,
            SAMPLER_DESCRIPTOR_UNSUPPORTED => STATUS_SAMPLER_DESCRIPTOR_UNSUPPORTED,
            _ => return INVALID_ARGUMENT,
        };
        // SAFETY: The caller supplied writable pointer storage.
        unsafe { output.write(ffi::c_string(value)) };
        SUCCESS
    })
}

/// # Safety
/// `flags` must address eight readable bytes for the duration of this call.
/// A non-null `file` must be an open C `FILE*` kept live while its logging
/// flags are enabled. Replacing the stream, disabling logging, or final HSA
/// shutdown waits for writes already using it to finish. A nonfinal
/// `hsa_shut_down` does not release the stream. Calls made recursively from a
/// custom stream callback are rejected. Null selects stderr.
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
        if runtime.system_event_handlers.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let status = runtime.ensure_system_event_worker();
        if status != SUCCESS {
            return status;
        }
        runtime
            .system_event_handlers
            // SAFETY: The C registration contract keeps callback data live and
            // synchronizes it until the runtime stops delivering events.
            .push((callback, unsafe { CallbackArg::new(data) }));
        SUCCESS
    })
}

/// # Safety
/// Any non-null `kind` must address writable `u32` storage.
#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_agent_preload(agent: HsaAgent, flags: u64) -> Status {
    boundary(|| {
        let (device, _call) = {
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
            (runtime.gpus[index].device.clone(), token)
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_profiling_async_copy_enable(enable: bool) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_mut() else {
            return NOT_INITIALIZED;
        };
        runtime.async_copy_profiling = if enable {
            runtime::AsyncCopyProfiling::Enabled
        } else {
            runtime::AsyncCopyProfiling::Disabled
        };
        SUCCESS
    })
}

/// # Safety
/// Any non-null `system_tick` must address writable `u64` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_profiling_convert_tick_to_system_domain(
    agent: HsaAgent,
    agent_tick: u64,
    system_tick: *mut u64,
) -> Status {
    boundary(|| {
        let (device, _call) = {
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
            (runtime.gpus[index].device.clone(), token)
        };
        let counters = match device.gpu().and_then(|gpu| gpu.clock_counters()) {
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

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_spm_acquire(agent: HsaAgent) -> Status {
    boundary(|| {
        let (device, _call) = match retained_gpu_device(agent) {
            Ok(retained) => retained,
            Err(status) => return status,
        };
        device
            .gpu()
            .and_then(|gpu| gpu.spm_acquire())
            .map_or_else(map_error, |()| SUCCESS)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_spm_release(agent: HsaAgent) -> Status {
    boundary(|| {
        let (device, _call) = match retained_gpu_device(agent) {
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
fn retained_gpu_device(agent: HsaAgent) -> Result<(rocddi::device::Device, InFlightToken), Status> {
    let guard = lock()?;
    let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
    let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
    let token = runtime.inflight.enter().ok_or(OUT_OF_RESOURCES)?;
    Ok((runtime.gpus[index].device.clone(), token))
}

/// # Safety
/// Scalar output pointers must be valid and writable, and `timeout` readable.
/// A non-null `destination` must hold `size` writable bytes accessible to
/// KFD. Retain both the old and new destinations through successful
/// replacement or unset, or conclusive teardown; a failed call may leave
/// either reachable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_spm_set_dest_buffer(
    agent: HsaAgent,
    size: usize,
    timeout: *mut u32,
    bytes_copied: *mut u32,
    destination: *mut c_void,
    data_loss: *mut bool,
) -> Status {
    boundary(|| {
        let (device, _call) = match retained_gpu_device(agent) {
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    #[ignore = "requires a qualified GPU and KFD runtime"]
    fn shutdown_waits_for_retained_native_call() {
        assert_eq!(hsa_init(), SUCCESS);
        let (device, call) = retained_gpu_device(HsaAgent {
            handle: GPU_AGENT_BASE,
        })
        .unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || send.send(hsa_shut_down()).unwrap());
        let start = std::time::Instant::now();
        while lock().unwrap().is_some() {
            assert!(start.elapsed() < std::time::Duration::from_secs(1));
            std::thread::yield_now();
        }
        assert!(receive.try_recv().is_err());
        drop(device);
        drop(call);
        assert_eq!(
            receive
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap(),
            SUCCESS
        );
        worker.join().unwrap();
    }

    #[test]
    #[ignore = "requires a qualified GPU and KFD runtime"]
    fn native_clock_queries_return_live_gpu_values() {
        assert_eq!(hsa_init(), SUCCESS);
        let agent = HsaAgent {
            handle: GPU_AGENT_BASE,
        };
        let mut timestamp = 0_u64;
        let mut frequency = 0_u64;
        // SAFETY: Each output points to initialized writable storage of the
        // public attribute's size for the complete call.
        unsafe {
            assert_eq!(
                hsa_system_get_info(
                    SYSTEM_INFO_TIMESTAMP,
                    std::ptr::from_mut(&mut timestamp).cast()
                ),
                SUCCESS
            );
            assert_eq!(
                hsa_system_get_info(
                    SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                    std::ptr::from_mut(&mut frequency).cast()
                ),
                SUCCESS
            );
        }
        assert!(timestamp > 0);
        assert!(frequency > 0);
        let mut counters = HsaAmdClockCounters {
            gpu_clock_counter: 0,
            cpu_clock_counter: 0,
            system_clock_counter: 0,
            system_clock_frequency: 0,
        };
        let mut available_memory = 0_u64;
        // SAFETY: Both attributes receive live writable output storage.
        unsafe {
            assert_eq!(
                hsa_agent_get_info(
                    agent,
                    AMD_AGENT_INFO_CLOCK_COUNTERS,
                    std::ptr::from_mut(&mut counters).cast()
                ),
                SUCCESS
            );
            assert_eq!(
                hsa_agent_get_info(
                    agent,
                    AMD_AGENT_INFO_MEMORY_AVAIL,
                    std::ptr::from_mut(&mut available_memory).cast()
                ),
                SUCCESS
            );
        }
        assert_eq!(counters.system_clock_frequency, frequency);
        assert!(available_memory > 0);
        let mut translated = 0_u64;
        // SAFETY: The output is writable for the duration of the call.
        assert_eq!(
            unsafe {
                hsa_amd_profiling_convert_tick_to_system_domain(
                    agent,
                    counters.gpu_clock_counter,
                    &raw mut translated,
                )
            },
            SUCCESS
        );
        assert!(translated > 0);
        assert_eq!(hsa_shut_down(), SUCCESS);
    }

    #[test]
    fn extension_metadata_matches_implemented_capabilities() {
        assert_eq!(HSA_RUNTIME_VERSION_MINOR, 21);
        assert_eq!(AMD_SYSTEM_INFO_EXT_VERSION_MAJOR, 0x207);
        assert_eq!(AMD_SYSTEM_INFO_EXT_VERSION_MINOR, 0x208);
        assert_eq!(supported_extension_minor(EXTENSION_AMD_LOADER, 1), Some(0));
        assert_eq!(supported_extension_minor(EXTENSION_AMD_LOADER, 2), None);
        for extension in [
            EXTENSION_IMAGES,
            EXTENSION_FINALIZER,
            EXTENSION_AMD_PROFILER,
            EXTENSION_AMD_AQLPROFILE,
            EXTENSION_AMD_PC_SAMPLING,
        ] {
            assert_eq!(supported_extension_minor(extension, 1), None);
            assert!(!legacy_extension_supported(extension, 1, 0));
        }
        assert!(legacy_agent_extension_supported(
            EXTENSION_AMD_LOADER,
            true,
            1,
            0
        ));
        assert!(!legacy_agent_extension_supported(
            EXTENSION_AMD_LOADER,
            false,
            1,
            0
        ));
        assert!(major_agent_extension_supported(
            EXTENSION_AMD_LOADER,
            true,
            1
        ));
        assert!(!major_agent_extension_supported(
            EXTENSION_AMD_LOADER,
            true,
            2
        ));
        assert_eq!(
            extension_name(EXTENSION_AMD_LOADER),
            Some(&b"HSA_EXTENSION_AMD_LOADER\0"[..])
        );
        assert_eq!(extension_name(EXTENSION_AMD_PC_SAMPLING), None);

        let mask = extension_mask();
        assert_eq!(mask[0], 0);
        assert_eq!(mask[64], 0b0010);
        for extension in [
            EXTENSION_IMAGES,
            EXTENSION_AMD_PROFILER,
            EXTENSION_AMD_AQLPROFILE,
            EXTENSION_AMD_PC_SAMPLING,
        ] {
            assert_eq!(mask[usize::from(extension / 8)] & (1 << (extension % 8)), 0);
        }
    }

    #[test]
    fn agent_extension_answers_never_exceed_system_capabilities() {
        let extensions = [
            EXTENSION_FINALIZER,
            EXTENSION_IMAGES,
            EXTENSION_PERFORMANCE_COUNTERS,
            EXTENSION_PROFILING_EVENTS,
            EXTENSION_AMD_PROFILER,
            EXTENSION_AMD_LOADER,
            EXTENSION_AMD_AQLPROFILE,
            EXTENSION_AMD_PC_SAMPLING,
        ];
        for extension in extensions {
            for major in 0..=2 {
                for minor in 0..=1 {
                    let system = legacy_extension_supported(extension, major, minor);
                    let agent = legacy_agent_extension_supported(extension, true, major, minor);
                    assert!(!agent || system);
                }
                let system = supported_extension_minor(extension, major).is_some();
                let agent = major_agent_extension_supported(extension, true, major);
                assert!(!agent || system);
            }
        }
    }

    #[test]
    fn status_strings_cover_core_and_amd_status_codes() {
        let cases = [
            (SUCCESS, "HSA_STATUS_SUCCESS"),
            (INFO_BREAK, "HSA_STATUS_INFO_BREAK"),
            (ERROR, "HSA_STATUS_ERROR"),
            (INVALID_ARGUMENT, "HSA_STATUS_ERROR_INVALID_ARGUMENT"),
            (
                INVALID_QUEUE_CREATION,
                "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION",
            ),
            (INVALID_ALLOCATION, "HSA_STATUS_ERROR_INVALID_ALLOCATION"),
            (INVALID_AGENT, "HSA_STATUS_ERROR_INVALID_AGENT"),
            (INVALID_REGION, "HSA_STATUS_ERROR_INVALID_REGION"),
            (INVALID_SIGNAL, "HSA_STATUS_ERROR_INVALID_SIGNAL"),
            (INVALID_QUEUE, "HSA_STATUS_ERROR_INVALID_QUEUE"),
            (OUT_OF_RESOURCES, "HSA_STATUS_ERROR_OUT_OF_RESOURCES"),
            (
                INVALID_PACKET_FORMAT,
                "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT",
            ),
            (RESOURCE_FREE, "HSA_STATUS_ERROR_RESOURCE_FREE"),
            (NOT_INITIALIZED, "HSA_STATUS_ERROR_NOT_INITIALIZED"),
            (REFCOUNT_OVERFLOW, "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW"),
            (
                INCOMPATIBLE_ARGUMENTS,
                "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS",
            ),
            (INVALID_INDEX, "HSA_STATUS_ERROR_INVALID_INDEX"),
            (INVALID_ISA, "HSA_STATUS_ERROR_INVALID_ISA"),
            (INVALID_CODE_OBJECT, "HSA_STATUS_ERROR_INVALID_CODE_OBJECT"),
            (INVALID_EXECUTABLE, "HSA_STATUS_ERROR_INVALID_EXECUTABLE"),
            (FROZEN_EXECUTABLE, "HSA_STATUS_ERROR_FROZEN_EXECUTABLE"),
            (INVALID_SYMBOL_NAME, "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME"),
            (
                VARIABLE_ALREADY_DEFINED,
                "HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED",
            ),
            (VARIABLE_UNDEFINED, "HSA_STATUS_ERROR_VARIABLE_UNDEFINED"),
            (EXCEPTION, "HSA_STATUS_ERROR_EXCEPTION"),
            (INVALID_ISA_NAME, "HSA_STATUS_ERROR_INVALID_ISA_NAME"),
            (INVALID_CODE_SYMBOL, "HSA_STATUS_ERROR_INVALID_CODE_SYMBOL"),
            (
                INVALID_EXECUTABLE_SYMBOL,
                "HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL",
            ),
            (INVALID_FILE, "HSA_STATUS_ERROR_INVALID_FILE"),
            (
                INVALID_CODE_OBJECT_READER,
                "HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER",
            ),
            (INVALID_CACHE, "HSA_STATUS_ERROR_INVALID_CACHE"),
            (INVALID_WAVEFRONT, "HSA_STATUS_ERROR_INVALID_WAVEFRONT"),
            (
                INVALID_SIGNAL_GROUP,
                "HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP",
            ),
            (
                INVALID_RUNTIME_STATE,
                "HSA_STATUS_ERROR_INVALID_RUNTIME_STATE",
            ),
            (FATAL, "HSA_STATUS_ERROR_FATAL"),
            (INVALID_MEMORY_POOL, "HSA_STATUS_ERROR_INVALID_MEMORY_POOL"),
            (
                MEMORY_APERTURE_VIOLATION,
                "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION",
            ),
            (ILLEGAL_INSTRUCTION, "HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION"),
            (MEMORY_FAULT, "HSA_STATUS_ERROR_MEMORY_FAULT"),
            (CU_MASK_REDUCED, "HSA_STATUS_CU_MASK_REDUCED"),
            (OUT_OF_REGISTERS, "HSA_STATUS_ERROR_OUT_OF_REGISTERS"),
            (RESOURCE_BUSY, "HSA_STATUS_ERROR_RESOURCE_BUSY"),
            (NOT_SUPPORTED, "HSA_STATUS_ERROR_NOT_SUPPORTED"),
            (XNACK_DISABLED, "HSA_STATUS_ERROR_XNACK_DISABLED"),
            (
                INVALID_DISPATCH_PARAMETERS,
                "HSA_STATUS_ERROR_INVALID_DISPATCH_PARAMETERS",
            ),
            (RESOURCE_NOT_READY, "HSA_STATUS_ERROR_RESOURCE_NOT_READY"),
        ];
        for (status, expected_name) in cases {
            let mut output = std::ptr::null();
            // SAFETY: output is writable pointer storage for the returned static string.
            assert_eq!(
                unsafe { hsa_status_string(status, &raw mut output) },
                SUCCESS
            );
            // SAFETY: hsa_status_string returned a pointer to a static NUL-terminated string.
            let value = unsafe { CStr::from_ptr(output) }.to_str().unwrap();
            assert!(
                value.starts_with(expected_name),
                "status {status:#x}: {value:?} does not start with {expected_name:?}"
            );
        }
    }

    #[test]
    fn status_string_rejects_unknown_status_without_writing_output() {
        let sentinel = c"unchanged".as_ptr();
        let mut output = sentinel;
        // SAFETY: output is writable pointer storage.
        assert_eq!(
            unsafe { hsa_status_string(u32::MAX, &raw mut output) },
            INVALID_ARGUMENT
        );
        assert_eq!(output, sentinel);
    }

    #[test]
    fn amd_agent_helpers_match_the_public_abi() {
        assert_eq!(std::mem::size_of::<HsaAmdEvent>(), 32);
        assert_eq!(std::mem::align_of::<HsaAmdEvent>(), 8);
        assert_eq!(std::mem::size_of::<HsaAmdDim3>(), 24);
        assert_eq!(std::mem::align_of::<HsaAmdDim3>(), 8);
        assert_eq!(std::mem::size_of::<HsaLuid>(), 8);
        assert_eq!(std::mem::align_of::<HsaLuid>(), 4);
        assert_eq!(AMD_AGENT_INFO_ASIC_FAMILY_ID, 0xa107);
        assert_eq!(AMD_AGENT_INFO_LUID, 0xa11a);
        assert_eq!(AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM, 0xa11e);
        assert_eq!(AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_SIZE, 0xa11f);
        assert_eq!(AMD_AGENT_INFO_CLUSTER_MAX_DIM, 0xa120);
        assert_eq!(AMD_AGENT_INFO_CLUSTER_MAX_SIZE, 0xa121);
        assert_eq!(AMD_AGENT_INFO_KERNEL_WG_MAX_DIM, 0xa122);
        assert_eq!(KERNEL_CLUSTER_MAX_DIM.x, i32::MAX as u64);
        assert_eq!(KERNEL_CLUSTER_MAX_DIM.y, u64::from(u16::MAX));
        assert_eq!(KERNEL_CLUSTER_MAX_DIM.z, u64::from(u16::MAX));
        assert_eq!(CLUSTER_MAX_DIM, HsaAmdDim3 { x: 1, y: 1, z: 1 });
        assert_eq!(
            KERNEL_CLUSTER_MAX_SIZE,
            KERNEL_CLUSTER_MAX_DIM.x * KERNEL_CLUSTER_MAX_DIM.y * KERNEL_CLUSTER_MAX_DIM.z
        );
        assert_eq!(agent_uuid(None), "CPU-XX");
        assert_eq!(
            agent_uuid(Some(&rocddi::topology::GpuInfo::default())),
            "GPU-XX"
        );
        assert_eq!(
            agent_uuid(Some(&rocddi::topology::GpuInfo {
                unique_id: Some(0x0123_4567_89ab_cdef),
                ..rocddi::topology::GpuInfo::default()
            })),
            "GPU-0123456789abcdef"
        );
        assert!(!host_alloc_dmabuf_supported(0));
        assert!(host_alloc_dmabuf_supported(1));
        for attribute in [
            AMD_AGENT_INFO_MEMORY_WIDTH,
            AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY,
            AMD_AGENT_INFO_COOPERATIVE_QUEUES,
            AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT,
            AMD_AGENT_INFO_MEMORY_AVAIL,
            AMD_AGENT_INFO_PM4_EMULATION,
            AMD_AGENT_INFO_LUID,
            AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE,
            AMD_AGENT_INFO_CUID,
            AMD_AGENT_INFO_KERNEL_WG_MAX_SIZE,
            AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_DIM,
            AMD_AGENT_INFO_KERNEL_CLUSTER_MAX_SIZE,
            AMD_AGENT_INFO_CLUSTER_MAX_DIM,
            AMD_AGENT_INFO_CLUSTER_MAX_SIZE,
            AMD_AGENT_INFO_KERNEL_WG_MAX_DIM,
        ] {
            assert!(cpu_rejects_amd_agent_info(attribute));
        }
        assert!(!cpu_rejects_amd_agent_info(AMD_AGENT_INFO_NEAREST_CPU));
        assert_eq!(nearest_cpu_agent(false).handle, 0);
        assert_eq!(nearest_cpu_agent(true).handle, CPU_AGENT);
        let _: unsafe extern "C" fn(*mut u8, *mut c_void) -> Status = hsa_amd_enable_logging;
        let _: unsafe extern "C" fn(HsaAgent, *mut u32) -> Status = hsa_amd_coherency_get_type;
        let _: unsafe extern "C" fn(HsaAgent, u32, *mut c_void) -> Status =
            hsa_amd_agent_set_attribute;
        let _: unsafe extern "C" fn(HsaAgent, u32) -> Status = hsa_amd_coherency_set_type;
        let _: extern "C" fn(HsaAgent, u64) -> Status = hsa_amd_agent_preload;
        let _: extern "C" fn(HsaAgent, usize) -> Status = hsa_amd_agent_set_async_scratch_limit;
        let _: unsafe extern "C" fn(HsaAgent, u64, *mut u64) -> Status =
            hsa_amd_profiling_convert_tick_to_system_domain;
        let _: extern "C" fn(HsaAgent) -> Status = hsa_amd_spm_acquire;
        let _: extern "C" fn(HsaAgent) -> Status = hsa_amd_spm_release;
        let _: unsafe extern "C" fn(
            HsaAgent,
            usize,
            *mut u32,
            *mut u32,
            *mut c_void,
            *mut bool,
        ) -> Status = hsa_amd_spm_set_dest_buffer;
    }

    #[test]
    fn isa_entry_points_match_the_public_abi() {
        let _: unsafe extern "C" fn(HsaAgent, u32, *mut u16) -> Status =
            hsa_agent_get_exception_policies;
        let _: unsafe extern "C" fn(u16, HsaAgent, u16, u16, *mut bool) -> Status =
            hsa_agent_extension_supported;
        let _: unsafe extern "C" fn(u16, HsaAgent, u16, *mut u16, *mut bool) -> Status =
            hsa_agent_major_extension_supported;
        let _: unsafe extern "C" fn(*const c_char, *mut HsaIsa) -> Status = hsa_isa_from_name;
        let _: unsafe extern "C" fn(HsaIsa, u32, u32, *mut c_void) -> Status = hsa_isa_get_info;
        let _: unsafe extern "C" fn(HsaIsa, u32, *mut u16) -> Status =
            hsa_isa_get_exception_policies;
        let _: unsafe extern "C" fn(HsaIsa, u32, u32, *mut u32) -> Status =
            hsa_isa_get_round_method;
        let _: unsafe extern "C" fn(HsaIsa, WavefrontCallback, *mut c_void) -> Status =
            hsa_isa_iterate_wavefronts;
        let _: unsafe extern "C" fn(HsaWavefront, u32, *mut c_void) -> Status =
            hsa_wavefront_get_info;
        let _: unsafe extern "C" fn(HsaIsa, HsaIsa, *mut bool) -> Status = hsa_isa_compatible;
    }

    #[test]
    fn gfx12_isa_names_match_rocr() {
        assert_eq!(
            isa_name(12, 0, 1, 0).as_deref(),
            Some("amdgcn-amd-amdhsa--gfx1201")
        );
        assert_eq!(
            isa_name(12, 0, 1, 1).as_deref(),
            Some("amdgcn-amd-amdhsa--gfx12-generic")
        );
        assert_eq!(isa_name(12, 0, 1, ISA_COUNT_PER_GPU), None);
    }
}
