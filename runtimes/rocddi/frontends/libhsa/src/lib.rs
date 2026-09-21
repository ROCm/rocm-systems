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
    clippy::missing_safety_doc,
    clippy::semicolon_if_nothing_returned,
    clippy::too_many_lines,
    clippy::wildcard_imports
)]

use std::ffi::{CStr, c_char, c_void};

#[cfg(target_os = "linux")]
core::arch::global_asm!(include_str!("exports_linux.S"));

mod ffi;
mod finalizer;
mod image;
mod loader;
mod memory;
mod pc_sampling;
mod queue;
mod runtime;
mod signal;
mod trap_handler_gfx12;

use ffi::*;
use runtime::{LifecycleTransition, Runtime, boundary, initialized_mut, lock, map_error};

const HSA_RUNTIME_VERSION_MINOR: u16 = 21;

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
    x: u32::MAX as u64,
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
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let (status, transitioned) = match result {
            Ok(runtime) => {
                let frequency = runtime
                    .gpus
                    .first()
                    .and_then(|gpu| gpu.device.gpu().ok())
                    .and_then(|gpu| gpu.clock_counters().ok())
                    .map_or(0, |counters| counters.system_frequency);
                match guard.publish_init(generation, runtime) {
                    Ok(()) => {
                        signal::set_system_frequency(frequency);
                        (SUCCESS, true)
                    }
                    Err(status) => (status, false),
                }
            }
            Err(status) => match guard.cancel_init(generation) {
                Ok(()) => (status, true),
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
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                SYSTEM_INFO_VERSION_MAJOR => write_value(value, 1_u16),
                SYSTEM_INFO_VERSION_MINOR => write_value(value, HSA_RUNTIME_VERSION_MINOR),
                SYSTEM_INFO_TIMESTAMP => match runtime.system_timestamp() {
                    Ok(timestamp) => write_value(value, timestamp),
                    Err(status) => status,
                },
                SYSTEM_INFO_TIMESTAMP_FREQUENCY => match runtime.gpus[0]
                    .device
                    .gpu()
                    .and_then(|gpu| gpu.clock_counters())
                {
                    Ok(counters) => write_value(value, counters.system_frequency),
                    Err(error) => map_error(error),
                },
                SYSTEM_INFO_SIGNAL_MAX_WAIT => write_value(value, u64::MAX),
                SYSTEM_INFO_ENDIANNESS => write_value(value, 0_u32),
                SYSTEM_INFO_MACHINE_MODEL => write_value(value, 1_u32),
                SYSTEM_INFO_EXTENSIONS => write_value(value, extension_mask(false)),
                AMD_SYSTEM_INFO_SVM_SUPPORTED => write_value(value, true),
                AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT => write_value(value, false),
                AMD_SYSTEM_INFO_MWAITX_ENABLED => write_value(value, false),
                AMD_SYSTEM_INFO_DMABUF_SUPPORTED => write_value(value, true),
                AMD_SYSTEM_INFO_EXT_VERSION_MAJOR => write_value(value, 1_u16),
                AMD_SYSTEM_INFO_EXT_VERSION_MINOR => write_value(value, 32_u16),
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

fn extension_mask(pc_sampling: bool) -> [u8; 128] {
    let mut extensions = [0_u8; 128];
    for extension in [
        EXTENSION_IMAGES,
        EXTENSION_AMD_PROFILER,
        EXTENSION_AMD_LOADER,
    ] {
        extensions[usize::from(extension / 8)] |= 1 << (extension % 8);
    }
    if pc_sampling {
        extensions[usize::from(EXTENSION_AMD_PC_SAMPLING / 8)] |=
            1 << (EXTENSION_AMD_PC_SAMPLING % 8);
    }
    extensions
}

fn supported_extension_minor(extension: u16, version_major: u16) -> Option<u16> {
    (version_major == 1 && matches!(extension, EXTENSION_IMAGES | EXTENSION_AMD_LOADER))
        .then_some(0)
}

fn legacy_extension_supported(extension: u16, version_major: u16, version_minor: u16) -> bool {
    matches!(version_major, 0 | 1)
        && version_minor == 0
        && matches!(
            extension,
            EXTENSION_IMAGES
                | EXTENSION_AMD_PROFILER
                | EXTENSION_AMD_LOADER
                | EXTENSION_AMD_PC_SAMPLING
        )
}

fn legacy_agent_extension_supported(gpu: bool, version_major: u16, version_minor: u16) -> bool {
    gpu && version_major <= 1 && version_minor == 0
}

fn major_agent_extension_supported(gpu: bool, version_major: u16) -> bool {
    gpu && version_major <= 1
}

fn valid_extension(extension: u16) -> bool {
    extension <= EXTENSION_PROFILING_EVENTS
        || (EXTENSION_AMD_PROFILER..=EXTENSION_AMD_PC_SAMPLING).contains(&extension)
}

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
        if guard.is_none() {
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
        if guard.is_none() {
            return NOT_INITIALIZED;
        }
        if version_minor.is_null() || result.is_null() {
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
            clock_counters,
            available_memory,
            host_alloc_dmabuf,
            cache_sizes,
            asic_family_id,
            hdp_flush,
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
                    None,
                    host_alloc_dmabuf_supported(runtime.gpus.len()),
                    runtime.agent_cache_sizes(agent),
                    0,
                    [0; 2],
                )
            } else if let Some(index) = runtime.gpu_index(agent) {
                let counters = (attribute == AMD_AGENT_INFO_CLOCK_COUNTERS).then(|| {
                    runtime.gpus[index]
                        .device
                        .gpu()
                        .and_then(|gpu| gpu.clock_counters())
                });
                let available_memory = (attribute == AMD_AGENT_INFO_MEMORY_AVAIL)
                    .then(|| runtime.gpus[index].device.available_memory());
                (
                    Some(index),
                    Some(runtime.gpus[index].endpoint.clone()),
                    runtime.gpus[index].name.clone(),
                    runtime.gpus[index].product_name.clone(),
                    runtime.host_compute_units,
                    counters,
                    available_memory,
                    host_alloc_dmabuf_supported(runtime.gpus.len()),
                    runtime.agent_cache_sizes(agent),
                    runtime.gpus[index].asic_family_id,
                    runtime.gpus[index].hdp_flush,
                )
            } else {
                return INVALID_AGENT;
            }
        };
        let gpu = endpoint.as_ref().and_then(|endpoint| endpoint.gpu());
        let gpu_only = gpu.is_some();
        if !gpu_only && cpu_rejects_amd_agent_info(attribute) {
            return INVALID_ARGUMENT;
        }
        let node = endpoint
            .as_ref()
            .map_or(0, |endpoint| endpoint.linux_kfd_drm_info().node_id);
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
                AGENT_INFO_EXTENSIONS => write_value(
                    value,
                    if gpu_only {
                        extension_mask(true)
                    } else {
                        [0; 128]
                    },
                ),
                AGENT_INFO_VERSION_MAJOR | AGENT_INFO_VERSION_MINOR => write_value(value, 1_u16),
                AMD_AGENT_INFO_CHIP_ID => write_value(value, pci.map_or(0, |info| info.device_id)),
                AMD_AGENT_INFO_CACHELINE_SIZE => {
                    write_value(value, if gpu_only { 256_u32 } else { 64 })
                }
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
                    write_value(value, if gpu_only { 256_u32 } else { 0 })
                }
                AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY => {
                    write_value(value, if gpu_only { 1258_u32 } else { 0 })
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
                AMD_AGENT_INFO_TIMESTAMP_FREQUENCY => write_value(
                    value,
                    if gpu_only {
                        100_000_000_u64
                    } else {
                        1_000_000_000
                    },
                ),
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
                AMD_AGENT_INFO_DRIVER_UID => write_value(
                    value,
                    endpoint
                        .as_ref()
                        .map_or(0, |endpoint| endpoint.linux_kfd_drm_info().gpu_id),
                ),
                AMD_AGENT_INFO_MAX_DATA_PREFETCH_REGIONS => write_value(value, 0_u32),
                AMD_AGENT_INFO_NUM_XCC => write_value(value, gpu.map_or(0, |info| info.xcc_count)),
                AMD_AGENT_INFO_NEAREST_CPU => write_value(value, nearest_cpu_agent(gpu_only)),
                AMD_AGENT_INFO_MEMORY_PROPERTIES | AMD_AGENT_INFO_AQL_EXTENSIONS => {
                    write_value(value, [0_u8; 8])
                }
                AMD_AGENT_INFO_SCRATCH_LIMIT_MAX => write_value(
                    value,
                    if gpu_only {
                        8_u64 * 1024 * 1024 * 1024
                    } else {
                        0
                    },
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
                AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE => write_value(value, gpu_only),
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
                EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS | EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS => {
                    write_value(value, if gpu_only { 16_384_u32 } else { 0 })
                }
                EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS => {
                    write_value(value, if gpu_only { u32::MAX } else { 0 })
                }
                EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS => write_value(
                    value,
                    if gpu_only {
                        [16_384_u32, 16_384]
                    } else {
                        [0; 2]
                    },
                ),
                EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS => write_value(
                    value,
                    if gpu_only {
                        [16_384_u32, 16_384, 8192]
                    } else {
                        [0; 3]
                    },
                ),
                EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS => {
                    write_value(value, if gpu_only { 8192_u32 } else { 0 })
                }
                EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES => {
                    write_value(value, if gpu_only { 128_u32 } else { 0 })
                }
                EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES => {
                    write_value(value, if gpu_only { 64_u32 } else { 0 })
                }
                EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS => {
                    write_value(value, if gpu_only { 16_u32 } else { 0 })
                }
                EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT => write_value(value, 256_usize),
                EXT_AGENT_INFO_IMAGE_SUPPORT => write_value(value, gpu_only),
                _ => INVALID_ARGUMENT,
            }
        }
    })
}

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
        let supported = legacy_agent_extension_supported(
            runtime.gpu_index(agent).is_some(),
            version_major,
            version_minor,
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { result.write(supported) };
        SUCCESS
    })
}

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
        let supported =
            major_agent_extension_supported(runtime.gpu_index(agent).is_some(), version_major);
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
        if guard.is_none() {
            return NOT_INITIALIZED;
        }
        match (extension, version_major) {
            (EXTENSION_IMAGES, 1) => {
                let functions = image::image_extension_table();
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
            (EXTENSION_AMD_PC_SAMPLING, 1) => {
                let functions = pc_sampling::extension_table();
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_enable_logging(flags: *mut u8, file: *mut c_void) -> Status {
    boundary(|| {
        if flags.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        // SAFETY: The public ABI requires flags to address an eight-byte array
        // that remains readable for the duration of this call.
        runtime
            .log_flags
            .copy_from_slice(unsafe { std::slice::from_raw_parts(flags, 8) });
        runtime.log_file = file as usize;
        SUCCESS
    })
}

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
            .push((callback, data as usize));
        SUCCESS
    })
}

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
pub unsafe extern "C" fn hsa_amd_coherency_set_type(agent: HsaAgent, kind: u32) -> Status {
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
pub extern "C" fn hsa_amd_agent_preload(agent: HsaAgent, _flags: u64) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        match guard.as_ref() {
            Some(runtime) if runtime.gpu_index(agent).is_some() => SUCCESS,
            Some(_) => INVALID_AGENT,
            None => NOT_INITIALIZED,
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
pub extern "C" fn hsa_amd_profiling_async_copy_enable(_enable: bool) -> Status {
    boundary(|| {
        lock().map_or(ERROR, |runtime| {
            if runtime.is_some() {
                SUCCESS
            } else {
                NOT_INITIALIZED
            }
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_profiling_convert_tick_to_system_domain(
    agent: HsaAgent,
    agent_tick: u64,
    system_tick: *mut u64,
) -> Status {
    boundary(|| {
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
        let translated = match runtime.translate_gpu_tick(index, agent_tick) {
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
        runtime.gpus[index]
            .device
            .gpu()
            .and_then(|gpu| gpu.spm_acquire())
            .map_or_else(map_error, |()| SUCCESS)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_spm_release(agent: HsaAgent) -> Status {
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
        runtime.gpus[index]
            .device
            .gpu()
            .and_then(|gpu| gpu.spm_release())
            .map_or_else(map_error, |()| SUCCESS)
    })
}

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
        let result = runtime.gpus[index].device.gpu().and_then(|gpu| {
            gpu.spm_set_destination(
                size,
                &mut timeout_value,
                &mut copied_value,
                (!destination.is_null()).then_some(destination as usize),
                &mut loss_value,
            )
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
    fn extension_metadata_matches_implemented_capabilities() {
        assert_eq!(HSA_RUNTIME_VERSION_MINOR, 21);
        assert_eq!(AMD_SYSTEM_INFO_EXT_VERSION_MAJOR, 0x207);
        assert_eq!(AMD_SYSTEM_INFO_EXT_VERSION_MINOR, 0x208);
        assert_eq!(supported_extension_minor(EXTENSION_AMD_LOADER, 1), Some(0));
        assert_eq!(supported_extension_minor(EXTENSION_AMD_LOADER, 2), None);
        assert_eq!(supported_extension_minor(EXTENSION_IMAGES, 1), Some(0));
        assert_eq!(supported_extension_minor(EXTENSION_FINALIZER, 1), None);
        assert_eq!(supported_extension_minor(EXTENSION_AMD_AQLPROFILE, 1), None);
        assert!(!legacy_extension_supported(EXTENSION_FINALIZER, 1, 0));
        assert!(!legacy_extension_supported(EXTENSION_AMD_AQLPROFILE, 1, 0));
        assert_eq!(
            supported_extension_minor(EXTENSION_AMD_PC_SAMPLING, 1),
            None
        );
        assert!(legacy_extension_supported(EXTENSION_AMD_PROFILER, 0, 0));
        assert!(!legacy_extension_supported(EXTENSION_AMD_PROFILER, 1, 1));
        assert!(legacy_agent_extension_supported(true, 0, 0));
        assert!(legacy_agent_extension_supported(true, 1, 0));
        assert!(!legacy_agent_extension_supported(true, 1, 1));
        assert!(!legacy_agent_extension_supported(false, 1, 0));
        assert!(major_agent_extension_supported(true, 0));
        assert!(major_agent_extension_supported(true, 1));
        assert!(!major_agent_extension_supported(true, 2));
        assert_eq!(
            extension_name(EXTENSION_AMD_LOADER),
            Some(&b"HSA_EXTENSION_AMD_LOADER\0"[..])
        );
        assert_eq!(extension_name(EXTENSION_AMD_PC_SAMPLING), None);

        let system = extension_mask(false);
        assert_eq!(system[0], 0b10);
        assert_eq!(system[64], 0b0011);
        let agent = extension_mask(true);
        assert_eq!(agent[0], 0b10);
        assert_eq!(agent[64], 0b1011);
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
        assert_eq!(KERNEL_CLUSTER_MAX_DIM.x, u64::from(u32::MAX));
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
