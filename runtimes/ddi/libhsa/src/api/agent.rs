// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA C ABI entry points for this domain.

use super::host_alloc_dmabuf_supported;
use super::output::{write_bytes, write_value};
use super::system::{
    extension_mask, legacy_agent_extension_supported, major_agent_extension_supported,
    valid_extension,
};
use std::ffi::c_void;

use crate::ffi::*;
use crate::isa::isa_handle;
use crate::runtime::{boundary, lock, map_error};
use crate::{image_gfx12, platform, runtime};

pub(crate) const KERNEL_CLUSTER_MAX_DIM: HsaAmdDim3 = HsaAmdDim3 {
    x: i32::MAX as u64,
    y: 65_535,
    z: 65_535,
};
pub(crate) const CLUSTER_MAX_DIM: HsaAmdDim3 = HsaAmdDim3 { x: 1, y: 1, z: 1 };
pub(crate) const KERNEL_CLUSTER_MAX_SIZE: u64 =
    KERNEL_CLUSTER_MAX_DIM.x * KERNEL_CLUSTER_MAX_DIM.y * KERNEL_CLUSTER_MAX_DIM.z;

pub(crate) fn agent_uuid(gpu: Option<&rocddi::topology::GpuInfo>) -> String {
    match gpu {
        None => "CPU-XX".to_owned(),
        Some(gpu) => gpu.unique_id.map_or_else(
            || "GPU-XX".to_owned(),
            |unique_id| format!("GPU-{unique_id:016x}"),
        ),
    }
}

pub(crate) fn cpu_rejects_amd_agent_info(attribute: u32) -> bool {
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
            | AMD_AGENT_INFO_PCIE_STORES_NEED_FLUSH
            | AMD_AGENT_INFO_HOST_ATOMICS_SUPPORTED
            | AMD_AGENT_INFO_SIGNAL_HINTS_SUPPORTED
    )
}

pub(crate) fn nearest_cpu_agent(gpu_only: bool) -> HsaAgent {
    HsaAgent {
        handle: if gpu_only { CPU_AGENT } else { 0 },
    }
}

/// # Safety
/// The callback must be callable with the C ABI; `data` must remain valid for
/// any access the callback performs during this call.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_iterate_agents")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_agent_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            pc_sampling_supported,
            host_xgmi_atomics,
            cacheline_size,
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
                    false,
                    false,
                    64,
                )
            } else if let Some(index) = runtime.gpu_index(agent) {
                let endpoint = &runtime.gpus[index].endpoint;
                let host_xgmi_atomics =
                    runtime
                        .topology
                        .route_from_host(endpoint)
                        .is_some_and(|link| {
                            link.kind == rocddi::topology::MemoryLinkType::Xgmi
                                && link.coherent_support
                                && link.atomic_support_64bit
                        });
                let cacheline_size = runtime
                    .topology
                    .caches_for_endpoint(endpoint)
                    .filter(|cache| cache.level == 2)
                    .filter_map(|cache| cache.line_size_bytes)
                    .find(|line| *line != 0)
                    .unwrap_or(256);
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
                    runtime.gpus[index].pc_sampling_supported,
                    host_xgmi_atomics,
                    cacheline_size,
                )
            } else {
                return INVALID_AGENT;
            }
        };
        let clock_counters = if attribute == AMD_AGENT_INFO_CLOCK_COUNTERS {
            native_device
                .as_ref()
                .map(|(device, _)| runtime::clock_counters(device))
        } else {
            None
        };
        let available_memory = if attribute == AMD_AGENT_INFO_MEMORY_AVAIL {
            native_device
                .as_ref()
                .map(|(device, _)| device.gpu().and_then(|gpu| gpu.available_memory()))
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
        let image_gpu = gpu.is_some_and(image_gfx12::supported_gpu);
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
                AGENT_INFO_EXTENSIONS => write_value(
                    value,
                    if gpu_only {
                        extension_mask(image_gpu, pc_sampling_supported)
                    } else {
                        [0; 128]
                    },
                ),
                AGENT_INFO_VERSION_MAJOR | AGENT_INFO_VERSION_MINOR => write_value(value, 1_u16),
                AMD_AGENT_INFO_CHIP_ID => write_value(value, pci.map_or(0, |info| info.device_id)),
                AMD_AGENT_INFO_CACHELINE_SIZE => write_value(value, cacheline_size),
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
                AMD_AGENT_INFO_PCIE_STORES_NEED_FLUSH => {
                    write_value(value, gpu_only && !host_xgmi_atomics)
                }
                AMD_AGENT_INFO_HOST_ATOMICS_SUPPORTED => write_value(value, host_xgmi_atomics),
                AMD_AGENT_INFO_SIGNAL_HINTS_SUPPORTED => write_value(value, false),
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
                EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS | EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS => {
                    write_value(value, if image_gpu { 16_384_u32 } else { 0 })
                }
                EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS => write_value(
                    value,
                    if image_gpu {
                        image_gfx12::BUFFER_MAX_ELEMENTS
                    } else {
                        0
                    },
                ),
                EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS | EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS => {
                    write_value(value, if image_gpu { [16_384_u32; 2] } else { [0; 2] })
                }
                EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS
                | EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS => write_value(value, [0_u32; 2]),
                EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS => write_value(
                    value,
                    if image_gpu {
                        [16_384_u32, 16_384, 8_192]
                    } else {
                        [0; 3]
                    },
                ),
                EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS => {
                    write_value(value, if image_gpu { 8_192_u32 } else { 0 })
                }
                EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES => {
                    write_value(value, if image_gpu { 128_u32 } else { 0 })
                }
                EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES => {
                    write_value(value, if image_gpu { 64_u32 } else { 0 })
                }
                EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS => {
                    write_value(value, if image_gpu { 16_u32 } else { 0 })
                }
                EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT => write_value(
                    value,
                    if image_gpu {
                        image_gfx12::ROW_ALIGNMENT
                    } else {
                        0
                    },
                ),
                EXT_AGENT_INFO_IMAGE_SUPPORT => write_value(value, image_gpu),
                _ => INVALID_ARGUMENT,
            }
        }
    })
}

/// # Safety
/// A non-null `value` must point to readable, aligned `size_t` storage.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_agent_set_attribute")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_agent_get_exception_policies")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_agent_extension_supported")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let supported_gpu = gpu_index.is_some_and(|index| match extension {
            EXTENSION_IMAGES => image_gfx12::supported_gpu(&runtime.gpus[index].info),
            EXTENSION_AMD_PC_SAMPLING => runtime.gpus[index].pc_sampling_supported,
            _ => true,
        });
        let supported = legacy_agent_extension_supported(
            extension,
            supported_gpu,
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_agent_major_extension_supported")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let supported_gpu = gpu_index.is_some_and(|index| match extension {
            EXTENSION_IMAGES => image_gfx12::supported_gpu(&runtime.gpus[index].info),
            EXTENSION_AMD_PC_SAMPLING => runtime.gpus[index].pc_sampling_supported,
            _ => true,
        });
        let supported = major_agent_extension_supported(extension, supported_gpu, version_major);
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
