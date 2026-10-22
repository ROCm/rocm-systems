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
    clippy::must_use_candidate,
    clippy::semicolon_if_nothing_returned,
    clippy::too_many_lines,
    clippy::wildcard_imports
)]

use std::ffi::{CStr, c_char, c_void};

// Unit-test executables call the Rust entry points directly. They do not
// export the shared library's versioned ABI symbols.
#[cfg(all(target_os = "linux", not(test)))]
core::arch::global_asm!(include_str!("exports_linux.S"));
#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
core::arch::global_asm!(include_str!(concat!(
    env!("OUT_DIR"),
    "/api_trace_dispatch_linux.S"
)));

#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
mod api_trace;
mod callback_arg;
mod ffi;
mod finalizer;
mod image_abi;
mod image_gfx12;
mod loader;
mod memory;
mod pc_sampling;
mod platform;
mod queue;
mod queue_intercept;
mod runtime;
mod signal;
mod tool_events;

use callback_arg::CallbackArg;
use ffi::*;
use runtime::{
    HsaState, InFlightToken, LifecycleTransition, boundary, initialized_mut, lock, map_error,
};

fn complete_deferred_shutdown(runtime: HsaState, generation: u64) {
    let mut transition = LifecycleTransition::stopping(generation);
    let status = runtime.stop();
    #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
    api_trace::unload();
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

fn isa_name(gpu: &rocddi::topology::GpuInfo, xnack_enabled: bool, variant: u64) -> Option<String> {
    let base = match variant {
        0 if (gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping) == (12, 5, 0)
            && gpu.asic_revision == 0 =>
        {
            "gfx1250-strict".to_owned()
        }
        0 => runtime::gfx_target_name(gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping),
        1 if gpu.gfx_major == 9 && matches!(gpu.gfx_minor, 4 | 5) => "gfx9-4-generic".to_owned(),
        1 if gpu.gfx_major == 9 => return None,
        1 if gpu.gfx_major == 12 && gpu.gfx_minor == 5 => "gfx12-5-generic".to_owned(),
        1 => format!("gfx{}-generic", gpu.gfx_major),
        _ => return None,
    };
    let features = if gpu.gfx_major == 9 {
        format!(
            ":sramecc{}:xnack{}",
            if gpu.sram_ecc_enabled { '+' } else { '-' },
            if xnack_enabled { '+' } else { '-' }
        )
    } else {
        String::new()
    };
    Some(format!("amdgcn-amd-amdhsa--{base}{features}"))
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
            | AMD_AGENT_INFO_PCIE_STORES_NEED_FLUSH
            | AMD_AGENT_INFO_HOST_ATOMICS_SUPPORTED
            | AMD_AGENT_INFO_SIGNAL_HINTS_SUPPORTED
    )
}

fn nearest_cpu_agent(gpu_only: bool) -> HsaAgent {
    HsaAgent {
        handle: if gpu_only { CPU_AGENT } else { 0 },
    }
}

mod api;
pub(crate) use api::agent::hsa_agent_get_info;

#[cfg(test)]
pub(crate) use api::lifecycle::{hsa_init, hsa_shut_down};
#[cfg(test)]
use api::{agent::*, amd::*, isa::*, system::*};

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
        assert!(legacy_extension_supported(EXTENSION_IMAGES, 1, 0));
        for extension in [
            EXTENSION_FINALIZER,
            EXTENSION_AMD_PROFILER,
            EXTENSION_AMD_AQLPROFILE,
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
        assert_eq!(
            extension_name(EXTENSION_AMD_PC_SAMPLING),
            Some(&b"HSA_EXTENSION_AMD_PC_SAMPLING\0"[..])
        );

        let mask = extension_mask(false, false);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[64], 0b0010);
        assert_eq!(extension_mask(true, false)[0], 0b0010);
        for extension in [
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
        let gpu = rocddi::topology::GpuInfo {
            gfx_major: 12,
            gfx_minor: 0,
            gfx_stepping: 1,
            ..Default::default()
        };
        assert_eq!(
            isa_name(&gpu, false, 0).as_deref(),
            Some("amdgcn-amd-amdhsa--gfx1201")
        );
        assert_eq!(
            isa_name(&gpu, false, 1).as_deref(),
            Some("amdgcn-amd-amdhsa--gfx12-generic")
        );
        assert_eq!(isa_name(&gpu, false, ISA_COUNT_PER_GPU), None);
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
        while lock().unwrap().as_ref().is_some() {
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
}
