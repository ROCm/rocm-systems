// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! System information, status strings, and extension discovery queries.

use std::ffi::{c_char, c_void};

use super::host_alloc_dmabuf_supported;
use super::output::write_value;
#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
use crate::api_trace;
use crate::ffi::{self, *};
use crate::runtime::{boundary, lock, map_error};
#[cfg(not(all(target_os = "linux", target_arch = "x86_64", not(test))))]
use crate::{image_abi, loader, pc_sampling};
use crate::{image_gfx12, runtime};

/// Minor version of the HSA runtime ABI reported to clients.
pub(crate) const HSA_RUNTIME_VERSION_MINOR: u16 = 21;

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_enable_high_precision_timestamps")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Enables high precision timestamp sampling for profiling.
pub extern "C" fn hsa_amd_enable_high_precision_timestamps() -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        runtime::enable_high_precision_timestamps();
        SUCCESS
    })
}

/// # Safety
/// Any non-null `value` must address aligned, writable storage for the type
/// selected by `attribute`.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_system_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            return match runtime::clock_counters(&device) {
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
                SYSTEM_INFO_EXTENSIONS => write_value(
                    value,
                    extension_mask(
                        runtime
                            .gpus
                            .iter()
                            .any(|gpu| image_gfx12::supported_gpu(&gpu.info)),
                        runtime.gpus.iter().any(|gpu| gpu.pc_sampling_supported),
                    ),
                ),
                AMD_SYSTEM_INFO_SVM_SUPPORTED => write_value(value, true),
                AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT => write_value(value, false),
                AMD_SYSTEM_INFO_MWAITX_ENABLED => write_value(value, false),
                AMD_SYSTEM_INFO_DMABUF_SUPPORTED => write_value(value, true),
                AMD_SYSTEM_INFO_EXT_VERSION_MAJOR => write_value(value, 1_u16),
                AMD_SYSTEM_INFO_EXT_VERSION_MINOR => write_value(value, 36_u16),
                AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED => write_value(value, true),
                AMD_SYSTEM_INFO_XNACK_ENABLED => write_value(
                    value,
                    runtime.gpus.first().is_some_and(|gpu| gpu.xnack_enabled),
                ),
                AMD_SYSTEM_INFO_FABRIC_HANDLES_SUPPORTED => write_value(
                    value,
                    !runtime.gpus.is_empty()
                        && runtime
                            .gpus
                            .iter()
                            .all(|gpu| gpu.info.fabric_handles_supported),
                ),
                AMD_SYSTEM_INFO_HOST_ALLOC_DMABUF_SUPPORTED => {
                    write_value(value, host_alloc_dmabuf_supported(runtime.gpus.len()))
                }
                _ => INVALID_ARGUMENT,
            }
        }
    })
}

/// Returns the NUL-terminated name of a supported HSA extension.
pub(crate) fn extension_name(extension: u16) -> Option<&'static [u8]> {
    match extension {
        EXTENSION_FINALIZER => Some(b"HSA_EXTENSION_FINALIZER\0"),
        EXTENSION_IMAGES => Some(b"HSA_EXTENSION_IMAGES\0"),
        EXTENSION_PERFORMANCE_COUNTERS => Some(b"HSA_EXTENSION_PERFORMANCE_COUNTERS\0"),
        EXTENSION_PROFILING_EVENTS => Some(b"HSA_EXTENSION_PROFILING_EVENTS\0"),
        EXTENSION_AMD_PROFILER => Some(b"HSA_EXTENSION_AMD_PROFILER\0"),
        EXTENSION_AMD_LOADER => Some(b"HSA_EXTENSION_AMD_LOADER\0"),
        EXTENSION_AMD_AQLPROFILE => Some(b"HSA_EXTENSION_AMD_AQLPROFILE\0"),
        EXTENSION_AMD_PC_SAMPLING => Some(b"HSA_EXTENSION_AMD_PC_SAMPLING\0"),
        _ => None,
    }
}

/// Builds the 128-byte extension support bitmap for the current runtime.
pub(crate) fn extension_mask(images: bool, pc_sampling: bool) -> [u8; 128] {
    let mut extensions = [0_u8; 128];
    extensions[usize::from(EXTENSION_AMD_LOADER / 8)] |= 1 << (EXTENSION_AMD_LOADER % 8);
    if images {
        extensions[usize::from(EXTENSION_IMAGES / 8)] |= 1 << (EXTENSION_IMAGES % 8);
    }
    if pc_sampling {
        extensions[usize::from(EXTENSION_AMD_PC_SAMPLING / 8)] |=
            1 << (EXTENSION_AMD_PC_SAMPLING % 8);
    }
    extensions
}

/// Returns the highest supported minor version for an extension major version.
pub(crate) fn supported_extension_minor(extension: u16, version_major: u16) -> Option<u16> {
    (version_major == 1
        && matches!(
            extension,
            EXTENSION_AMD_LOADER | EXTENSION_IMAGES | EXTENSION_AMD_PC_SAMPLING
        ))
    .then_some(0)
}

/// Checks whether a legacy extension is exposed by this runtime.
pub(crate) fn legacy_extension_supported(
    extension: u16,
    version_major: u16,
    version_minor: u16,
) -> bool {
    version_minor == 0
        && match extension {
            EXTENSION_AMD_LOADER => matches!(version_major, 0 | 1),
            EXTENSION_IMAGES => version_major == 1,
            EXTENSION_AMD_PC_SAMPLING => version_major == 1,
            _ => false,
        }
}

/// Checks whether an agent exposes a legacy extension.
pub(crate) fn legacy_agent_extension_supported(
    extension: u16,
    gpu: bool,
    version_major: u16,
    version_minor: u16,
) -> bool {
    gpu && legacy_extension_supported(extension, version_major, version_minor)
}

/// Checks whether an agent exposes the requested major extension version.
pub(crate) fn major_agent_extension_supported(
    extension: u16,
    gpu: bool,
    version_major: u16,
) -> bool {
    gpu && supported_extension_minor(extension, version_major).is_some()
}

/// Checks whether an extension ID is defined by the core or AMD ABI range.
pub(crate) fn valid_extension(extension: u16) -> bool {
    extension <= EXTENSION_PROFILING_EVENTS
        || (EXTENSION_AMD_PROFILER..=EXTENSION_AMD_PC_SAMPLING).contains(&extension)
}

/// # Safety
/// Any non-null `name` must address writable storage for one C string
/// pointer.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_extension_get_name")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_extension_get_name(
    extension: u16,
    name: *mut *const c_char,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_system_extension_supported")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !valid_extension(extension) || result.is_null() {
            return INVALID_ARGUMENT;
        }
        let supported = legacy_extension_supported(extension, version_major, version_minor)
            && match extension {
                EXTENSION_IMAGES => runtime
                    .gpus
                    .iter()
                    .any(|gpu| image_gfx12::supported_gpu(&gpu.info)),
                EXTENSION_AMD_PC_SAMPLING => {
                    runtime.gpus.iter().any(|gpu| gpu.pc_sampling_supported)
                }
                _ => true,
            };
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
    unsafe(export_name = "__hsa_impl_hsa_system_major_extension_supported")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !valid_extension(extension) || version_minor.is_null() || result.is_null() {
            return INVALID_ARGUMENT;
        }
        let minor =
            supported_extension_minor(extension, version_major).filter(|_| match extension {
                EXTENSION_IMAGES => runtime
                    .gpus
                    .iter()
                    .any(|gpu| image_gfx12::supported_gpu(&gpu.info)),
                EXTENSION_AMD_PC_SAMPLING => {
                    runtime.gpus.iter().any(|gpu| gpu.pc_sampling_supported)
                }
                _ => true,
            });
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_system_get_extension_table")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
/// Any non-null `table` must address at least `table_length` writable bytes.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_system_get_major_extension_table")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if matches!(extension, EXTENSION_AMD_PROFILER | EXTENSION_AMD_AQLPROFILE) {
            return NOT_SUPPORTED;
        }
        match (extension, version_major) {
            (EXTENSION_IMAGES, 1) => {
                if !runtime
                    .gpus
                    .iter()
                    .any(|gpu| image_gfx12::supported_gpu(&gpu.info))
                {
                    return NOT_SUPPORTED;
                }
                #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
                let functions = api_trace::image_extension_table();
                #[cfg(not(all(target_os = "linux", target_arch = "x86_64", not(test))))]
                let functions = image_abi::image_extension_table();
                #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
                if functions.contains(&0) {
                    return ERROR;
                }
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
                #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
                let functions = api_trace::loader_extension_table();
                #[cfg(not(all(target_os = "linux", target_arch = "x86_64", not(test))))]
                let functions = loader::loader_extension_table();
                #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
                if functions.contains(&0) {
                    return ERROR;
                }
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
                #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
                let functions = api_trace::pc_sampling_extension_table();
                #[cfg(not(all(target_os = "linux", target_arch = "x86_64", not(test))))]
                let functions = pc_sampling::extension_table();
                #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
                if functions.contains(&0) {
                    return ERROR;
                }
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_status_string")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
