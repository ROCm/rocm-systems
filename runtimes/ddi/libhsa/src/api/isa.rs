// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! ISA and wavefront handle queries exposed through the HSA C ABI.

use std::ffi::{CStr, c_char, c_void};

use super::output::write_value;
use crate::ffi::*;
use crate::isa::{isa_handle, isa_name, wavefront_handle};
use crate::runtime::{boundary, lock};

/// # Safety
/// The callback must be callable with the C ABI; `data` must remain valid for
/// any access the callback performs during this call.
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_agent_iterate_isas")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_agent_iterate_isas(
    agent: HsaAgent,
    callback: IsaCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let (index, info, xnack_enabled) = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(index) = runtime.gpu_index(agent) else {
                return if runtime.is_cpu_agent(agent) {
                    SUCCESS
                } else {
                    INVALID_AGENT
                };
            };
            let info = runtime.gpus[index].info();
            (index, info, runtime.gpus[index].xnack_enabled)
        };
        for variant in 0..ISA_COUNT_PER_GPU {
            if isa_name(&info, xnack_enabled, variant).is_none() {
                continue;
            }
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_isa_from_name")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
                let Some(candidate) = isa_name(&gpu.info(), gpu.xnack_enabled, variant) else {
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_isa_get_info_alt")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_isa_get_info_alt(
    isa: HsaIsa,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let (info, xnack_enabled, variant) = {
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
            (
                runtime.gpus[index].info(),
                runtime.gpus[index].xnack_enabled,
                variant,
            )
        };
        let Some(name) = isa_name(&info, xnack_enabled, variant) else {
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_isa_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_isa_get_info(
    isa: HsaIsa,
    attribute: u32,
    index: u32,
    value: *mut c_void,
) -> Status {
    if !matches!(attribute, 3 | 4) {
        // SAFETY: This deprecated entry point shares the public output contract
        // of hsa_isa_get_info_alt for non-call-convention attributes.
        return unsafe { hsa_isa_get_info_alt(isa, attribute, value) };
    }
    if index != 0 {
        return INVALID_INDEX;
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_isa_get_exception_policies")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_isa_get_round_method")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_isa_iterate_wavefronts")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_wavefront_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            runtime.gpus[index].info().wavefront_size
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
#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_isa_compatible")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some((code_index, code_variant)) = runtime.isa_parts(code_object_isa) else {
            return INVALID_ISA;
        };
        let Some((agent_index, agent_variant)) = runtime.isa_parts(agent_isa) else {
            return INVALID_ISA;
        };
        let code = &runtime.gpus[code_index];
        let agent = &runtime.gpus[agent_index];
        let code_name = isa_name(&code.info(), code.xnack_enabled, code_variant);
        let agent_name = isa_name(&agent.info(), agent.xnack_enabled, agent_variant);
        let compatible = code_name == agent_name
            || (code_variant == 1
                && agent_variant == 0
                && code_name == isa_name(&agent.info(), agent.xnack_enabled, 1));
        // SAFETY: The caller supplied writable output storage.
        unsafe { result.write(compatible) };
        SUCCESS
    })
}
