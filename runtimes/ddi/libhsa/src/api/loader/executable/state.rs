// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA executable state C entry points.

use std::ffi::c_char;

use crate::ffi::*;
use crate::loader::code_object::*;
use crate::loader::executable::state::*;
use crate::loader::*;
use crate::runtime::{boundary, initialized_mut, lock};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates a mutable executable with the selected profile.
pub unsafe extern "C" fn hsa_executable_create(
    profile: u32,
    state: u32,
    _options: *const c_char,
    executable: *mut HsaExecutable,
) -> Status {
    boundary(|| {
        let _mutation = match EXECUTABLE_TRAVERSAL.write() {
            Ok(guard) => guard,
            Err(_) => return ERROR,
        };
        if executable.is_null()
            || !matches!(profile, PROFILE_BASE | PROFILE_FULL)
            || !matches!(state, EXECUTABLE_STATE_UNFROZEN | EXECUTABLE_STATE_FROZEN)
        {
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
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.loader.executables.insert(
            handle,
            Executable {
                profile,
                default_rounding: DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                core: rocloader::Executable::new(state == EXECUTABLE_STATE_FROZEN),
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { executable.write(HsaExecutable { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_create_alt")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates a mutable executable using the alternate ABI signature.
pub unsafe extern "C" fn hsa_executable_create_alt(
    profile: u32,
    rounding: u32,
    _options: *const c_char,
    executable: *mut HsaExecutable,
) -> Status {
    boundary(|| {
        let _mutation = match EXECUTABLE_TRAVERSAL.write() {
            Ok(guard) => guard,
            Err(_) => return ERROR,
        };
        if executable.is_null()
            || !matches!(profile, PROFILE_BASE | PROFILE_FULL)
            || !matches!(
                rounding,
                DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT
                    | DEFAULT_FLOAT_ROUNDING_MODE_ZERO
                    | DEFAULT_FLOAT_ROUNDING_MODE_NEAR
            )
        {
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
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.loader.executables.insert(
            handle,
            Executable {
                profile,
                default_rounding: rounding,
                core: rocloader::Executable::new(false),
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { executable.write(HsaExecutable { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_load_agent_code_object")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Loads a code object into one agent’s executable scope.
pub unsafe extern "C" fn hsa_executable_load_agent_code_object(
    executable: HsaExecutable,
    agent: HsaAgent,
    reader: HsaCodeObjectReader,
    _options: *const c_char,
    loaded: *mut HsaLoadedCodeObject,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(record) = runtime.loader.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        if record.core.is_frozen() {
            return FROZEN_EXECUTABLE;
        }
        let Some((bytes, storage)) = runtime
            .loader
            .readers
            .get(&reader.handle)
            .map(Reader::snapshot)
        else {
            return INVALID_CODE_OBJECT_READER;
        };
        // SAFETY: The optional loaded-code-object output follows the public ABI.
        unsafe {
            load_executable_code_object(runtime, executable, agent, bytes, storage, false, loaded)
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_load_program_code_object")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Loads a code object into program-wide executable scope.
pub unsafe extern "C" fn hsa_executable_load_program_code_object(
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    _options: *const c_char,
    loaded: *mut HsaLoadedCodeObject,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(record) = runtime.loader.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        if record.core.is_frozen() {
            return FROZEN_EXECUTABLE;
        }
        let Some((bytes, storage)) = runtime
            .loader
            .readers
            .get(&reader.handle)
            .map(Reader::snapshot)
        else {
            return INVALID_CODE_OBJECT_READER;
        };
        // SAFETY: The optional loaded-code-object output follows the public ABI.
        unsafe {
            load_executable_code_object(
                runtime,
                executable,
                HsaAgent { handle: 0 },
                bytes,
                storage,
                true,
                loaded,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_load_code_object")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Loads a code object into the requested executable scope.
pub unsafe extern "C" fn hsa_executable_load_code_object(
    executable: HsaExecutable,
    agent: HsaAgent,
    code_object: HsaCodeObject,
    _options: *const c_char,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(record) = runtime.loader.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        if record.core.is_frozen() {
            return FROZEN_EXECUTABLE;
        }
        let Some(bytes) = runtime
            .loader
            .code_objects
            .get(&code_object.handle)
            .map(|code_object| code_object.bytes.clone())
        else {
            return INVALID_CODE_OBJECT;
        };
        // SAFETY: The deprecated entry point does not request a loaded handle.
        unsafe {
            load_executable_code_object(
                runtime,
                executable,
                agent,
                bytes,
                ReaderStorage::Memory,
                false,
                std::ptr::null_mut(),
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_freeze")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Finalizes an executable so its symbols can be queried and used.
pub unsafe extern "C" fn hsa_executable_freeze(
    executable: HsaExecutable,
    _options: *const c_char,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(executable) = runtime.loader.executables.get_mut(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        match executable.core.freeze() {
            Ok(()) => SUCCESS,
            Err(error) => map_loader_error(error),
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases an executable and its loaded code-object owners.
pub unsafe extern "C" fn hsa_executable_destroy(executable: HsaExecutable) -> Status {
    boundary(|| {
        let _mutation = match EXECUTABLE_TRAVERSAL.write() {
            Ok(guard) => guard,
            Err(_) => return ERROR,
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(executable) = runtime.loader.executables.remove(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        for &handle in executable.core.symbol_ids() {
            runtime.loader.symbols.remove(&handle);
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_validate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Validates that an executable is ready for use.
pub unsafe extern "C" fn hsa_executable_validate(
    executable: HsaExecutable,
    result: *mut u32,
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
        if !runtime.loader.executables.contains_key(&executable.handle) {
            return INVALID_EXECUTABLE;
        }
        // Every accepted code object has already passed structural validation
        // and relocation while being loaded.
        unsafe { result.write(0) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_validate_alt")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Validates an executable and writes the failing symbol if any.
pub unsafe extern "C" fn hsa_executable_validate_alt(
    executable: HsaExecutable,
    _options: *const c_char,
    result: *mut u32,
) -> Status {
    // SAFETY: Options are intentionally ignored, matching ROCr validation.
    unsafe { hsa_executable_validate(executable, result) }
}
