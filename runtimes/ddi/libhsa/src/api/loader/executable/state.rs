// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA executable state C entry points.

use std::ffi::{c_char, c_void};

use crate::ffi::*;
use crate::loader::executable::state::*;
use crate::loader::*;
use crate::runtime::{boundary, initialized_mut, lock};
use rocloader::CodeObject as LoaderCodeObject;

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_create")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
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
            Err(()) => return ERROR,
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
                debugger: crate::debugger::ExecutableDebugInfo::default(),
                core: rocloader::Executable::new(state == EXECUTABLE_STATE_FROZEN),
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { executable.write(HsaExecutable { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_create_alt")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
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
            Err(()) => return ERROR,
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
                debugger: crate::debugger::ExecutableDebugInfo::default(),
                core: rocloader::Executable::new(false),
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { executable.write(HsaExecutable { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_load_agent_code_object")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
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
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_load_program_code_object")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
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
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_load_code_object")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
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
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_freeze")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
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
        let Some(record) = runtime.loader.executables.get_mut(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        if record.core.is_frozen() {
            return FROZEN_EXECUTABLE;
        }
        if !record.core.can_release() {
            return ERROR;
        }
        let Some(inflight) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        if let Err(status) = record.debugger.prepare(&record.core) {
            return status;
        }
        let publications = match record.core.begin_freeze() {
            Ok(work) => work,
            Err(error) => return map_loader_error(error),
        };
        let stop = runtime.operation_stop();
        let health = runtime.health.clone();
        drop(guard);
        // The executable remains registered and rejects mutation/destruction
        // while native copies execute without holding the HSA lifecycle lock.
        let mut status = SUCCESS;
        for publication in &publications {
            if let Err(error) = publication.publish(&stop) {
                status = map_loader_error(error);
                break;
            }
        }
        if publications
            .iter()
            .any(|publication| !publication.can_release())
        {
            health.quarantine();
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => {
                health.quarantine();
                return status;
            }
        };
        let Some(record) = guard
            .as_mut()
            .and_then(|runtime| runtime.loader.executables.get_mut(&executable.handle))
        else {
            health.quarantine();
            return INVALID_RUNTIME_STATE;
        };
        if let Err(error) = record.core.finish_freeze() {
            if status == SUCCESS {
                status = map_loader_error(error);
            }
        } else if status == SUCCESS {
            record.debugger.publish();
        }
        drop(inflight);
        status
    })
}

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_destroy")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
/// Releases an executable and its loaded code-object owners.
pub unsafe extern "C" fn hsa_executable_destroy(executable: HsaExecutable) -> Status {
    boundary(|| {
        let _mutation = match EXECUTABLE_TRAVERSAL.write() {
            Ok(guard) => guard,
            Err(()) => return ERROR,
        };
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
        if !record.core.can_release() {
            return ERROR;
        }
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
    all(hsa_tool_dispatch, not(test)),
    unsafe(
        export_name = "__hsa_impl_hsa_ven_amd_loader_executable_load_agent_code_object_at_address"
    )
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
/// Stages an agent code object for a caller-owned executable GPU mapping.
pub unsafe extern "C" fn hsa_ven_amd_loader_executable_load_agent_code_object_at_address(
    executable: HsaExecutable,
    agent: HsaAgent,
    reader: HsaCodeObjectReader,
    _options: *const c_char,
    address: *mut c_void,
    size: usize,
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
        if let Err(error) = record.core.can_load(false) {
            return map_loader_error(error);
        }
        let Some(index) = runtime.gpu_index(agent) else {
            return INVALID_AGENT;
        };
        if runtime.gpus[index].endpoint.local_memory_bytes == 0 {
            return INVALID_AGENT;
        }
        let Some((bytes, storage)) = runtime
            .loader
            .readers
            .get(&reader.handle)
            .map(Reader::snapshot)
        else {
            return INVALID_CODE_OBJECT_READER;
        };
        let load_size = match LoaderCodeObject::parse(&bytes).and_then(|object| object.load_size())
        {
            Ok(size) => size,
            Err(error) => return map_loader_error(error),
        };
        if address.is_null()
            || address as usize % 256 != 0
            || load_size > size
            || (address as usize).checked_add(size).is_none()
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The API caller retains the supplied GPU-writable/executable
        // mapping through executable destruction, including failed native
        // publication. No host access to that range is performed here.
        unsafe {
            load_executable_code_object_at(
                runtime,
                executable,
                agent,
                bytes,
                storage,
                false,
                Some((address as usize as u64, size)),
                loaded,
            )
        }
    })
}

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_validate")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
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
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_validate_alt")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
/// Validates an executable and writes the failing symbol if any.
pub unsafe extern "C" fn hsa_executable_validate_alt(
    executable: HsaExecutable,
    _options: *const c_char,
    result: *mut u32,
) -> Status {
    // SAFETY: Options are intentionally ignored, matching ROCr validation.
    unsafe { hsa_executable_validate(executable, result) }
}
