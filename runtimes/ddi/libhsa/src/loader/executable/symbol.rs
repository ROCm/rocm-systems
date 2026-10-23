// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Executable symbol metadata, definitions, and iteration.

use super::*;

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the requested executable attribute.
pub unsafe extern "C" fn hsa_executable_get_info(
    executable: HsaExecutable,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(executable) = guard
            .as_ref()
            .and_then(|runtime| runtime.loader.executables.get(&executable.handle))
        else {
            return if guard.as_ref().is_some() {
                INVALID_EXECUTABLE
            } else {
                NOT_INITIALIZED
            };
        };
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                EXECUTABLE_INFO_PROFILE => value.cast::<u32>().write(executable.profile),
                EXECUTABLE_INFO_STATE => {
                    value.cast::<u32>().write(if executable.core.is_frozen() {
                        EXECUTABLE_STATE_FROZEN
                    } else {
                        EXECUTABLE_STATE_UNFROZEN
                    })
                }
                EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE => {
                    value.cast::<u32>().write(executable.default_rounding);
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_get_symbol_by_name")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Finds an executable symbol by its name and agent.
pub unsafe extern "C" fn hsa_executable_get_symbol_by_name(
    executable: HsaExecutable,
    name: *const c_char,
    agent: *const HsaAgent,
    symbol: *mut HsaExecutableSymbol,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if name.is_null() || symbol.is_null() {
            return INVALID_ARGUMENT;
        }
        let Some(executable) = runtime.loader.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        // SAFETY: The ABI requires a NUL-terminated symbol name.
        let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
            return INVALID_ARGUMENT;
        };
        let requested_agent = if agent.is_null() {
            HsaAgent { handle: 0 }
        } else {
            // SAFETY: A non-null optional agent points to one HsaAgent.
            unsafe { *agent }
        };
        let scope = if requested_agent.handle == 0 {
            Scope::Program
        } else {
            Scope::Agent(requested_agent.handle)
        };
        let Some(handle) = executable.core.symbol_id(name, scope) else {
            return INVALID_SYMBOL_NAME;
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { symbol.write(HsaExecutableSymbol { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_symbol_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the requested executable-symbol attribute.
pub unsafe extern "C" fn hsa_executable_symbol_get_info(
    symbol: HsaExecutableSymbol,
    attribute: u32,
    value: *mut c_void,
) -> Status {
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
        let Some(reference) = runtime.loader.symbols.get(&symbol.handle) else {
            return INVALID_EXECUTABLE_SYMBOL;
        };
        let Some(executable) = runtime.loader.executables.get(&reference.executable) else {
            return INVALID_EXECUTABLE_SYMBOL;
        };
        let Some(symbol) = executable.core.symbol(symbol.handle) else {
            return INVALID_EXECUTABLE_SYMBOL;
        };
        let frozen = executable.core.is_frozen();
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                SYMBOL_INFO_TYPE => value.cast::<u32>().write(hsa_symbol_kind(symbol.kind)),
                SYMBOL_INFO_NAME_LENGTH => {
                    value
                        .cast::<u32>()
                        .write(u32::try_from(symbol.name.len()).unwrap_or(u32::MAX));
                }
                SYMBOL_INFO_NAME => {
                    std::ptr::copy_nonoverlapping(
                        symbol.name.as_ptr(),
                        value.cast::<u8>(),
                        symbol.name.len(),
                    );
                }
                SYMBOL_INFO_MODULE_NAME_LENGTH => value
                    .cast::<u32>()
                    .write(u32::try_from(symbol.module_name.len()).unwrap_or(u32::MAX)),
                SYMBOL_INFO_MODULE_NAME => std::ptr::copy_nonoverlapping(
                    symbol.module_name.as_ptr(),
                    value.cast::<u8>(),
                    symbol.module_name.len(),
                ),
                SYMBOL_INFO_LINKAGE => value
                    .cast::<u32>()
                    .write(hsa_symbol_linkage(symbol.linkage)),
                SYMBOL_INFO_IS_DEFINITION => value.cast::<bool>().write(symbol.is_definition),
                SYMBOL_INFO_AGENT => value.cast::<HsaAgent>().write(reference.agent),
                SYMBOL_INFO_VARIABLE_ALLOCATION => value
                    .cast::<u32>()
                    .write(hsa_variable_allocation(symbol.allocation)),
                SYMBOL_INFO_VARIABLE_SEGMENT => value
                    .cast::<u32>()
                    .write(hsa_variable_segment(symbol.segment)),
                SYMBOL_INFO_VARIABLE_ALIGNMENT => value.cast::<u32>().write(symbol.alignment),
                SYMBOL_INFO_VARIABLE_SIZE => value.cast::<u32>().write(symbol.size),
                SYMBOL_INFO_VARIABLE_IS_CONST => value.cast::<bool>().write(symbol.is_const),
                SYMBOL_INFO_VARIABLE_ADDRESS
                | SYMBOL_INFO_KERNEL_OBJECT
                | SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT => {
                    value
                        .cast::<u64>()
                        .write(if frozen { symbol.address } else { 0 });
                }
                SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE => {
                    value.cast::<u32>().write(symbol.kernarg_size);
                }
                SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT => {
                    value.cast::<u32>().write(symbol.kernarg_alignment);
                }
                SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE => {
                    value.cast::<u32>().write(symbol.group_size);
                }
                SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE => {
                    value.cast::<u32>().write(symbol.private_size);
                }
                SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK => {
                    value.cast::<bool>().write(symbol.dynamic_callstack);
                }
                SYMBOL_INFO_KERNEL_CALL_CONVENTION
                | SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION => {
                    value.cast::<u32>().write(symbol.call_convention);
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_get_symbol")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Finds an executable symbol by its qualified name.
pub unsafe extern "C" fn hsa_executable_get_symbol(
    executable: HsaExecutable,
    module_name: *const c_char,
    symbol_name: *const c_char,
    agent: HsaAgent,
    _call_convention: i32,
    symbol: *mut HsaExecutableSymbol,
) -> Status {
    boundary(|| {
        if symbol_name.is_null() || symbol.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires NUL-terminated symbol and optional module names.
        let Ok(symbol_name) = (unsafe { CStr::from_ptr(symbol_name) }).to_str() else {
            return INVALID_ARGUMENT;
        };
        let module_name = if module_name.is_null() {
            ""
        } else {
            let Ok(module_name) = (unsafe { CStr::from_ptr(module_name) }).to_str() else {
                return INVALID_ARGUMENT;
            };
            module_name
        };
        let name = if module_name.is_empty() {
            symbol_name.to_owned()
        } else {
            format!("{module_name}::{symbol_name}")
        };
        if name.is_empty() {
            return INVALID_SYMBOL_NAME;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(executable) = runtime.loader.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        let handle = executable
            .core
            .symbol_id(&name, Scope::Program)
            .or_else(|| executable.core.symbol_id(&name, Scope::Agent(agent.handle)));
        let Some(handle) = handle else {
            return INVALID_SYMBOL_NAME;
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { symbol.write(HsaExecutableSymbol { handle }) };
        SUCCESS
    })
}

/// Binds an external variable address into an executable.
fn define_external_variable(
    runtime: &mut HsaState,
    executable: HsaExecutable,
    agent: HsaAgent,
    name: &str,
    address: *mut c_void,
    allocation: VariableAllocation,
    segment: VariableSegment,
) -> Status {
    let Some(record) = runtime.loader.executables.get(&executable.handle) else {
        return INVALID_EXECUTABLE;
    };
    if record.core.is_frozen() {
        return FROZEN_EXECUTABLE;
    }
    let scope = if agent.handle == 0 {
        Scope::Program
    } else {
        Scope::Agent(agent.handle)
    };
    if record.core.symbol_id(name, scope).is_some() {
        return VARIABLE_ALREADY_DEFINED;
    }
    let handle = match runtime.allocate_handle() {
        Ok(handle) => handle,
        Err(status) => return status,
    };
    let Some(record) = runtime.loader.executables.get_mut(&executable.handle) else {
        return INVALID_EXECUTABLE;
    };
    if let Err(error) = record.core.define_external(Definition {
        symbol: rocloader::Symbol::external_variable(name, address as u64, allocation, segment),
        scope,
        id: handle,
    }) {
        return map_loader_error(error);
    }
    runtime.loader.symbols.insert(
        handle,
        SymbolReference {
            executable: executable.handle,
            agent,
        },
    );
    SUCCESS
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_global_variable_define")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Binds a program-global variable to host storage.
pub unsafe extern "C" fn hsa_executable_global_variable_define(
    executable: HsaExecutable,
    name: *const c_char,
    address: *mut c_void,
) -> Status {
    boundary(|| {
        if name.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires a NUL-terminated variable name.
        let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
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
        define_external_variable(
            runtime,
            executable,
            HsaAgent { handle: 0 },
            name,
            address,
            VariableAllocation::Program,
            VariableSegment::Global,
        )
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_agent_global_variable_define")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Binds an agent-global variable to host storage.
pub unsafe extern "C" fn hsa_executable_agent_global_variable_define(
    executable: HsaExecutable,
    agent: HsaAgent,
    name: *const c_char,
    address: *mut c_void,
) -> Status {
    boundary(|| {
        if name.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires a NUL-terminated variable name.
        let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
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
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        define_external_variable(
            runtime,
            executable,
            agent,
            name,
            address,
            VariableAllocation::Agent,
            VariableSegment::Global,
        )
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_readonly_variable_define")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Binds a read-only variable to host storage.
pub unsafe extern "C" fn hsa_executable_readonly_variable_define(
    executable: HsaExecutable,
    agent: HsaAgent,
    name: *const c_char,
    address: *mut c_void,
) -> Status {
    boundary(|| {
        if name.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires a NUL-terminated variable name.
        let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
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
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        define_external_variable(
            runtime,
            executable,
            agent,
            name,
            address,
            VariableAllocation::Agent,
            VariableSegment::Readonly,
        )
    })
}

/// Collects executable symbol handles matching the requested agent.
fn executable_symbol_handles(
    runtime: &HsaState,
    executable: HsaExecutable,
    agent: Option<HsaAgent>,
) -> Result<Vec<u64>, Status> {
    let Some(executable) = runtime.loader.executables.get(&executable.handle) else {
        return Err(INVALID_EXECUTABLE);
    };
    Ok(executable
        .core
        .symbol_ids()
        .iter()
        .copied()
        .filter(|handle| {
            runtime.loader.symbols.get(handle).is_some_and(|symbol| {
                agent.map_or(symbol.agent.handle == 0, |agent| symbol.agent == agent)
            })
        })
        .collect())
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_iterate_symbols")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Visits symbols in a live executable.
pub unsafe extern "C" fn hsa_executable_iterate_symbols(
    executable: HsaExecutable,
    callback: ExecutableSymbolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let handles = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(record) = runtime.loader.executables.get(&executable.handle) else {
                return INVALID_EXECUTABLE;
            };
            record.core.symbol_ids().to_vec()
        };
        for handle in handles {
            // SAFETY: Traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(executable, HsaExecutableSymbol { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_iterate_agent_symbols")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Visits executable symbols for one agent.
pub unsafe extern "C" fn hsa_executable_iterate_agent_symbols(
    executable: HsaExecutable,
    agent: HsaAgent,
    callback: AgentExecutableSymbolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let handles = {
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
            match executable_symbol_handles(runtime, executable, Some(agent)) {
                Ok(handles) => handles,
                Err(status) => return status,
            }
        };
        for handle in handles {
            // SAFETY: Traversal callbacks are synchronous and data stays live.
            let status =
                unsafe { callback(executable, agent, HsaExecutableSymbol { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_iterate_program_symbols")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Visits executable symbols in program scope.
pub unsafe extern "C" fn hsa_executable_iterate_program_symbols(
    executable: HsaExecutable,
    callback: ExecutableSymbolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let handles = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            match executable_symbol_handles(runtime, executable, None) {
                Ok(handles) => handles,
                Err(status) => return status,
            }
        };
        for handle in handles {
            // SAFETY: Traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(executable, HsaExecutableSymbol { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}
