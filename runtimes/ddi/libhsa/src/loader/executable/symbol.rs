// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Executable symbol metadata, definitions, and iteration.

use super::*;

/// Binds an external variable address into an executable.
pub(crate) fn define_external_variable(
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

/// Collects executable symbol handles matching the requested agent.
pub(crate) fn executable_symbol_handles(
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
