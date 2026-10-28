// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Executable creation, loading, publication, and destruction.

use super::*;

/// Selects the program or agent namespace for a loaded symbol.
pub(in crate::loader) fn loaded_symbol_agent(program: bool, agent: HsaAgent) -> HsaAgent {
    if program {
        HsaAgent { handle: 0 }
    } else {
        agent
    }
}

/// Loads code-object bytes and publishes a handle in the selected scope.
pub(crate) unsafe fn load_executable_code_object(
    runtime: &mut HsaState,
    executable: HsaExecutable,
    agent: HsaAgent,
    bytes: Arc<[u8]>,
    storage: ReaderStorage,
    program: bool,
    loaded: *mut HsaLoadedCodeObject,
) -> Status {
    let Some(executable_record) = runtime.loader.executables.get(&executable.handle) else {
        return INVALID_EXECUTABLE;
    };
    if let Err(error) = executable_record.core.can_load(program) {
        return map_loader_error(error);
    }
    let (mut object, symbols) =
        match parse_and_load(runtime, executable, agent, bytes, storage, program) {
            Ok(result) => result,
            Err(status) => return status,
        };
    let loaded_handle = match runtime.allocate_handle() {
        Ok(handle) => handle,
        Err(status) => return status,
    };
    object.metadata.handle = loaded_handle;
    let mut handles = Vec::with_capacity(symbols.len());
    for _ in &symbols {
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        handles.push(handle);
    }

    let symbol_agent = loaded_symbol_agent(program, agent);
    let scope = if program {
        Scope::Program
    } else {
        Scope::Agent(agent.handle)
    };
    let mut definitions = Vec::with_capacity(symbols.len());
    let mut published = Vec::with_capacity(symbols.len());
    for (symbol, handle) in symbols.into_iter().zip(handles) {
        definitions.push(Definition {
            symbol,
            scope,
            id: handle,
        });
        published.push((
            handle,
            SymbolReference {
                executable: executable.handle,
                agent: symbol_agent,
            },
        ));
    }
    let Some(record) = runtime.loader.executables.get_mut(&executable.handle) else {
        return INVALID_EXECUTABLE;
    };
    if let Err(error) = record.core.commit(program, object, definitions) {
        return map_loader_error(error);
    }
    for (handle, symbol) in published {
        runtime.loader.symbols.insert(handle, symbol);
    }
    if !loaded.is_null() {
        // SAFETY: A non-null optional output points to writable storage.
        unsafe {
            loaded.write(HsaLoadedCodeObject {
                handle: loaded_handle,
            })
        };
    }
    SUCCESS
}
