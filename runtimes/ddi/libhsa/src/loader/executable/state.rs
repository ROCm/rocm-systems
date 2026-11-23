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
    runtime: &mut HsaSession,
    executable: HsaExecutable,
    agent: HsaAgent,
    bytes: Arc<[u8]>,
    storage: ReaderStorage,
    program: bool,
    loaded: *mut HsaLoadedCodeObject,
) -> Status {
    // SAFETY: Ordinary loads do not borrow an application destination.
    unsafe {
        load_executable_code_object_at(
            runtime, executable, agent, bytes, storage, program, None, loaded,
        )
    }
}

/// Loads an ordinary image or retains an application-owned destination.
#[allow(
    clippy::too_many_arguments,
    reason = "the loader adapter preserves the C entry point arguments and adds optional destination ownership"
)]
pub(crate) unsafe fn load_executable_code_object_at(
    runtime: &mut HsaSession,
    executable: HsaExecutable,
    agent: HsaAgent,
    bytes: Arc<[u8]>,
    storage: ReaderStorage,
    program: bool,
    destination: Option<(u64, usize)>,
    loaded: *mut HsaLoadedCodeObject,
) -> Status {
    let Some(executable_record) = runtime.loader.executables.get(&executable.handle) else {
        return INVALID_EXECUTABLE;
    };
    if let Err(error) = executable_record.core.can_load(program) {
        return map_loader_error(error);
    }
    let (mut object, symbols) =
        // SAFETY: The caller retains any supplied destination through the
        // executable's lifetime and uncertain native publication.
        match unsafe { parse_and_load(runtime, executable, agent, bytes, storage, program, destination) } {
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

/// Publishes an executable while retaining its HSA generation outside the lock.
///
/// The loader reserves the freeze transition before native publication. Public
/// mutation and destruction stay blocked until that work conclusively retires.
pub(crate) fn freeze_executable(executable: HsaExecutable) -> Result<(), Status> {
    let mut guard = lock()?;
    let runtime = initialized_mut(&mut guard)?;
    let record = runtime
        .loader
        .executables
        .get_mut(&executable.handle)
        .ok_or(INVALID_EXECUTABLE)?;
    if record.core.is_frozen() {
        return Err(FROZEN_EXECUTABLE);
    }
    if !record.core.can_release() {
        return Err(ERROR);
    }
    let inflight = runtime.control.inflight.enter().ok_or(OUT_OF_RESOURCES)?;
    record.debugger.prepare(&record.core)?;
    let publications = record.core.begin_freeze().map_err(map_loader_error)?;
    let stop = runtime.control.operation_stop();
    let health = runtime.control.health.clone();
    drop(guard);

    let published = publications
        .iter()
        .try_for_each(|publication| publication.publish(&stop).map_err(map_loader_error));
    if publications
        .iter()
        .any(|publication| !publication.can_release())
    {
        health.quarantine();
    }
    let mut guard = lock().inspect_err(|_| health.quarantine())?;
    let record = guard
        .as_mut()
        .and_then(|runtime| runtime.loader.executables.get_mut(&executable.handle))
        .ok_or_else(|| {
            health.quarantine();
            INVALID_RUNTIME_STATE
        })?;
    // Complete even a retired failure so the caller may retry. Preserve the
    // original publication error if completing the transition also fails.
    let finished = record.core.finish_freeze().map_err(map_loader_error);
    let result = published.and(finished);
    if result.is_ok() {
        record.debugger.publish();
    }
    drop(inflight);
    result
}

/// Removes public symbol reachability before releasing an executable owner.
pub(crate) fn destroy_executable(executable: HsaExecutable) -> Result<(), Status> {
    let _mutation = EXECUTABLE_TRAVERSAL.write().map_err(|()| ERROR)?;
    let mut guard = lock()?;
    let runtime = initialized_mut(&mut guard)?;
    let record = runtime
        .loader
        .executables
        .get(&executable.handle)
        .ok_or(INVALID_EXECUTABLE)?;
    if !record.core.can_release() {
        return Err(ERROR);
    }
    let executable = runtime
        .loader
        .executables
        .remove(&executable.handle)
        .ok_or(INVALID_EXECUTABLE)?;
    for handle in executable.core.symbol_ids() {
        runtime.loader.symbols.remove(handle);
    }
    Ok(())
}
