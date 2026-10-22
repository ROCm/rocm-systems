// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA code-object readers, executable state, and ELF loader integration.
//!
//! Readers first take an owned snapshot of caller memory or a bounded file
//! range. `rocloader` interprets AMD HSA code objects, constructs their rocddi
//! images, and owns executable publication and lifetime. This module owns HSA
//! handles, C ABI queries, and status translation.
//!
//! File readers use rocddi's Linux descriptor provider to snapshot caller
//! storage while leaving descriptor ownership with the application.

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, RwLock};

use crate::platform::memory as linux_interop;
use rocddi::memory::device::MemoryKind;
#[cfg(test)]
use rocelf::Layout as ElfLayout;
#[cfg(test)]
use rocelf::LoadSegment;
#[cfg(test)]
use rocelf::constants::*;
#[cfg(test)]
use rocelf::read_u64;
use rocloader::{
    CodeObject as LoaderCodeObject, Definition, Scope, SymbolKind, SymbolLinkage, SymbolTableOrder,
    Target, VariableAllocation, VariableSegment,
};

use crate::ffi::*;
use crate::runtime::{HsaState, boundary, initialized_mut, lock, map_error};

/// Keeps executable handles stable across loader traversal callbacks.
/// Acquire this before the lifecycle mutex; callbacks may query that mutex.
static EXECUTABLE_TRAVERSAL: RwLock<()> = RwLock::new(());

/// Origin retained for loader queries after bytes have been snapshotted.
#[derive(Clone)]
enum ReaderStorage {
    /// Snapshot from caller-owned memory.
    Memory,
    /// Snapshot from a file with a retained source URI.
    File {
        /// Source URI preserved after the descriptor is released.
        uri: Vec<u8>,
    },
}

impl ReaderStorage {
    fn from_file(descriptor: i32, offset: usize, size: usize) -> Self {
        crate::platform::code_object_file_uri(descriptor, offset, size)
            .map_or(Self::Memory, |uri| Self::File { uri })
    }
}

/// Immutable code-object byte snapshot associated with one reader handle.
pub(crate) struct Reader {
    /// Immutable snapshot independent of caller storage lifetime.
    bytes: Arc<[u8]>,
    /// Source description retained for loader queries.
    storage: ReaderStorage,
}

/// Parsed code-object metadata cached independently of any executable load.
pub(crate) struct CodeObject {
    /// Immutable ELF bytes shared with readers and executable loads.
    bytes: Arc<[u8]>,
    /// Symbol name to public handle mapping.
    names: HashMap<String, u64>,
    /// Public symbol handles in enumeration order.
    symbol_handles: Vec<u64>,
    /// Whether symbols have already been parsed and published.
    symbols_initialized: bool,
}

/// HSA-visible symbol facts belonging to one parsed code object.
pub(crate) struct CodeSymbol {
    /// Public handle of the containing code object.
    code_object: u64,
    /// Symbol name reported by HSA queries.
    name: String,
    /// Module name reported by HSA queries.
    module_name: String,
    /// HSA kernel, variable, or indirect-function kind.
    kind: u32,
    /// HSA program or module linkage.
    linkage: u32,
    /// Whether this record defines rather than references a symbol.
    is_definition: bool,
    /// HSA variable allocation scope.
    allocation: u32,
    /// HSA variable memory segment.
    segment: u32,
    /// Required variable alignment in bytes.
    alignment: u32,
    /// Symbol byte size.
    size: u32,
    /// Whether a variable is read-only.
    is_const: bool,
    /// Kernel argument segment byte size.
    kernarg_size: u32,
    /// Kernel argument segment alignment.
    kernarg_alignment: u32,
    /// Kernel group segment byte size.
    group_size: u32,
    /// Kernel private segment byte size.
    private_size: u32,
    /// Whether kernel calls may require dynamic stack space.
    dynamic_callstack: bool,
    /// Kernel calling convention identifier.
    call_convention: u32,
    /// Kernel wavefront width.
    wavefront_size: u32,
}

/// ABI metadata kept alongside an owned image in the loader executable.
pub(crate) struct LoadedMetadata {
    /// Public loaded-object handle.
    handle: u64,
    /// Agent whose device holds this loaded image.
    agent: HsaAgent,
    /// Public loaded-object type.
    kind: u32,
    /// Source URI reported to loader clients.
    uri: Vec<u8>,
}

type LoadedObject = rocloader::LoadedObject<LoadedMetadata>;

/// HSA profile and rounding policy around the reusable executable owner.
pub(crate) struct Executable {
    /// HSA profile selected at creation.
    pub(crate) profile: u32,
    /// Default floating-point rounding mode.
    pub(crate) default_rounding: u32,
    /// Loader-owned images and scoped symbol index.
    pub(crate) core: rocloader::Executable<LoadedMetadata>,
}

/// Resolves a complete range in a loaded executable to its CPU mapping.
pub(crate) fn loaded_host_range(
    loader: &LoaderState,
    address: usize,
    size: usize,
) -> Result<Option<usize>, Status> {
    for executable in loader.executables.values() {
        match executable.core.host_range(address as u64, size) {
            Ok(Some(host)) => return Ok(Some(host)),
            Ok(None) => {}
            Err(rocloader::Error::InvalidRange) => return Err(INVALID_ALLOCATION),
            Err(error) => return Err(map_loader_error(error)),
        }
    }
    Ok(None)
}

#[cfg(test)]
fn loaded_segments(
    loads: &[LoadSegment],
    device_base: u64,
    virtual_base: u64,
) -> Result<Vec<rocloader::LoadedSegment>, Status> {
    rocloader::loaded_segments(loads, device_base, virtual_base).map_err(map_loader_error)
}

/// Maps an HSA symbol handle to its loader-owned executable and agent scope.
pub(crate) struct SymbolReference {
    /// Public executable handle that owns this symbol.
    pub(crate) executable: u64,
    /// Agent scope of this symbol, if any.
    pub(crate) agent: HsaAgent,
}

/// HSA handles around code objects and rocloader-owned executables.
pub(crate) struct LoaderState {
    /// Parsed code objects indexed by public handle.
    code_objects: HashMap<u64, CodeObject>,
    /// Code-object symbols indexed by public handle.
    code_symbols: HashMap<u64, CodeSymbol>,
    /// Caller-supplied code-object readers.
    readers: HashMap<u64, Reader>,
    /// Executable owners indexed by public handle.
    executables: HashMap<u64, Executable>,
    /// Symbols published from loaded executables.
    symbols: HashMap<u64, SymbolReference>,
}

impl LoaderState {
    pub(crate) fn new() -> Self {
        Self {
            code_objects: HashMap::new(),
            code_symbols: HashMap::new(),
            readers: HashMap::new(),
            executables: HashMap::new(),
            symbols: HashMap::new(),
        }
    }

    /// Drops HSA symbol handles before their executable and reader owners.
    pub(crate) fn clear(&mut self) {
        self.symbols.clear();
        self.executables.clear();
        self.readers.clear();
        self.code_symbols.clear();
        self.code_objects.clear();
    }
}

mod code_object;
mod executable;
mod query;
mod reader;

use code_object::map_loader_error;

#[cfg(any(test, not(target_arch = "x86_64")))]
pub(crate) fn loader_extension_table() -> [usize; 7] {
    [
        query::hsa_ven_amd_loader_query_host_address as *const () as usize,
        query::hsa_ven_amd_loader_query_segment_descriptors as *const () as usize,
        query::hsa_ven_amd_loader_query_executable as *const () as usize,
        query::hsa_ven_amd_loader_executable_iterate_loaded_code_objects as *const () as usize,
        query::hsa_ven_amd_loader_loaded_code_object_get_info as *const () as usize,
        query::hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size as *const ()
            as usize,
        query::hsa_ven_amd_loader_iterate_executables as *const () as usize,
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
