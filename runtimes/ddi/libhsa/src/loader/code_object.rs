// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Code-object parsing, symbols, and legacy HSA code-object calls.

use super::*;
use crate::isa::{isa_handle, isa_name};

/// HSA code-symbol metadata staged before receiving an opaque handle.
struct ParsedCodeSymbol {
    /// Qualified name including the module when present.
    full_name: String,
    /// Unqualified symbol name.
    name: String,
    /// Module namespace containing the symbol.
    module_name: String,
    /// Size of the variable or function object in bytes.
    size: u32,
    /// HSA discriminant for variable, kernel, or indirect function.
    kind: u32,
    /// HSA module or program linkage discriminant.
    linkage: u32,
    /// Whether this record defines the symbol.
    is_definition: bool,
    /// HSA allocation scope for a variable symbol.
    allocation: u32,
    /// HSA memory segment for a variable symbol.
    segment: u32,
    /// Required alignment of the variable in bytes.
    alignment: u32,
    /// Whether the variable is immutable.
    is_const: bool,
    /// Kernel-argument segment size in bytes.
    kernarg_size: u32,
    /// Kernel-argument segment alignment in bytes.
    kernarg_alignment: u32,
    /// Static group-memory requirement in bytes.
    group_size: u32,
    /// Private-memory requirement in bytes per work-item.
    private_size: u32,
    /// Whether the kernel uses a dynamic call stack.
    dynamic_callstack: bool,
    /// Calling-convention value for the function symbol.
    call_convention: u32,
    /// Number of work-items in a wavefront for this kernel.
    wavefront_size: u32,
}

/// Maps loader error to the corresponding HSA representation.
pub(crate) fn map_loader_error(error: rocloader::Error) -> Status {
    match error {
        rocloader::Error::InvalidObject => INVALID_CODE_OBJECT,
        rocloader::Error::UndefinedSymbol => VARIABLE_UNDEFINED,
        rocloader::Error::InvalidTarget => INVALID_ISA_NAME,
        rocloader::Error::IncompatibleTarget | rocloader::Error::ProgramAlreadyLoaded => {
            INCOMPATIBLE_ARGUMENTS
        }
        rocloader::Error::OutOfResources => OUT_OF_RESOURCES,
        rocloader::Error::Device(error) => map_error(error),
        rocloader::Error::InvalidRange => INVALID_ALLOCATION,
        rocloader::Error::Frozen => FROZEN_EXECUTABLE,
        rocloader::Error::AlreadyDefined => VARIABLE_ALREADY_DEFINED,
    }
}

/// Maps a parsed symbol kind to its HSA ABI discriminant.
pub(crate) fn hsa_symbol_kind(kind: SymbolKind) -> u32 {
    match kind {
        SymbolKind::Kernel => SYMBOL_KIND_KERNEL,
        SymbolKind::Variable => SYMBOL_KIND_VARIABLE,
        SymbolKind::IndirectFunction => SYMBOL_KIND_INDIRECT_FUNCTION,
    }
}

/// Maps parsed symbol linkage to its HSA ABI discriminant.
pub(crate) fn hsa_symbol_linkage(linkage: SymbolLinkage) -> u32 {
    match linkage {
        SymbolLinkage::Module => SYMBOL_LINKAGE_MODULE,
        SymbolLinkage::Program => SYMBOL_LINKAGE_PROGRAM,
    }
}

/// Maps variable allocation scope to its HSA ABI value.
pub(crate) fn hsa_variable_allocation(allocation: VariableAllocation) -> u32 {
    match allocation {
        VariableAllocation::Agent => VARIABLE_ALLOCATION_AGENT,
        VariableAllocation::Program => VARIABLE_ALLOCATION_PROGRAM,
    }
}

/// Maps a variable segment to its HSA ABI value.
pub(crate) fn hsa_variable_segment(segment: VariableSegment) -> u32 {
    match segment {
        VariableSegment::Global => VARIABLE_SEGMENT_GLOBAL,
        VariableSegment::Readonly => VARIABLE_SEGMENT_READONLY,
    }
}

#[cfg(test)]
/// Parses ELF layout through the production code-object parser for tests.
fn parse_elf_layout(bytes: &[u8]) -> Result<ElfLayout, Status> {
    LoaderCodeObject::parse(bytes)
        .map(|object| object.layout())
        .map_err(map_loader_error)
}

impl From<rocloader::Symbol> for ParsedCodeSymbol {
    /// Converts parsed loader metadata to HSA symbol query values.
    fn from(symbol: rocloader::Symbol) -> Self {
        Self {
            full_name: symbol.full_name,
            name: symbol.name,
            module_name: symbol.module_name,
            size: symbol.size,
            kind: hsa_symbol_kind(symbol.kind),
            linkage: hsa_symbol_linkage(symbol.linkage),
            is_definition: symbol.is_definition,
            allocation: hsa_variable_allocation(symbol.allocation),
            segment: hsa_variable_segment(symbol.segment),
            alignment: symbol.alignment,
            is_const: symbol.is_const,
            kernarg_size: symbol.kernarg_size,
            kernarg_alignment: symbol.kernarg_alignment,
            group_size: symbol.group_size,
            private_size: symbol.private_size,
            dynamic_callstack: symbol.dynamic_callstack,
            call_convention: symbol.call_convention,
            wavefront_size: symbol.wavefront_size,
        }
    }
}

#[cfg(test)]
/// Parses symbols with the requested section-table precedence for tests.
fn parse_symbols(
    bytes: &[u8],
    _layout: ElfLayout,
    preferred_tables: &[u32],
) -> Result<Vec<ParsedCodeSymbol>, Status> {
    let order = if preferred_tables.first() == Some(&SECTION_DYNSYM) {
        SymbolTableOrder::DynamicFirst
    } else {
        SymbolTableOrder::RegularFirst
    };
    LoaderCodeObject::parse(bytes)
        .and_then(|object| object.symbols(order))
        .map(|symbols| symbols.into_iter().map(ParsedCodeSymbol::from).collect())
        .map_err(map_loader_error)
}

#[cfg(test)]
/// Applies AMDGPU relocations to a test image using the production loader.
fn apply_relocations(
    bytes: &[u8],
    layout: ElfLayout,
    virtual_base: u64,
    device_base: u64,
    image: &mut [u8],
    resolve_external: impl Fn(&str) -> Option<u64>,
) -> Result<(), Status> {
    rocelf::apply_amdgpu_relocations(
        bytes,
        layout,
        virtual_base,
        device_base,
        image,
        resolve_external,
    )
    .map_err(|error| map_loader_error(error.into()))
}

/// Resolves executable symbol to a validated runtime resource.
fn resolve_executable_symbol(
    runtime: &HsaState,
    executable: HsaExecutable,
    agent: HsaAgent,
    name: &str,
) -> Option<u64> {
    runtime
        .loader
        .executables
        .get(&executable.handle)?
        .core
        .resolve(name, agent.handle)
}

/// Parses a code object and loads its segments for the selected agent.
pub(super) fn parse_and_load(
    runtime: &HsaState,
    executable: HsaExecutable,
    agent: HsaAgent,
    code_object: Arc<[u8]>,
    storage: ReaderStorage,
    program: bool,
) -> Result<(LoadedObject, Vec<rocloader::Symbol>), Status> {
    let object = LoaderCodeObject::parse(&code_object).map_err(map_loader_error)?;
    let (gpu_index, memory_kind, object_kind) = if program {
        let gpu_index = runtime
            .gpus
            .iter()
            .position(|gpu| object.validate_target(Target::from(&gpu.info())).is_ok())
            .ok_or(INCOMPATIBLE_ARGUMENTS)?;
        (gpu_index, MemoryKind::System, LOADER_OBJECT_KIND_PROGRAM)
    } else {
        let Some(gpu_index) = runtime.gpu_index(agent) else {
            return Err(INVALID_AGENT);
        };
        (
            gpu_index,
            MemoryKind::DeviceLocal {
                host_visible: true,
                coherent: false,
                extended_coherent: false,
                uncached: false,
                contiguous: false,
            },
            LOADER_OBJECT_KIND_AGENT,
        )
    };
    let gpu = &runtime.gpus[gpu_index];
    let uri = match storage {
        ReaderStorage::Memory => format!(
            "memory://{}#offset=0x{:x}&size={}",
            std::process::id(),
            code_object.as_ptr() as usize,
            code_object.len()
        )
        .into_bytes(),
        ReaderStorage::File { uri } => uri,
    };
    let (image, symbols) = rocloader::load_image(
        &gpu.device,
        code_object,
        memory_kind,
        runtime.host_page_size as u64,
        |name| resolve_executable_symbol(runtime, executable, agent, name),
    )
    .map_err(map_loader_error)?;
    Ok((
        LoadedObject {
            image,
            metadata: LoadedMetadata {
                handle: 0,
                agent,
                kind: object_kind,
                uri,
            },
        },
        symbols,
    ))
}

/// Returns the ISA handle required by a parsed code object.
pub(crate) fn code_object_isa(
    runtime: &HsaState,
    object: &LoaderCodeObject<'_>,
) -> Result<HsaIsa, Status> {
    let (target, variant) = object.target().map_err(map_loader_error)?;
    for (index, gpu) in runtime.gpus.iter().enumerate() {
        let Some(name) = isa_name(&gpu.info(), gpu.xnack_enabled, variant) else {
            continue;
        };
        let mut active = Target::from(&gpu.info());
        active.xnack_enabled = Some(gpu.xnack_enabled);
        if name
            .strip_prefix("amdgcn-amd-amdhsa--")
            .and_then(|name| name.split(':').next())
            == Some(target)
            && object.validate_target(active).is_ok()
        {
            return Ok(isa_handle(index, variant));
        }
    }
    Err(INVALID_ISA_NAME)
}

/// Ensures code object symbols exists before use.
pub(crate) fn ensure_code_object_symbols(
    runtime: &mut HsaState,
    handle: u64,
) -> Result<(), Status> {
    let Some(code_object) = runtime.loader.code_objects.get(&handle) else {
        return Err(INVALID_CODE_OBJECT);
    };
    if code_object.symbols_initialized {
        return Ok(());
    }
    let bytes = code_object.bytes.clone();
    let symbols = rocloader::CodeObject::parse(&bytes)
        .map_err(map_loader_error)?
        .symbols(SymbolTableOrder::RegularFirst)
        .map_err(map_loader_error)?
        .into_iter()
        .map(ParsedCodeSymbol::from)
        .collect::<Vec<_>>();
    let mut names = HashMap::new();
    let mut handles = Vec::with_capacity(symbols.len());
    for symbol in symbols {
        let symbol_handle = runtime.allocate_handle()?;
        names.insert(symbol.full_name, symbol_handle);
        runtime.loader.code_symbols.insert(
            symbol_handle,
            CodeSymbol {
                code_object: handle,
                name: symbol.name,
                module_name: symbol.module_name,
                kind: symbol.kind,
                linkage: symbol.linkage,
                is_definition: symbol.is_definition,
                allocation: symbol.allocation,
                segment: symbol.segment,
                alignment: symbol.alignment,
                size: symbol.size,
                is_const: symbol.is_const,
                kernarg_size: symbol.kernarg_size,
                kernarg_alignment: symbol.kernarg_alignment,
                group_size: symbol.group_size,
                private_size: symbol.private_size,
                dynamic_callstack: symbol.dynamic_callstack,
                call_convention: symbol.call_convention,
                wavefront_size: symbol.wavefront_size,
            },
        );
        handles.push(symbol_handle);
    }
    let Some(code_object) = runtime.loader.code_objects.get_mut(&handle) else {
        for symbol_handle in handles {
            runtime.loader.code_symbols.remove(&symbol_handle);
        }
        return Err(INVALID_CODE_OBJECT);
    };
    code_object.names = names;
    code_object.symbol_handles = handles;
    code_object.symbols_initialized = true;
    Ok(())
}

/// Looks up a code-object symbol and writes its HSA handle.
pub(crate) unsafe fn code_object_get_symbol_impl(
    code_object: HsaCodeObject,
    module_name: *const c_char,
    symbol_name: *const c_char,
    symbol: *mut HsaCodeSymbol,
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
        let full_name = if module_name.is_empty() {
            symbol_name.to_owned()
        } else {
            format!("{module_name}::{symbol_name}")
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if let Err(status) = ensure_code_object_symbols(runtime, code_object.handle) {
            return status;
        }
        let Some(&handle) = runtime
            .loader
            .code_objects
            .get(&code_object.handle)
            .and_then(|code_object| code_object.names.get(&full_name))
        else {
            return INVALID_SYMBOL_NAME;
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { symbol.write(HsaCodeSymbol { handle }) };
        SUCCESS
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
