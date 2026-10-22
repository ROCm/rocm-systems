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
use std::sync::Arc;

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

/// HSA code-symbol metadata staged before receiving an opaque handle.
struct ParsedCodeSymbol {
    full_name: String,
    name: String,
    module_name: String,
    size: u32,
    kind: u32,
    linkage: u32,
    is_definition: bool,
    allocation: u32,
    segment: u32,
    alignment: u32,
    is_const: bool,
    kernarg_size: u32,
    kernarg_alignment: u32,
    group_size: u32,
    private_size: u32,
    dynamic_callstack: bool,
    call_convention: u32,
    wavefront_size: u32,
}

fn map_loader_error(error: rocloader::Error) -> Status {
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

fn hsa_symbol_kind(kind: SymbolKind) -> u32 {
    match kind {
        SymbolKind::Kernel => SYMBOL_KIND_KERNEL,
        SymbolKind::Variable => SYMBOL_KIND_VARIABLE,
        SymbolKind::IndirectFunction => SYMBOL_KIND_INDIRECT_FUNCTION,
    }
}

fn hsa_symbol_linkage(linkage: SymbolLinkage) -> u32 {
    match linkage {
        SymbolLinkage::Module => SYMBOL_LINKAGE_MODULE,
        SymbolLinkage::Program => SYMBOL_LINKAGE_PROGRAM,
    }
}

fn hsa_variable_allocation(allocation: VariableAllocation) -> u32 {
    match allocation {
        VariableAllocation::Agent => VARIABLE_ALLOCATION_AGENT,
        VariableAllocation::Program => VARIABLE_ALLOCATION_PROGRAM,
    }
}

fn hsa_variable_segment(segment: VariableSegment) -> u32 {
    match segment {
        VariableSegment::Global => VARIABLE_SEGMENT_GLOBAL,
        VariableSegment::Readonly => VARIABLE_SEGMENT_READONLY,
    }
}

#[cfg(test)]
fn parse_elf_layout(bytes: &[u8]) -> Result<ElfLayout, Status> {
    LoaderCodeObject::parse(bytes)
        .map(|object| object.layout())
        .map_err(map_loader_error)
}

impl From<rocloader::Symbol> for ParsedCodeSymbol {
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

fn parse_and_load(
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
            .position(|gpu| object.validate_target(Target::from(&gpu.info)).is_ok())
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

fn code_object_isa(runtime: &HsaState, object: &LoaderCodeObject<'_>) -> Result<HsaIsa, Status> {
    let (target, variant) = object.target().map_err(map_loader_error)?;
    for (index, gpu) in runtime.gpus.iter().enumerate() {
        let Some(name) = crate::isa_name(
            gpu.info.gfx_major,
            gpu.info.gfx_minor,
            gpu.info.gfx_stepping,
            gpu.info.asic_revision,
            variant,
        ) else {
            continue;
        };
        if name.strip_prefix("amdgcn-amd-amdhsa--") == Some(target) {
            return Ok(crate::isa_handle(index, variant));
        }
    }
    Err(INVALID_ISA_NAME)
}

fn ensure_code_object_symbols(runtime: &mut HsaState, handle: u64) -> Result<(), Status> {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_deserialize")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_deserialize(
    serialized_code_object: *mut c_void,
    serialized_code_object_size: usize,
    _options: *const c_char,
    code_object: *mut HsaCodeObject,
) -> Status {
    boundary(|| {
        if serialized_code_object.is_null()
            || serialized_code_object_size == 0
            || code_object.is_null()
        {
            return INVALID_ARGUMENT;
        }
        let mut bytes = Vec::new();
        if bytes
            .try_reserve_exact(serialized_code_object_size)
            .is_err()
        {
            return OUT_OF_RESOURCES;
        }
        bytes.resize(serialized_code_object_size, 0);
        // SAFETY: The caller supplies serialized_code_object_size readable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                serialized_code_object.cast::<u8>(),
                bytes.as_mut_ptr(),
                serialized_code_object_size,
            );
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
        runtime.loader.code_objects.insert(
            handle,
            CodeObject {
                bytes: bytes.into(),
                names: HashMap::new(),
                symbol_handles: Vec::new(),
                symbols_initialized: false,
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { code_object.write(HsaCodeObject { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_serialize")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_serialize(
    code_object: HsaCodeObject,
    alloc_callback: CodeObjectAllocCallback,
    callback_data: HsaCallbackData,
    _options: *const c_char,
    serialized_code_object: *mut *mut c_void,
    serialized_code_object_size: *mut usize,
) -> Status {
    boundary(|| {
        let Some(alloc_callback) = alloc_callback else {
            return INVALID_ARGUMENT;
        };
        if serialized_code_object.is_null() || serialized_code_object_size.is_null() {
            return INVALID_ARGUMENT;
        }
        let bytes = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(code_object) = runtime.loader.code_objects.get(&code_object.handle) else {
                return INVALID_CODE_OBJECT;
            };
            code_object.bytes.clone()
        };
        // SAFETY: The callback and output storage are supplied by the caller.
        let status = unsafe { alloc_callback(bytes.len(), callback_data, serialized_code_object) };
        if status != SUCCESS {
            return status;
        }
        // SAFETY: A successful callback must initialize the output pointer.
        let destination = unsafe { serialized_code_object.read() };
        if destination.is_null() {
            return OUT_OF_RESOURCES;
        }
        // SAFETY: The callback allocated at least bytes.len() writable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination.cast::<u8>(), bytes.len());
            serialized_code_object_size.write(bytes.len());
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_destroy(code_object: HsaCodeObject) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(code_object) = runtime.loader.code_objects.remove(&code_object.handle) else {
            return INVALID_CODE_OBJECT;
        };
        for handle in code_object.symbol_handles {
            runtime.loader.code_symbols.remove(&handle);
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_get_info(
    code_object: HsaCodeObject,
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
        let Some(code_object) = runtime.loader.code_objects.get(&code_object.handle) else {
            return INVALID_CODE_OBJECT;
        };
        let object = match LoaderCodeObject::parse(&code_object.bytes) {
            Ok(object) => object,
            Err(error) => return map_loader_error(error),
        };
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                CODE_OBJECT_INFO_VERSION => {
                    let version = match object.version() {
                        Ok(version) => version,
                        Err(error) => return map_loader_error(error),
                    };
                    let text = format!("{version}.0");
                    std::ptr::write_bytes(value.cast::<u8>(), 0, 64);
                    std::ptr::copy_nonoverlapping(
                        text.as_ptr(),
                        value.cast::<u8>(),
                        text.len().min(63),
                    );
                }
                CODE_OBJECT_INFO_TYPE => value.cast::<u32>().write(CODE_OBJECT_TYPE_PROGRAM),
                CODE_OBJECT_INFO_ISA => {
                    let isa = match code_object_isa(runtime, &object) {
                        Ok(isa) => isa,
                        Err(status) => return status,
                    };
                    value.cast::<HsaIsa>().write(isa);
                }
                CODE_OBJECT_INFO_MACHINE_MODEL => {
                    value.cast::<u32>().write(MACHINE_MODEL_LARGE);
                }
                CODE_OBJECT_INFO_PROFILE => value.cast::<u32>().write(PROFILE_BASE),
                CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE => {
                    value.cast::<u32>().write(DEFAULT_FLOAT_ROUNDING_MODE_NEAR);
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

unsafe fn code_object_get_symbol_impl(
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_get_symbol")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_get_symbol(
    code_object: HsaCodeObject,
    symbol_name: *const c_char,
    symbol: *mut HsaCodeSymbol,
) -> Status {
    // SAFETY: This forwards the public arguments without changing their contracts.
    unsafe { code_object_get_symbol_impl(code_object, std::ptr::null(), symbol_name, symbol) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_get_symbol_from_name")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_get_symbol_from_name(
    code_object: HsaCodeObject,
    module_name: *const c_char,
    symbol_name: *const c_char,
    symbol: *mut HsaCodeSymbol,
) -> Status {
    // SAFETY: This forwards the public arguments without changing their contracts.
    unsafe { code_object_get_symbol_impl(code_object, module_name, symbol_name, symbol) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_symbol_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_symbol_get_info(
    code_symbol: HsaCodeSymbol,
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
        let Some(symbol) = runtime.loader.code_symbols.get(&code_symbol.handle) else {
            return INVALID_CODE_SYMBOL;
        };
        if !runtime
            .loader
            .code_objects
            .contains_key(&symbol.code_object)
        {
            return INVALID_CODE_SYMBOL;
        }
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                SYMBOL_INFO_TYPE => value.cast::<u32>().write(symbol.kind),
                SYMBOL_INFO_NAME_LENGTH => value
                    .cast::<u32>()
                    .write(u32::try_from(symbol.name.len()).unwrap_or(u32::MAX)),
                SYMBOL_INFO_NAME => std::ptr::copy_nonoverlapping(
                    symbol.name.as_ptr(),
                    value.cast::<u8>(),
                    symbol.name.len(),
                ),
                SYMBOL_INFO_MODULE_NAME_LENGTH => value
                    .cast::<u32>()
                    .write(u32::try_from(symbol.module_name.len()).unwrap_or(u32::MAX)),
                SYMBOL_INFO_MODULE_NAME => std::ptr::copy_nonoverlapping(
                    symbol.module_name.as_ptr(),
                    value.cast::<u8>(),
                    symbol.module_name.len(),
                ),
                SYMBOL_INFO_LINKAGE => value.cast::<u32>().write(symbol.linkage),
                SYMBOL_INFO_IS_DEFINITION => value.cast::<bool>().write(symbol.is_definition),
                SYMBOL_INFO_VARIABLE_ALLOCATION => value.cast::<u32>().write(symbol.allocation),
                SYMBOL_INFO_VARIABLE_SEGMENT => value.cast::<u32>().write(symbol.segment),
                SYMBOL_INFO_VARIABLE_ALIGNMENT => value.cast::<u32>().write(symbol.alignment),
                SYMBOL_INFO_VARIABLE_SIZE => value.cast::<u32>().write(symbol.size),
                SYMBOL_INFO_VARIABLE_IS_CONST => value.cast::<bool>().write(symbol.is_const),
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
                SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE => {
                    value.cast::<u32>().write(symbol.wavefront_size);
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_iterate_symbols")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_iterate_symbols(
    code_object: HsaCodeObject,
    callback: CodeObjectSymbolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let handles = {
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
            runtime.loader.code_objects[&code_object.handle]
                .symbol_handles
                .clone()
        };
        for handle in handles {
            // SAFETY: Traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(code_object, HsaCodeSymbol { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_reader_create_from_file")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_reader_create_from_file(
    file: i32,
    reader: *mut HsaCodeObjectReader,
) -> Status {
    boundary(|| {
        if file < 0 || reader.is_null() {
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
        let size = match linux_interop::descriptor_length(file)
            .ok()
            .and_then(|length| usize::try_from(length).ok())
        {
            Some(0) => return INVALID_CODE_OBJECT,
            Some(size) => size,
            None => return INVALID_FILE,
        };
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(size).is_err() {
            return OUT_OF_RESOURCES;
        }
        bytes.resize(size, 0);
        if linux_interop::read_descriptor_exact_at(file, &mut bytes, 0).is_err() {
            return INVALID_FILE;
        }
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.loader.readers.insert(
            handle,
            Reader {
                bytes: bytes.into(),
                storage: ReaderStorage::from_file(file, 0, size),
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { reader.write(HsaCodeObjectReader { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_reader_create_from_memory")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_reader_create_from_memory(
    code_object: *const c_void,
    size: usize,
    reader: *mut HsaCodeObjectReader,
) -> Status {
    boundary(|| {
        if code_object.is_null() || size == 0 || size > isize::MAX as usize || reader.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(size).is_err() {
            return OUT_OF_RESOURCES;
        }
        // SAFETY: The caller supplies size readable bytes for the call, and
        // the checked byte length is within Rust's maximum slice extent.
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(code_object.cast::<u8>(), size)
        });
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
        runtime.loader.readers.insert(
            handle,
            Reader {
                bytes: bytes.into(),
                storage: ReaderStorage::Memory,
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { reader.write(HsaCodeObjectReader { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_reader_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_code_object_reader_destroy(reader: HsaCodeObjectReader) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.loader.readers.remove(&reader.handle).is_some() {
            SUCCESS
        } else {
            INVALID_CODE_OBJECT_READER
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_executable_create(
    profile: u32,
    state: u32,
    _options: *const c_char,
    executable: *mut HsaExecutable,
) -> Status {
    boundary(|| {
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
pub unsafe extern "C" fn hsa_executable_create_alt(
    profile: u32,
    rounding: u32,
    _options: *const c_char,
    executable: *mut HsaExecutable,
) -> Status {
    boundary(|| {
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

fn loaded_symbol_agent(program: bool, agent: HsaAgent) -> HsaAgent {
    if program {
        HsaAgent { handle: 0 }
    } else {
        agent
    }
}

unsafe fn load_executable_code_object(
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_load_agent_code_object")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .map(|reader| (reader.bytes.clone(), reader.storage.clone()))
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
            .map(|reader| (reader.bytes.clone(), reader.storage.clone()))
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
pub unsafe extern "C" fn hsa_executable_destroy(executable: HsaExecutable) -> Status {
    boundary(|| {
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
pub unsafe extern "C" fn hsa_executable_validate_alt(
    executable: HsaExecutable,
    _options: *const c_char,
    result: *mut u32,
) -> Status {
    // SAFETY: Options are intentionally ignored, matching ROCr validation.
    unsafe { hsa_executable_validate(executable, result) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_executable_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_loader_query_host_address")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_ven_amd_loader_query_host_address(
    device_address: *const c_void,
    host_address: *mut *const c_void,
) -> Status {
    boundary(|| {
        if device_address.is_null() || host_address.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        match loaded_host_range(&runtime.loader, device_address as usize, 1) {
            Ok(Some(host)) => {
                // SAFETY: The caller supplied writable output storage.
                unsafe { host_address.write(host as *const c_void) };
                SUCCESS
            }
            Ok(None) | Err(_) => INVALID_ARGUMENT,
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_loader_query_segment_descriptors")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_ven_amd_loader_query_segment_descriptors(
    segment_descriptors: *mut HsaLoaderSegmentDescriptor,
    num_segment_descriptors: *mut usize,
) -> Status {
    boundary(|| {
        if num_segment_descriptors.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied readable count storage.
        let requested = unsafe { num_segment_descriptors.read() };
        if (requested == 0) != segment_descriptors.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let actual = runtime
            .loader
            .executables
            .values()
            .map(|executable| {
                executable
                    .core
                    .loaded()
                    .iter()
                    .map(|object| object.image.segments().len())
                    .sum::<usize>()
            })
            .sum::<usize>();
        if requested == 0 {
            // SAFETY: The caller supplied writable count storage.
            unsafe { num_segment_descriptors.write(actual) };
            return SUCCESS;
        }
        if requested != actual {
            return INCOMPATIBLE_ARGUMENTS;
        }
        let mut output_index = 0;
        let mut executable_handles = runtime
            .loader
            .executables
            .keys()
            .copied()
            .collect::<Vec<_>>();
        executable_handles.sort_unstable();
        for executable_handle in executable_handles {
            let executable = &runtime.loader.executables[&executable_handle];
            for object in executable.core.loaded() {
                for segment in object.image.segments() {
                    let descriptor = HsaLoaderSegmentDescriptor {
                        agent: object.metadata.agent,
                        executable: HsaExecutable {
                            handle: executable_handle,
                        },
                        code_object_storage_type: LOADER_STORAGE_MEMORY,
                        code_object_storage_base: object.image.code_object().as_ptr().cast(),
                        code_object_storage_size: object.image.code_object().len(),
                        code_object_storage_offset: segment.storage_offset,
                        segment_base: segment.address as usize as *const c_void,
                        segment_size: segment.size,
                    };
                    // SAFETY: The caller supplied exactly actual writable entries.
                    unsafe { segment_descriptors.add(output_index).write(descriptor) };
                    output_index += 1;
                }
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_loader_query_executable")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_ven_amd_loader_query_executable(
    device_address: *const c_void,
    executable: *mut HsaExecutable,
) -> Status {
    boundary(|| {
        if device_address.is_null() || executable.is_null() {
            return INVALID_ARGUMENT;
        }
        let address = device_address as u64;
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(handle) = runtime
            .loader
            .executables
            .iter()
            .find_map(|(handle, record)| {
                record
                    .core
                    .loaded()
                    .iter()
                    .any(|object| object.image.contains(address))
                    .then_some(*handle)
            })
        else {
            return INVALID_ARGUMENT;
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { executable.write(HsaExecutable { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_loader_executable_iterate_loaded_code_objects")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
    executable: HsaExecutable,
    callback: LoadedCodeObjectCallback,
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
            record
                .core
                .loaded()
                .iter()
                .map(|object| HsaLoadedCodeObject {
                    handle: object.metadata.handle,
                })
                .collect::<Vec<_>>()
        };
        for loaded in handles {
            // SAFETY: Loader traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(executable, loaded, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_loader_loaded_code_object_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_ven_amd_loader_loaded_code_object_get_info(
    loaded_code_object: HsaLoadedCodeObject,
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
        let Some((executable_handle, object)) =
            runtime
                .loader
                .executables
                .iter()
                .find_map(|(handle, executable)| {
                    executable
                        .core
                        .loaded()
                        .iter()
                        .find(|object| object.metadata.handle == loaded_code_object.handle)
                        .map(|object| (*handle, object))
                })
        else {
            return INVALID_CODE_OBJECT;
        };
        let load_delta =
            i128::from(object.image.device_base()) - i128::from(object.image.virtual_base());
        let Ok(load_delta) = i64::try_from(load_delta) else {
            return INVALID_CODE_OBJECT;
        };
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                LOADER_INFO_EXECUTABLE => value.cast::<HsaExecutable>().write(HsaExecutable {
                    handle: executable_handle,
                }),
                LOADER_INFO_KIND => value.cast::<u32>().write(object.metadata.kind),
                LOADER_INFO_AGENT => value.cast::<HsaAgent>().write(object.metadata.agent),
                LOADER_INFO_STORAGE_TYPE => value.cast::<u32>().write(LOADER_STORAGE_MEMORY),
                LOADER_INFO_STORAGE_MEMORY_BASE => value
                    .cast::<u64>()
                    .write(object.image.code_object().as_ptr() as usize as u64),
                LOADER_INFO_STORAGE_MEMORY_SIZE => value
                    .cast::<u64>()
                    .write(u64::try_from(object.image.code_object().len()).unwrap_or(u64::MAX)),
                LOADER_INFO_STORAGE_FILE => return INVALID_ARGUMENT,
                LOADER_INFO_LOAD_DELTA => value.cast::<i64>().write(load_delta),
                LOADER_INFO_LOAD_BASE => value.cast::<u64>().write(object.image.device_base()),
                LOADER_INFO_LOAD_SIZE => value.cast::<u64>().write(object.image.size()),
                LOADER_INFO_URI_LENGTH => value
                    .cast::<u32>()
                    .write(u32::try_from(object.metadata.uri.len()).unwrap_or(u32::MAX)),
                LOADER_INFO_URI => std::ptr::copy_nonoverlapping(
                    object.metadata.uri.as_ptr(),
                    value.cast::<u8>(),
                    object.metadata.uri.len(),
                ),
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(
        export_name = "__hsa_impl_hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size"
    )
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    file: i32,
    offset: usize,
    size: usize,
    reader: *mut HsaCodeObjectReader,
) -> Status {
    boundary(|| {
        if file < 0 {
            return INVALID_FILE;
        }
        if size == 0 {
            return INVALID_CODE_OBJECT;
        }
        if reader.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(size).is_err() {
            return OUT_OF_RESOURCES;
        }
        bytes.resize(size, 0);
        if linux_interop::read_descriptor_exact_at(file, &mut bytes, offset as u64).is_err() {
            return INVALID_FILE;
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
        runtime.loader.readers.insert(
            handle,
            Reader {
                bytes: bytes.into(),
                storage: ReaderStorage::from_file(file, offset, size),
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { reader.write(HsaCodeObjectReader { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_loader_iterate_executables")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_ven_amd_loader_iterate_executables(
    callback: ExecutableCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let mut handles = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            runtime
                .loader
                .executables
                .keys()
                .copied()
                .collect::<Vec<_>>()
        };
        handles.sort_unstable();
        for handle in handles {
            // SAFETY: Loader traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(HsaExecutable { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[cfg(any(test, not(target_arch = "x86_64")))]
pub(crate) fn loader_extension_table() -> [usize; 7] {
    [
        hsa_ven_amd_loader_query_host_address as *const () as usize,
        hsa_ven_amd_loader_query_segment_descriptors as *const () as usize,
        hsa_ven_amd_loader_query_executable as *const () as usize,
        hsa_ven_amd_loader_executable_iterate_loaded_code_objects as *const () as usize,
        hsa_ven_amd_loader_loaded_code_object_get_info as *const () as usize,
        hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size as *const ()
            as usize,
        hsa_ven_amd_loader_iterate_executables as *const () as usize,
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn memory_reader_rejects_oversized_slice_before_access() {
        let byte = 0_u8;
        let mut reader = HsaCodeObjectReader { handle: 0 };
        // SAFETY: The oversized length is rejected before the byte is read.
        let status = unsafe {
            hsa_code_object_reader_create_from_memory(
                (&raw const byte).cast(),
                isize::MAX as usize + 1,
                &raw mut reader,
            )
        };
        assert_eq!(status, INVALID_ARGUMENT);
    }

    fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn write_i64(bytes: &mut [u8], offset: usize, value: i64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn gfx1201_code_object() -> Vec<u8> {
        const SECTION_OFFSET: usize = 64;
        const SECTION_COUNT: usize = 5;
        const STRING_OFFSET: usize = SECTION_OFFSET + SECTION_COUNT * 64;
        const SYMBOL_OFFSET: usize = 448;
        const TEXT_OFFSET: usize = 544;
        const DATA_OFFSET: usize = 608;
        const STRINGS: &[u8] = b"\0module::kernel.kd\0global\0";
        let mut bytes = vec![0_u8; DATA_OFFSET + 8];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[7] = ELF_OS_ABI_AMDGPU_HSA;
        bytes[8] = 4;
        write_u16(&mut bytes, 16, ELF_TYPE_DYN);
        write_u16(&mut bytes, 18, ELF_MACHINE_AMDGPU);
        write_u32(&mut bytes, 20, 1);
        write_u64(&mut bytes, 40, SECTION_OFFSET as u64);
        write_u32(&mut bytes, 48, 0x4e); // gfx1201 machine ID
        write_u16(&mut bytes, 52, ELF_HEADER_SIZE as u16);
        write_u16(&mut bytes, 54, 56);
        write_u16(&mut bytes, 58, 64);
        write_u16(&mut bytes, 60, SECTION_COUNT as u16);

        let strings = SECTION_OFFSET + 64;
        write_u32(&mut bytes, strings + 4, SECTION_STRTAB);
        write_u64(&mut bytes, strings + 24, STRING_OFFSET as u64);
        write_u64(&mut bytes, strings + 32, STRINGS.len() as u64);
        bytes[STRING_OFFSET..STRING_OFFSET + STRINGS.len()].copy_from_slice(STRINGS);

        let symbols = SECTION_OFFSET + 128;
        write_u32(&mut bytes, symbols + 4, SECTION_SYMTAB);
        write_u64(&mut bytes, symbols + 24, SYMBOL_OFFSET as u64);
        write_u64(&mut bytes, symbols + 32, 72);
        write_u32(&mut bytes, symbols + 40, 1);
        write_u64(&mut bytes, symbols + 56, 24);

        let text = SECTION_OFFSET + 192;
        write_u32(&mut bytes, text + 4, SECTION_PROGBITS);
        write_u64(&mut bytes, text + 16, 0x1000);
        write_u64(&mut bytes, text + 24, TEXT_OFFSET as u64);
        write_u64(&mut bytes, text + 32, 64);
        write_u64(&mut bytes, text + 48, 64);

        let data = SECTION_OFFSET + 256;
        write_u32(&mut bytes, data + 4, SECTION_PROGBITS);
        write_u64(
            &mut bytes,
            data + 8,
            SECTION_FLAG_WRITE | SECTION_FLAG_AMDGPU_HSA_AGENT,
        );
        write_u64(&mut bytes, data + 16, 0x2000);
        write_u64(&mut bytes, data + 24, DATA_OFFSET as u64);
        write_u64(&mut bytes, data + 32, 8);
        write_u64(&mut bytes, data + 48, 8);

        let kernel = SYMBOL_OFFSET + 24;
        write_u32(&mut bytes, kernel, 1);
        bytes[kernel + 4] = (SYMBOL_BINDING_GLOBAL << 4) | SYMBOL_TYPE_AMDGPU_HSA_KERNEL;
        write_u16(&mut bytes, kernel + 6, 3);
        write_u64(&mut bytes, kernel + 8, 0x1000);
        write_u64(&mut bytes, kernel + 16, 64);

        let variable = SYMBOL_OFFSET + 48;
        write_u32(&mut bytes, variable, 19);
        bytes[variable + 4] = (SYMBOL_BINDING_GLOBAL << 4) | SYMBOL_TYPE_OBJECT;
        write_u16(&mut bytes, variable + 6, 4);
        write_u64(&mut bytes, variable + 8, 0x2000);
        write_u64(&mut bytes, variable + 16, 8);

        write_u32(&mut bytes, TEXT_OFFSET, 32);
        write_u32(&mut bytes, TEXT_OFFSET + 4, 4);
        write_u32(&mut bytes, TEXT_OFFSET + 8, 16);
        write_u16(&mut bytes, TEXT_OFFSET + 56, 1 << 11);
        bytes
    }

    #[test]
    fn loader_extension_table_contains_every_version_1_03_entry_point() {
        assert!(loader_extension_table().into_iter().all(|entry| entry != 0));
    }

    #[test]
    fn file_reader_entry_point_matches_the_public_abi() {
        let _: unsafe extern "C" fn(i32, *mut HsaCodeObjectReader) -> Status =
            hsa_code_object_reader_create_from_file;
    }

    #[test]
    fn legacy_entry_points_match_the_public_abi() {
        let _: unsafe extern "C" fn(
            *mut c_void,
            usize,
            *const c_char,
            *mut HsaCodeObject,
        ) -> Status = hsa_code_object_deserialize;
        let _: unsafe extern "C" fn(
            HsaCodeObject,
            CodeObjectAllocCallback,
            HsaCallbackData,
            *const c_char,
            *mut *mut c_void,
            *mut usize,
        ) -> Status = hsa_code_object_serialize;
        let _: unsafe extern "C" fn(HsaCodeObject) -> Status = hsa_code_object_destroy;
        let _: unsafe extern "C" fn(HsaCodeObject, u32, *mut c_void) -> Status =
            hsa_code_object_get_info;
        let _: unsafe extern "C" fn(HsaCodeObject, *const c_char, *mut HsaCodeSymbol) -> Status =
            hsa_code_object_get_symbol;
        let _: unsafe extern "C" fn(
            HsaCodeObject,
            *const c_char,
            *const c_char,
            *mut HsaCodeSymbol,
        ) -> Status = hsa_code_object_get_symbol_from_name;
        let _: unsafe extern "C" fn(HsaCodeSymbol, u32, *mut c_void) -> Status =
            hsa_code_symbol_get_info;
        let _: unsafe extern "C" fn(
            HsaCodeObject,
            CodeObjectSymbolCallback,
            *mut c_void,
        ) -> Status = hsa_code_object_iterate_symbols;
        let _: unsafe extern "C" fn(u32, u32, *const c_char, *mut HsaExecutable) -> Status =
            hsa_executable_create;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            HsaAgent,
            HsaCodeObject,
            *const c_char,
        ) -> Status = hsa_executable_load_code_object;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            HsaCodeObjectReader,
            *const c_char,
            *mut HsaLoadedCodeObject,
        ) -> Status = hsa_executable_load_program_code_object;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            *const c_char,
            *const c_char,
            HsaAgent,
            i32,
            *mut HsaExecutableSymbol,
        ) -> Status = hsa_executable_get_symbol;
        let _: unsafe extern "C" fn(HsaExecutable, *const c_char, *mut c_void) -> Status =
            hsa_executable_global_variable_define;
        let _: unsafe extern "C" fn(HsaExecutable, HsaAgent, *const c_char, *mut c_void) -> Status =
            hsa_executable_readonly_variable_define;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            ExecutableSymbolCallback,
            *mut c_void,
        ) -> Status = hsa_executable_iterate_symbols;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            HsaAgent,
            AgentExecutableSymbolCallback,
            *mut c_void,
        ) -> Status = hsa_executable_iterate_agent_symbols;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            ExecutableSymbolCallback,
            *mut c_void,
        ) -> Status = hsa_executable_iterate_program_symbols;
    }

    #[test]
    fn public_loader_constants_match_hsa_headers() {
        assert_eq!(INVALID_CODE_SYMBOL, 0x1018);
        assert_eq!(INVALID_EXECUTABLE_SYMBOL, 0x1019);
        assert_eq!(INVALID_CODE_OBJECT_READER, 0x1021);
        assert_eq!(SYMBOL_KIND_VARIABLE, 0);
        assert_eq!(SYMBOL_KIND_KERNEL, 1);
        assert_eq!(SYMBOL_KIND_INDIRECT_FUNCTION, 2);
    }

    #[test]
    fn parses_gfx1201_code_object_metadata_and_symbols() {
        let bytes = gfx1201_code_object();
        let layout = parse_elf_layout(&bytes);
        assert!(layout.is_ok());
        let Some(layout) = layout.ok() else {
            return;
        };
        let object = LoaderCodeObject::parse(&bytes).unwrap();
        assert_eq!(object.version().ok(), Some(6));
        assert_eq!(object.target().ok(), Some(("gfx1201", 0)));
        let symbols = parse_symbols(&bytes, layout, &[SECTION_SYMTAB]);
        assert!(symbols.is_ok());
        let Some(symbols) = symbols.ok() else {
            return;
        };
        assert_eq!(symbols.len(), 2);

        let kernel = &symbols[0];
        assert_eq!(kernel.full_name, "module::kernel.kd");
        assert_eq!(kernel.module_name, "module");
        assert_eq!(kernel.name, "kernel.kd");
        assert_eq!(kernel.kind, SYMBOL_KIND_KERNEL);
        assert_eq!(kernel.linkage, SYMBOL_LINKAGE_PROGRAM);
        assert!(kernel.is_definition);
        assert_eq!(kernel.kernarg_size, 16);
        assert_eq!(kernel.group_size, 32);
        assert_eq!(kernel.private_size, 4);
        assert!(kernel.dynamic_callstack);
        assert_eq!(kernel.wavefront_size, 32);

        let variable = &symbols[1];
        assert_eq!(variable.name, "global");
        assert_eq!(variable.kind, SYMBOL_KIND_VARIABLE);
        assert_eq!(variable.allocation, VARIABLE_ALLOCATION_AGENT);
        assert_eq!(variable.segment, VARIABLE_SEGMENT_GLOBAL);
        assert_eq!(variable.alignment, 8);
        assert!(variable.is_const);
    }

    #[test]
    fn rejects_non_hsa_elf_images() {
        let mut bytes = gfx1201_code_object();
        bytes[7] = 0;
        assert!(matches!(parse_elf_layout(&bytes), Err(INVALID_CODE_OBJECT)));
    }

    #[test]
    fn rejects_truncated_and_corrupt_elf_tables() {
        let original = gfx1201_code_object();
        // The last section header ends at byte 384. Every shorter prefix is
        // missing either the ELF header or part of its declared section table.
        for length in 0..384 {
            assert!(matches!(
                parse_elf_layout(&original[..length]),
                Err(INVALID_CODE_OBJECT)
            ));
        }

        let mut bytes = original.clone();
        write_u64(&mut bytes, 32, u64::MAX);
        write_u16(&mut bytes, 56, 1);
        assert!(matches!(parse_elf_layout(&bytes), Err(INVALID_CODE_OBJECT)));

        let mut bytes = original.clone();
        write_u64(&mut bytes, 40, u64::MAX);
        assert!(matches!(parse_elf_layout(&bytes), Err(INVALID_CODE_OBJECT)));

        let symbol_section = 64 + 128;
        let string_section = 64 + 64;
        let cases: &[(usize, u64)] = &[
            (symbol_section + 24, u64::MAX),
            (symbol_section + 32, u64::MAX),
            (symbol_section + 56, 0),
            (string_section + 24, u64::MAX),
            (string_section + 32, u64::MAX),
        ];
        for &(offset, value) in cases {
            let mut bytes = original.clone();
            write_u64(&mut bytes, offset, value);
            let layout = parse_elf_layout(&bytes).unwrap();
            assert!(matches!(
                parse_symbols(&bytes, layout, &[SECTION_SYMTAB]),
                Err(INVALID_CODE_OBJECT)
            ));
        }

        let mut bytes = original.clone();
        write_u32(&mut bytes, symbol_section + 40, 5);
        let layout = parse_elf_layout(&bytes).unwrap();
        assert!(matches!(
            parse_symbols(&bytes, layout, &[SECTION_SYMTAB]),
            Err(INVALID_CODE_OBJECT)
        ));

        let mut bytes = original.clone();
        bytes[385] = 0xff; // invalid UTF-8 in the first symbol name
        let layout = parse_elf_layout(&bytes).unwrap();
        assert!(matches!(
            parse_symbols(&bytes, layout, &[SECTION_SYMTAB]),
            Err(INVALID_CODE_OBJECT)
        ));

        let mut bytes = original;
        bytes[409] = b'x'; // remove the final NUL in the string table
        let layout = parse_elf_layout(&bytes).unwrap();
        assert!(matches!(
            parse_symbols(&bytes, layout, &[SECTION_SYMTAB]),
            Err(INVALID_CODE_OBJECT)
        ));
    }

    #[test]
    fn generated_elf_mutations_keep_parser_and_relocator_bounded() {
        let original = gfx1201_code_object();
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        // Deterministic malformed-input sweep, including table offsets,
        // counts, symbol strings and bytes that can become relocation headers.
        for case in 0..4096 {
            let mut bytes = original.clone();
            for _ in 0..=(case % 8) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let index = usize::try_from(seed % u64::try_from(bytes.len()).unwrap()).unwrap();
                bytes[index] ^= (seed >> 32) as u8 | 1;
            }
            let length = if case % 4 == 0 {
                usize::try_from(seed % u64::try_from(bytes.len() + 1).unwrap()).unwrap()
            } else {
                bytes.len()
            };
            let bytes = &bytes[..length];
            if let Ok(object) = LoaderCodeObject::parse(bytes) {
                let layout = object.layout();
                let _ = object.version();
                let _ = parse_symbols(bytes, layout, &[SECTION_SYMTAB, SECTION_DYNSYM]);
                let mut image = [0_u8; 256];
                let _ = apply_relocations(bytes, layout, 0x100, 0x1000, &mut image, |_| None);
            }
        }
    }

    #[test]
    fn applies_relative_and_defined_absolute_relocations() {
        const SECTION_ENTRY_SIZE: usize = 64;
        const SECTION_COUNT: usize = 4;
        const SYMBOL_OFFSET: usize = SECTION_ENTRY_SIZE * SECTION_COUNT;
        const RELOCATION_OFFSET: usize = SYMBOL_OFFSET + 48;
        const STRING_OFFSET: usize = RELOCATION_OFFSET + 48;
        let mut bytes = vec![0_u8; STRING_OFFSET + 1];

        let symbol_section = SECTION_ENTRY_SIZE;
        write_u32(&mut bytes, symbol_section + 4, SECTION_DYNSYM);
        write_u64(&mut bytes, symbol_section + 24, SYMBOL_OFFSET as u64);
        write_u64(&mut bytes, symbol_section + 32, 48);
        write_u32(&mut bytes, symbol_section + 40, 3);
        write_u64(&mut bytes, symbol_section + 56, 24);
        write_u16(&mut bytes, SYMBOL_OFFSET + 24 + 6, 1);
        write_u64(&mut bytes, SYMBOL_OFFSET + 24 + 8, 0x140);

        let string_section = SECTION_ENTRY_SIZE * 3;
        write_u32(&mut bytes, string_section + 4, SECTION_STRTAB);
        write_u64(&mut bytes, string_section + 24, STRING_OFFSET as u64);
        write_u64(&mut bytes, string_section + 32, 1);

        let relocation_section = SECTION_ENTRY_SIZE * 2;
        write_u32(&mut bytes, relocation_section + 4, SECTION_RELA);
        write_u64(
            &mut bytes,
            relocation_section + 24,
            RELOCATION_OFFSET as u64,
        );
        write_u64(&mut bytes, relocation_section + 32, 48);
        write_u32(&mut bytes, relocation_section + 40, 1);
        write_u64(&mut bytes, relocation_section + 56, 24);

        write_u64(&mut bytes, RELOCATION_OFFSET, 0x120);
        write_u64(
            &mut bytes,
            RELOCATION_OFFSET + 8,
            u64::from(RELOCATION_RELATIVE_64),
        );
        write_i64(&mut bytes, RELOCATION_OFFSET + 16, 0x180);
        write_u64(&mut bytes, RELOCATION_OFFSET + 24, 0x128);
        write_u64(
            &mut bytes,
            RELOCATION_OFFSET + 32,
            (1_u64 << 32) | u64::from(RELOCATION_ABSOLUTE_64),
        );
        write_i64(&mut bytes, RELOCATION_OFFSET + 40, 8);

        let mut image = [0_u8; 256];
        let layout = ElfLayout {
            os_abi: ELF_OS_ABI_AMDGPU_HSA,
            abi_version: 0,
            kind: ELF_TYPE_DYN,
            machine: ELF_MACHINE_AMDGPU,
            flags: 0,
            program_offset: 0,
            program_entry_size: 0,
            program_count: 0,
            section_offset: 0,
            section_entry_size: SECTION_ENTRY_SIZE,
            section_count: SECTION_COUNT,
        };
        assert_eq!(
            apply_relocations(&bytes, layout, 0x100, 0x1000, &mut image, |_| None,),
            Ok(())
        );
        assert_eq!(read_u64(&image, 0x20), Some(0x1080));
        assert_eq!(read_u64(&image, 0x28), Some(0x1048));
    }

    #[test]
    fn reports_one_combined_segment_for_v2_code_objects() {
        let loads = [
            LoadSegment {
                file_offset: 0,
                virtual_address: 0,
                file_size: 0x17c0,
                memory_size: 0x17c0,
            },
            LoadSegment {
                file_offset: 0x1800,
                virtual_address: 0x2800,
                file_size: 0x6c8,
                memory_size: 0x6c8,
            },
            LoadSegment {
                file_offset: 0x1ec8,
                virtual_address: 0x3ec8,
                file_size: 0x70,
                memory_size: 0x138,
            },
            LoadSegment {
                file_offset: 0x1f38,
                virtual_address: 0x4f38,
                file_size: 4,
                memory_size: 4,
            },
        ];
        let segments = loaded_segments(&loads, 0x0010_0000, 0);
        assert!(segments.is_ok());
        let Some(segments) = segments.ok() else {
            return;
        };
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].storage_offset, 0);
        assert_eq!(segments[0].address, 0x0010_0000);
        assert_eq!(segments[0].size, 0x4f3c);
    }

    #[test]
    fn scopes_v2_code_object_symbols_to_the_load_agent() {
        let agent = HsaAgent { handle: 7 };
        let symbol_agent = loaded_symbol_agent(false, agent);
        assert_eq!(symbol_agent.handle, agent.handle);
    }

    #[test]
    fn executable_symbol_lookup_checks_initialization_before_arguments() {
        assert!(lock().is_ok_and(|runtime| runtime.as_ref().is_none()));
        // SAFETY: An uninitialized runtime returns before inspecting the
        // deliberately invalid input pointers.
        assert_eq!(
            unsafe {
                hsa_executable_get_symbol_by_name(
                    HsaExecutable { handle: 0 },
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            },
            NOT_INITIALIZED
        );
    }
}
