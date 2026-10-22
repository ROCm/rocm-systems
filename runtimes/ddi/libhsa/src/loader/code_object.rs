// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Code-object parsing, symbols, and legacy HSA code-object calls.

use super::*;

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

pub(super) fn map_loader_error(error: rocloader::Error) -> Status {
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

pub(super) fn hsa_symbol_kind(kind: SymbolKind) -> u32 {
    match kind {
        SymbolKind::Kernel => SYMBOL_KIND_KERNEL,
        SymbolKind::Variable => SYMBOL_KIND_VARIABLE,
        SymbolKind::IndirectFunction => SYMBOL_KIND_INDIRECT_FUNCTION,
    }
}

pub(super) fn hsa_symbol_linkage(linkage: SymbolLinkage) -> u32 {
    match linkage {
        SymbolLinkage::Module => SYMBOL_LINKAGE_MODULE,
        SymbolLinkage::Program => SYMBOL_LINKAGE_PROGRAM,
    }
}

pub(super) fn hsa_variable_allocation(allocation: VariableAllocation) -> u32 {
    match allocation {
        VariableAllocation::Agent => VARIABLE_ALLOCATION_AGENT,
        VariableAllocation::Program => VARIABLE_ALLOCATION_PROGRAM,
    }
}

pub(super) fn hsa_variable_segment(segment: VariableSegment) -> u32 {
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
        let Some(name) = crate::isa_name(&gpu.info, gpu.xnack_enabled, variant) else {
            continue;
        };
        let mut active = Target::from(&gpu.info);
        active.xnack_enabled = Some(gpu.xnack_enabled);
        if name
            .strip_prefix("amdgcn-amd-amdhsa--")
            .and_then(|name| name.split(':').next())
            == Some(target)
            && object.validate_target(active).is_ok()
        {
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
