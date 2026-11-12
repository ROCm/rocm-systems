// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Legacy HSA code-object C entry points.

use std::collections::HashMap;
use std::ffi::{c_char, c_void};

use rocloader::CodeObject as LoaderCodeObject;

use crate::ffi::*;
use crate::loader::code_object::*;
use crate::loader::*;
use crate::runtime::{boundary, initialized_mut, lock};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_deserialize")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Deserializes a code object and publishes its HSA handle.
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
/// Serializes a code object into caller-allocated storage.
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
/// Releases a deserialized code object and its symbols.
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
/// Writes the requested code-object attribute.
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_code_object_get_symbol")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Finds a symbol in a code object by module and name.
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
/// Finds a code-object symbol by its qualified name.
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
/// Writes the requested code-symbol attribute.
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
/// Visits symbols in a deserialized code object.
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
