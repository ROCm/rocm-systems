// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Loader ABI and executable ownership regression tests.

use super::code_object::*;
use super::executable::state::*;
use super::executable::symbol::*;
use super::*;
use crate::api::loader::reader::*;

#[test]
/// Verifies that memory reader rejects oversized slice before access.
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

#[test]
/// Verifies that loader extension table contains every version 1 03 entry point.
fn loader_extension_table_contains_every_version_1_03_entry_point() {
    assert!(loader_extension_table().into_iter().all(|entry| entry != 0));
}

#[test]
/// Verifies that file reader entry point matches the public ABI.
fn file_reader_entry_point_matches_the_public_abi() {
    let _: unsafe extern "C" fn(i32, *mut HsaCodeObjectReader) -> Status =
        hsa_code_object_reader_create_from_file;
}

#[test]
/// Verifies that legacy entry points match the public ABI.
fn legacy_entry_points_match_the_public_abi() {
    let _: unsafe extern "C" fn(*mut c_void, usize, *const c_char, *mut HsaCodeObject) -> Status =
        hsa_code_object_deserialize;
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
    let _: unsafe extern "C" fn(HsaCodeObject, CodeObjectSymbolCallback, *mut c_void) -> Status =
        hsa_code_object_iterate_symbols;
    let _: unsafe extern "C" fn(u32, u32, *const c_char, *mut HsaExecutable) -> Status =
        hsa_executable_create;
    let _: unsafe extern "C" fn(HsaExecutable, HsaAgent, HsaCodeObject, *const c_char) -> Status =
        hsa_executable_load_code_object;
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
    let _: unsafe extern "C" fn(HsaExecutable, ExecutableSymbolCallback, *mut c_void) -> Status =
        hsa_executable_iterate_symbols;
    let _: unsafe extern "C" fn(
        HsaExecutable,
        HsaAgent,
        AgentExecutableSymbolCallback,
        *mut c_void,
    ) -> Status = hsa_executable_iterate_agent_symbols;
    let _: unsafe extern "C" fn(HsaExecutable, ExecutableSymbolCallback, *mut c_void) -> Status =
        hsa_executable_iterate_program_symbols;
}

#[test]
/// Verifies that public loader constants match HSA headers.
fn public_loader_constants_match_hsa_headers() {
    assert_eq!(INVALID_CODE_SYMBOL, 0x1018);
    assert_eq!(INVALID_EXECUTABLE_SYMBOL, 0x1019);
    assert_eq!(INVALID_CODE_OBJECT_READER, 0x1021);
    assert_eq!(SYMBOL_KIND_VARIABLE, 0);
    assert_eq!(SYMBOL_KIND_KERNEL, 1);
    assert_eq!(SYMBOL_KIND_INDIRECT_FUNCTION, 2);
}

#[test]
/// Checks reporting of one combined segment for v2 code objects.
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
/// Checks scoping of v2 code object symbols to the load agent.
fn scopes_v2_code_object_symbols_to_the_load_agent() {
    let agent = HsaAgent { handle: 7 };
    let symbol_agent = loaded_symbol_agent(false, agent);
    assert_eq!(symbol_agent.handle, agent.handle);
}

#[test]
/// Verifies that executable symbol lookup checks initialization before
/// arguments.
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
