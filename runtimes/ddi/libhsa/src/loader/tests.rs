// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Loader ABI and executable ownership regression tests.

use super::executable::state::loaded_symbol_agent;
use super::*;
use crate::api::loader::code_object::*;
use crate::api::loader::executable::{state::*, symbol::*};
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
/// Verifies that loader extension table contains every version 1.04 entry point.
fn loader_extension_table_contains_every_version_1_04_entry_point() {
    assert_eq!(loader_extension_table().len(), 9);
    assert!(loader_extension_table().into_iter().all(|entry| entry != 0));
}

#[test]
/// Verifies that file reader entry point matches the public ABI.
fn file_reader_entry_point_matches_the_public_abi() {
    let _: unsafe extern "C" fn(HsaFile, *mut HsaCodeObjectReader) -> Status =
        hsa_code_object_reader_create_from_file;
    let _: unsafe extern "C" fn(HsaFile, usize, usize, *mut HsaCodeObjectReader) -> Status =
        crate::api::loader::query::hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size;
    let _: unsafe extern "C" fn(HsaCodeObjectReader, *mut usize) -> Status =
        crate::api::loader::query::hsa_ven_amd_loader_code_object_reader_get_load_size;
    let _: unsafe extern "C" fn(
        HsaExecutable,
        HsaAgent,
        HsaCodeObjectReader,
        *const c_char,
        *mut c_void,
        usize,
        *mut HsaLoadedCodeObject,
    ) -> Status = hsa_ven_amd_loader_executable_load_agent_code_object_at_address;
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

#[cfg(unix)]
#[test]
/// A file snapshot keeps immutable bytes after both native owners are gone.
fn bounded_file_snapshot_retains_bytes_and_optional_origin() {
    use std::io::{Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;
    let path = std::env::temp_dir().join(format!("hsa-file-snapshot-{}", std::process::id()));
    let mut caller = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    caller.write_all(b"prefix-payload-suffix").unwrap();
    caller.seek(SeekFrom::Start(2)).unwrap();
    let file = crate::platform::acquire_code_object_file(caller.as_raw_fd()).unwrap();
    let reader = Reader::snapshot_file(&file, 7, 7).unwrap();
    assert_eq!(caller.stream_position().unwrap(), 2);
    assert!(matches!(reader.snapshot().1, ReaderStorage::File { .. }));
    assert_eq!(
        Reader::snapshot_file(&file, usize::MAX, 1).err(),
        Some(OUT_OF_RESOURCES)
    );
    assert_eq!(
        Reader::snapshot_file(&file, 19, 4).err(),
        Some(INVALID_FILE)
    );
    std::fs::remove_file(path).unwrap();
    let anonymous = Reader::snapshot_file(&file, 7, 7).unwrap();
    assert!(matches!(anonymous.snapshot().1, ReaderStorage::Memory));
    caller.seek(SeekFrom::Start(7)).unwrap();
    caller.write_all(b"changed").unwrap();
    drop(file);
    drop(caller);
    assert_eq!(&*reader.snapshot().0, b"payload");
    assert_eq!(&*anonymous.snapshot().0, b"payload");
}
