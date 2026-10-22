// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Code-object reader creation and destruction.

use super::*;

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
