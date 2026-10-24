// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMD loader extension queries over published executable images.

use super::*;

/// Copies a URI as a C string; `URI_LENGTH` excludes the written terminator.
///
/// # Safety
/// `value` must address at least `uri.len() + 1` writable bytes.
unsafe fn write_uri(value: *mut c_void, uri: &[u8]) {
    // SAFETY: The caller provided space for the URI and its terminator.
    unsafe {
        std::ptr::copy_nonoverlapping(uri.as_ptr(), value.cast::<u8>(), uri.len());
        value.cast::<u8>().add(uri.len()).write(0);
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ven_amd_loader_query_host_address")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Translates a loaded device address to its host mapping.
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
/// Copies descriptors for loaded executable segments.
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
/// Finds the executable containing a loaded device address.
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
/// Visits code objects loaded into an executable.
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
/// Writes a requested loaded-code-object attribute.
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
                LOADER_INFO_URI => write_uri(value, &object.metadata.uri),
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
/// Snapshots a bounded file range into a code-object reader.
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
/// Visits all live executable handles.
pub unsafe extern "C" fn hsa_ven_amd_loader_iterate_executables(
    callback: ExecutableCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let _traversal = match EXECUTABLE_TRAVERSAL.read() {
            Ok(guard) => guard,
            Err(_) => return ERROR,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Verifies that a URI overwrites a nonzero buffer through its terminator.
    fn uri_copy_writes_nul_terminator() {
        let uri = b"memory://123#offset=0&size=32";
        let mut output = vec![0xa5; uri.len() + 1];
        // SAFETY: The buffer contains one byte more than the URI length.
        unsafe { write_uri(output.as_mut_ptr().cast(), uri) };
        assert_eq!(&output[..uri.len()], uri);
        assert_eq!(output[uri.len()], 0);
    }
}
