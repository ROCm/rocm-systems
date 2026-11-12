// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Windows HSA host inputs; GPU and external transports require a provider.

use crate::ffi::{ERROR, HsaAmdAisFileHandle, HsaFile, NOT_SUPPORTED, Status};
use rocddi::topology::{Endpoint, TopologyNode};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

/// The Windows process containing this loaded runtime image.
static PROCESS: AtomicU32 = AtomicU32::new(0);

/// Checks ownership before acquiring runtime locks.
pub(crate) fn process_is_current() -> bool {
    let process = PROCESS.load(Ordering::Acquire);
    process == 0 || process == std::process::id()
}

/// Binds this runtime image to its creating process.
pub(crate) fn bind_process() -> Result<(), Status> {
    let current = std::process::id();
    let _ = PROCESS.compare_exchange(0, current, Ordering::AcqRel, Ordering::Acquire);
    process_is_current().then_some(()).ok_or(ERROR)
}

/// Returns the default until a provider supplies native node identity.
pub(crate) fn node_id(_endpoint: Option<&Endpoint>) -> u32 {
    0
}
/// Reports that no native HSA CPU ordinal is qualified.
pub(crate) fn native_cpu_node_ordinal(_node: &TopologyNode) -> Option<u32> {
    None
}
/// Returns the host default driver identifier.
pub(crate) fn driver_uid(_endpoint: Option<&Endpoint>) -> u32 {
    0
}

/// Rejects null and `INVALID_HANDLE_VALUE` inputs.
pub(crate) fn invalid_file(file: HsaFile) -> bool {
    file.is_null() || file as usize == usize::MAX
}

/// Retains a borrowed Windows file handle for the loader.
pub(crate) fn acquire_code_object_file(file: HsaFile) -> std::io::Result<std::fs::File> {
    rocddi::host::windows::duplicate_file(file)
}

/// Origin metadata is optional; the loader always retains a byte snapshot.
pub(crate) fn code_object_file_uri(
    _file: &std::fs::File,
    _offset: usize,
    _size: usize,
) -> Option<Vec<u8>> {
    None
}

/// AIS's optional native path is not qualified on Windows.
pub(crate) enum AisOperation {
    /// Read file bytes into the allocation.
    Read,
    /// Write allocation bytes into the file.
    Write,
}

/// Rejects AIS without reading a Linux union member or modifying outputs.
pub(crate) unsafe fn ais_file_io(
    _handle: HsaAmdAisFileHandle,
    _device_pointer: *mut c_void,
    _size: u64,
    _file_offset: i64,
    _size_copied: *mut u64,
    _operation_status: *mut i32,
    _operation: AisOperation,
) -> Status {
    let guard = match crate::runtime::lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    if guard.as_ref().is_none() {
        return crate::ffi::NOT_INITIALIZED;
    }
    NOT_SUPPORTED
}
