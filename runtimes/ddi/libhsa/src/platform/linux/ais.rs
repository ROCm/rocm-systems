// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux AIS wire status and descriptor adaptation.

use crate::ffi::*;
use crate::memory::pointer::{owned_memory, pointer_description};
use crate::runtime::lock;
use linux_interop::AisFileOperation as AisOperation;
use rocddi::memory::interop::linux as linux_interop;
use std::ffi::c_void;

/// Linux I/O error fallback when native progress is unknown.
const EIO: i32 = 5;

/// Writes an AIS transfer result into the caller’s output record.
unsafe fn publish_ais_result(
    result: linux_interop::AisFileResult,
    size_copied: *mut u64,
    operation_status: *mut i32,
) -> Status {
    // SAFETY: Both outputs are optional in the public ABI and, when non-null,
    // point to caller-owned writable storage for the duration of this call.
    unsafe {
        if !size_copied.is_null() {
            size_copied.write(result.size_copied);
        }
        if !operation_status.is_null() {
            operation_status.write(result.status);
        }
    }
    if result.status == 0 { SUCCESS } else { ERROR }
}

/// Transfers data between an HSA allocation and an AIS file handle.
pub(crate) unsafe fn ais_transfer(
    descriptor: i32,
    host: usize,
    size: usize,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
    operation: AisOperation,
) -> Status {
    if size > 0 && (host == 0 || host.checked_add(size).is_none() || size > isize::MAX as usize) {
        return ERROR;
    }
    // SAFETY: ais_file_io validates the full host mapping and holds the
    // runtime lock until this synchronous positioned I/O call returns.
    let buffer = match operation {
        AisOperation::Read => linux_interop::AisHostBuffer::Read(if size == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(host as *mut u8, size) }
        }),
        AisOperation::Write => linux_interop::AisHostBuffer::Write(if size == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(host as *const u8, size) }
        }),
    };
    let result = linux_interop::ais_host_transfer(descriptor, buffer, file_offset);
    // SAFETY: The caller owns the optional output storage for this call.
    unsafe { publish_ais_result(result, size_copied, operation_status) }
}

/// Reads or writes a file descriptor for an AIS transfer.
pub(crate) unsafe fn ais_file_io(
    handle: HsaAmdAisFileHandle,
    device_pointer: *mut c_void,
    size: u64,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
    operation: AisOperation,
) -> Status {
    // SAFETY: Linux defines the public union's active member as fd.
    let descriptor = unsafe { handle.fd };
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    if device_pointer.is_null() || descriptor < 0 {
        return INVALID_ARGUMENT;
    }
    let transfer_size = match usize::try_from(size.min(linux_interop::AIS_MAX_TRANSFER_BYTES)) {
        Ok(size) => size,
        Err(_) => return ERROR,
    };
    let Some(description) = pointer_description(runtime, device_pointer as usize) else {
        return ERROR;
    };
    if runtime.is_cpu_agent(description.owner) {
        return ERROR;
    }
    let address = device_pointer as usize;
    let Some(offset) = description.offset_range(address, transfer_size) else {
        return ERROR;
    };
    // Hold the runtime lock through either synchronous transfer, preventing
    // another HSA call from freeing the allocation during the borrow.
    let status = if let Some(host) = description.host_address(address) {
        // SAFETY: The mapped host range stays owned by the locked runtime.
        // The public call keeps its descriptor live until this call returns.
        unsafe {
            ais_transfer(
                descriptor,
                host,
                transfer_size,
                file_offset,
                size_copied,
                operation_status,
                operation,
            )
        }
    } else {
        let Some(memory) = owned_memory(runtime, address).filter(|memory| {
            memory.description.agent_base == description.agent_base
                && memory.description.host_base == description.host_base
                && memory.description.offset_range(address, transfer_size) == Some(offset)
        }) else {
            return ERROR;
        };
        let offset = match u64::try_from(offset) {
            Ok(offset) => offset,
            Err(_) => return ERROR,
        };
        let size = match u64::try_from(transfer_size) {
            Ok(size) => size,
            Err(_) => return ERROR,
        };
        match linux_interop::ais_transfer(
            &memory.allocation,
            descriptor,
            offset,
            size,
            file_offset,
            operation,
        ) {
            Ok(result) => {
                // SAFETY: Optional outputs belong to the caller for this call.
                unsafe { publish_ais_result(result, size_copied, operation_status) }
            }
            Err(error) => {
                // The ioctl input overlaps its output. On error KFD's copied
                // count cannot be trusted, so leave size_copied untouched.
                let errno = error
                    .native_status()
                    .and_then(rocddi::NativeStatus::errno_code)
                    .filter(|errno| *errno > 0)
                    .unwrap_or(EIO);
                if !operation_status.is_null() {
                    // SAFETY: The non-null optional output is caller-owned.
                    unsafe { operation_status.write(-errno) };
                }
                ERROR
            }
        }
    };
    drop(guard);
    status
}
