// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Typed writes into caller-owned HSA ABI output storage.

use std::ffi::c_void;

use crate::ffi::{INVALID_ARGUMENT, SUCCESS, Status};

pub(super) unsafe fn write_value<T: Copy>(output: *mut c_void, value: T) -> Status {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: The HSA ABI requires output to point to writable storage for T.
    unsafe { output.cast::<T>().write(value) };
    SUCCESS
}

pub(super) unsafe fn write_bytes(output: *mut c_void, bytes: &[u8], extent: usize) -> Status {
    if output.is_null() || bytes.len() > extent {
        return INVALID_ARGUMENT;
    }
    // SAFETY: The HSA ABI requires output to provide the fixed attribute extent.
    unsafe {
        std::ptr::write_bytes(output.cast::<u8>(), 0, extent);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.cast::<u8>(), bytes.len());
    }
    SUCCESS
}
