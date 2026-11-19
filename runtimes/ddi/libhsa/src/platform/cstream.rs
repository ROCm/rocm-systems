// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Borrowed C streams require a qualified CRT contract across the public ABI.

use crate::ffi::Status;
use std::ffi::c_void;

/// A configured stream borrowed through the caller's logging lifetime.
#[derive(Clone, Copy)]
pub(crate) struct BorrowedCStream {
    #[cfg(target_os = "linux")]
    /// Borrowed Linux libc stream.
    native: std::ptr::NonNull<c_void>,
    #[cfg(not(target_os = "linux"))]
    /// No Windows CRT stream contract is qualified.
    unavailable: std::convert::Infallible,
}

// SAFETY: The frontend serializes writes/reconfiguration and retains the
// caller's borrow. Linux libc serializes other writers of the same stream.
unsafe impl Send for BorrowedCStream {}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    /// Writes complete byte elements to a libc stream.
    fn fwrite(buffer: *const c_void, size: usize, count: usize, stream: *mut c_void) -> usize;
    /// Flushes the retained libc stream.
    fn fflush(stream: *mut c_void) -> i32;
}

impl BorrowedCStream {
    /// Rejects foreign CRT pointers until a Windows stream ABI is qualified.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "native stream support varies by target"
    )]
    pub(crate) fn acquire(stream: *mut c_void) -> Result<Option<Self>, Status> {
        #[cfg(target_os = "linux")]
        {
            Ok(std::ptr::NonNull::new(stream).map(|native| Self { native }))
        }
        #[cfg(not(target_os = "linux"))]
        {
            if stream.is_null() {
                Ok(None)
            } else {
                Err(crate::ffi::NOT_SUPPORTED)
            }
        }
    }

    /// Writes while the frontend retains and serializes the borrowed stream.
    ///
    /// # Safety
    /// The caller's stream remains open, writable, and governed by the selected
    /// native CRT for this call. No concurrent reconfiguration can end its borrow.
    pub(crate) unsafe fn write(self, bytes: &[u8]) {
        #[cfg(target_os = "linux")]
        // SAFETY: The logging owner holds the configured stream borrow.
        unsafe {
            let _ = fwrite(bytes.as_ptr().cast(), 1, bytes.len(), self.native.as_ptr());
            let _ = fflush(self.native.as_ptr());
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = bytes;
            match self.unavailable {}
        }
    }
}
