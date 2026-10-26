// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMD loader extension queries over published executable images.

use super::*;

/// Copies a URI as a C string; `URI_LENGTH` excludes the written terminator.
///
/// # Safety
/// `value` must address at least `uri.len() + 1` writable bytes.
pub(crate) unsafe fn write_uri(value: *mut c_void, uri: &[u8]) {
    // SAFETY: The caller provided space for the URI and its terminator.
    unsafe {
        std::ptr::copy_nonoverlapping(uri.as_ptr(), value.cast::<u8>(), uri.len());
        value.cast::<u8>().add(uri.len()).write(0);
    }
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
