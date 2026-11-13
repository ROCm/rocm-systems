// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native Windows input acquisition for sibling runtime adapters.

/// Acquires an owned file reference while leaving a borrowed C handle intact.
///
/// # Errors
/// Native invalid-handle, permission, and resource-exhaustion failures are
/// preserved. Raw input is validated by Windows before constructing an owner.
pub fn duplicate_file(handle: std::os::windows::io::RawHandle) -> std::io::Result<std::fs::File> {
    crate::os::windows::file::duplicate_file(handle)
}
