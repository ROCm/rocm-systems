// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux name and environment conventions for qualified HSA tool dispatch.

use rocddi::host::module::ResidentModule;
use std::ffi::{CStr, CString, OsStr};
use std::os::unix::ffi::OsStrExt;

/// Linux registrar SONAME; another tool ABI must qualify its own name.
pub(crate) const REGISTRAR: &CStr = c"librocprofiler-register.so.0";

/// Acquires a process-resident module from the Linux HSA byte-string name.
///
/// # Safety
/// The caller trusts the named tool's foreign loader code.
pub(crate) unsafe fn acquire(name: &CStr, loaded_only: bool) -> Option<ResidentModule> {
    let name = OsStr::from_bytes(name.to_bytes());
    // SAFETY: The caller establishes the module execution obligation.
    unsafe {
        if loaded_only {
            ResidentModule::acquire_loaded(name)
        } else {
            ResidentModule::load(name)
        }
    }
    .ok()
}

/// Collects native names before a separate retained acquisition of each tool.
pub(crate) fn loaded_names() -> Vec<CString> {
    rocddi::host::module::loaded_names()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|name| CString::new(name.as_bytes()).ok())
        .collect()
}

/// Reports Linux loader security policy before reading tool environment names.
pub(crate) fn secure_execution() -> bool {
    rocddi::host::linux::secure_execution()
}
