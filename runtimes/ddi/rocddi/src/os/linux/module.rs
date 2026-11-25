// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! ELF loader references with explicit process residency and native diagnostics.

#![allow(unsafe_code)]

use crate::{Error, ErrorKind};
use std::ffi::{CStr, CString, OsStr, OsString, c_void};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::ptr::NonNull;

pub(crate) struct ResidentModule(usize);

unsafe fn loader_error(operation: &'static str, diagnostic: *const std::ffi::c_char) -> Error {
    // SAFETY: dlerror returns this thread's diagnostic, copied before another call.
    let message = if diagnostic.is_null() {
        "native loader failure".into()
    } else {
        unsafe { CStr::from_ptr(diagnostic) }
            .to_string_lossy()
            .into_owned()
    };
    // dlerror has no errno contract. Preserve its diagnostic without fabricating
    // a native numeric status from unrelated thread-local errno state.
    super::native_error(operation, std::io::Error::other(message))
}

impl ResidentModule {
    pub(crate) unsafe fn acquire(name: &OsStr, loaded_only: bool) -> Result<Self, Error> {
        let name = CString::new(name.as_bytes()).map_err(|_| {
            crate::error::error(ErrorKind::InvalidArgument, "module name contains NUL")
        })?;
        let flags = libc::RTLD_NOW
            | libc::RTLD_LOCAL
            | libc::RTLD_NODELETE
            | if loaded_only { libc::RTLD_NOLOAD } else { 0 };
        // SAFETY: The caller permits foreign initialization; name is terminated.
        let handle = unsafe { libc::dlopen(name.as_ptr(), flags) };
        if handle.is_null() {
            return Err(unsafe { loader_error("dlopen resident module", libc::dlerror()) });
        }
        Ok(Self(handle as usize))
    }

    pub(crate) fn id(&self) -> usize {
        self.0
    }

    pub(crate) unsafe fn symbol(&self, name: &CStr) -> Result<Option<NonNull<c_void>>, Error> {
        // SAFETY: The module reference is retained and name is terminated. Clear
        // prior loader errors before resolving. A null address is unusable to
        // callers regardless of whether it denotes a zero-valued symbol.
        unsafe { libc::dlerror() };
        let address = unsafe { libc::dlsym(self.0 as *mut c_void, name.as_ptr()) };
        // SAFETY: Query this thread's loader diagnostic before another native
        // call can replace it. Missing symbols remain a native diagnostic.
        let diagnostic = unsafe { libc::dlerror() };
        if !diagnostic.is_null() {
            return Err(unsafe { loader_error("dlsym resident module", diagnostic) });
        }
        Ok(NonNull::new(address))
    }
}

impl Drop for ResidentModule {
    fn drop(&mut self) {
        // SAFETY: One successful acquisition owns this reference; NODELETE
        // keeps code resident after reference release, even on attachment failure.
        unsafe { libc::dlclose(self.0 as *mut c_void) };
    }
}

struct Snapshot {
    names: Vec<OsString>,
    failed: bool,
}

unsafe extern "C" fn collect(
    info: *mut libc::dl_phdr_info,
    _size: usize,
    data: *mut c_void,
) -> libc::c_int {
    // SAFETY: dl_iterate_phdr passes a live record and our borrowed state for
    // this synchronous invocation; names are copied before the callback returns.
    let snapshot = unsafe { &mut *data.cast::<Snapshot>() };
    let name = unsafe { CStr::from_ptr((*info).dlpi_name) }.to_bytes();
    if name.is_empty() {
        return 0;
    }
    let mut bytes = Vec::new();
    if bytes.try_reserve_exact(name.len()).is_err() || snapshot.names.try_reserve(1).is_err() {
        snapshot.failed = true;
        return 1;
    }
    bytes.extend_from_slice(name);
    snapshot.names.push(OsString::from_vec(bytes));
    0
}

pub(crate) fn loaded_names() -> Result<Vec<OsString>, Error> {
    let mut snapshot = Snapshot {
        names: Vec::new(),
        failed: false,
    };
    // SAFETY: The callback and all its state remain live until iteration returns.
    unsafe { libc::dl_iterate_phdr(Some(collect), (&raw mut snapshot).cast()) };
    if snapshot.failed {
        return Err(Error::Capacity {
            resource: "loaded module snapshot",
        });
    }
    Ok(snapshot.names)
}
