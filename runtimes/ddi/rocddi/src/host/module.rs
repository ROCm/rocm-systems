// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process-resident native modules for callbacks and retained function tables.
//!
//! A successful acquisition pins code through process exit. Releasing a loader
//! reference does not retire callbacks or invalidate exported addresses. Public
//! API table policy, symbol signatures and tool attachment remain frontend-owned.

#![allow(unsafe_code)]

use crate::os::{ProcessIdentity, module as native};
use crate::{Error, ErrorKind};
use std::ffi::{CStr, OsStr, OsString, c_void};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

/// Opaque process-local identity of an acquired image, independent of its path.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModuleId(usize);

/// Retained native module reference with process-lifetime code residency.
pub struct ResidentModule {
    native: ManuallyDrop<native::ResidentModule>,
    process: ProcessIdentity,
}

impl ResidentModule {
    /// Loads and pins a module using the OS loader's native name semantics.
    ///
    /// # Safety
    /// The caller trusts the module and any foreign initialization/finalization
    /// code executed by the native loader to obey the process's ABI and invariants.
    ///
    /// # Errors
    /// Reports invalid names, process capture failure or native loader failure.
    pub unsafe fn load(name: &OsStr) -> Result<Self, Error> {
        // SAFETY: The caller accepts foreign code execution by this load.
        unsafe { Self::acquire(name, false) }
    }

    /// Pins an already-loaded module without loading a missing image.
    ///
    /// # Safety
    /// The caller accepts foreign loader execution, including symbol resolution.
    ///
    /// # Errors
    /// Reports an absent module, invalid name or native acquisition failure.
    pub unsafe fn acquire_loaded(name: &OsStr) -> Result<Self, Error> {
        // SAFETY: The caller accepts the native loader operation.
        unsafe { Self::acquire(name, true) }
    }

    unsafe fn acquire(name: &OsStr, loaded_only: bool) -> Result<Self, Error> {
        let process = ProcessIdentity::capture()
            .map_err(|source| crate::os::native_error("module process identity", source))?;
        // SAFETY: Both public constructors establish the foreign-code obligation.
        let native = unsafe { native::ResidentModule::acquire(name, loaded_only) }?;
        Ok(Self {
            native: ManuallyDrop::new(native),
            process,
        })
    }

    /// Returns the retained image's identity for deduplication.
    #[must_use]
    pub fn id(&self) -> ModuleId {
        ModuleId(self.native.id())
    }

    /// Resolves an untyped symbol whose address remains resident through exit.
    ///
    /// # Safety
    /// The caller accepts execution of native symbol resolvers. The returned
    /// address has no inferred type; casting, access and invocation require the
    /// exact ABI and lifetime contract of the exported symbol.
    ///
    /// # Errors
    /// Rejects inherited owners in a fork child or native lookup failure.
    pub unsafe fn symbol(&self, name: &CStr) -> Result<Option<NonNull<c_void>>, Error> {
        self.process.check().map_err(|_| {
            crate::error::error(
                ErrorKind::InvalidArgument,
                "module belongs to another process",
            )
        })?;
        // SAFETY: The process check and retained native reference precede lookup.
        unsafe { self.native.symbol(name) }
    }
}

impl Drop for ResidentModule {
    fn drop(&mut self) {
        if self.process.check().is_ok() {
            // SAFETY: Native destruction releases one reference without
            // unmapping pinned code. Inherited owners skip native loader locks.
            unsafe { ManuallyDrop::drop(&mut self.native) };
        }
    }
}

/// Takes a snapshot of loaded image names when the OS implements discovery.
/// Names are observations, not references; acquisition can fail after the snapshot.
///
/// # Errors
/// Reports allocation/native errors or `Unsupported` for an unimplemented snapshot.
pub fn loaded_names() -> Result<Vec<OsString>, Error> {
    native::loaded_names()
}
