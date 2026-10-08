// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Opaque client callback data retained across HSA worker handoff.

use std::ffi::c_void;

/// A C callback argument that may be delivered by a runtime worker.
///
/// The runtime never dereferences this pointer. Registration's unsafe C ABI
/// gives the caller responsibility for its lifetime and thread access.
#[derive(Clone, Copy)]
pub(crate) struct CallbackArg(*mut c_void);

impl CallbackArg {
    /// # Safety
    /// Any referenced storage must remain live until callback delivery ends.
    /// The caller must synchronize access performed by callbacks and other
    /// threads. This applies even when delivery outlives the registration call.
    pub(crate) const unsafe fn new(pointer: *mut c_void) -> Self {
        Self(pointer)
    }

    pub(crate) const fn as_ptr(self) -> *mut c_void {
        self.0
    }
}

// SAFETY: Moving or sharing the opaque pointer cannot access its referent.
// Only the caller-supplied unsafe callback can do that, under the lifetime and
// synchronization contract of the registration entry point.
unsafe impl Send for CallbackArg {}
// SAFETY: See the Send contract above. This wrapper exposes no safe dereference.
unsafe impl Sync for CallbackArg {}
