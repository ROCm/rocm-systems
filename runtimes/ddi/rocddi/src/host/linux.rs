// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Internal Linux fork witness bridge for the AMDF and HSA adapters.
//!
//! The C ABIs and rocddi's platform-neutral owners do not expose this type.

#![allow(unsafe_code)]

use crate::os::linux::process_identity::{FreshForkWitness, fork_witness_valid};
use std::io;

/// A private page that Linux zeroes in a fork child before it can run.
pub struct ForkWitness(FreshForkWitness);

impl ForkWitness {
    /// Captures a witness before an owner can acquire process-local locks.
    pub fn capture() -> io::Result<Self> {
        FreshForkWitness::try_new().map(Self)
    }

    /// Reports whether this owner's page still belongs to its process.
    #[must_use]
    pub fn is_current(&self) -> bool {
        fork_witness_valid(Some(self.0.address()))
    }

    /// Returns the page address for a private Linux dispatch trampoline.
    #[must_use]
    pub fn address(&self) -> usize {
        self.0.address()
    }

    /// Keeps a published witness mapped until this process image exits.
    #[must_use]
    pub fn retain_until_exit(self) -> usize {
        self.0.retain_until_exit()
    }

    /// Checks a page previously returned by `retain_until_exit`.
    ///
    /// # Safety
    /// `address` must be a live address returned by `retain_until_exit` in
    /// this process image or inherited from that image through fork.
    #[must_use]
    pub unsafe fn retained_is_current(address: usize) -> bool {
        fork_witness_valid(Some(address))
    }
}

/// Acquires an owned reference to a borrowed C file descriptor without taking
/// the caller's descriptor or changing its shared cursor.
///
/// # Errors
/// Invalid descriptors and native resource exhaustion are reported by the OS.
pub fn duplicate_file(descriptor: i32) -> std::io::Result<std::fs::File> {
    crate::os::linux::file::duplicate_file(descriptor)
}
