// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Windows has no fork inheritance of Rust owners. A process witness still
//! makes the ownership contract explicit; module retention is frontend-owned.

use std::io;

#[derive(Clone)]
pub(crate) struct ProcessIdentity(u32);

#[allow(
    clippy::unnecessary_wraps,
    reason = "OS-selected process service contract"
)]
pub(crate) fn note_runtime_image() -> io::Result<()> {
    Ok(())
}

impl ProcessIdentity {
    #[allow(
        clippy::unnecessary_wraps,
        reason = "OS-selected process service contract"
    )]
    pub(crate) fn capture() -> io::Result<Self> {
        Ok(Self(std::process::id()))
    }

    pub(crate) fn check(&self) -> io::Result<()> {
        if self.0 == std::process::id() {
            Ok(())
        } else {
            Err(io::ErrorKind::Unsupported.into())
        }
    }
}
