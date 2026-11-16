// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host operations shared by runtime frontends without a session or device.
//!
//! Clock reads and cache maintenance are live operations with their own
//! capability and failure contracts. Passive host topology is in
//! [`crate::topology::host`].

pub mod cache;
pub mod clock;
pub mod file;
/// Linux process ownership support for the sibling runtime adapters.
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub mod linux;
pub mod module;

/// Windows native file acquisition for sibling runtime adapters.
#[cfg(windows)]
#[doc(hidden)]
pub mod windows;
