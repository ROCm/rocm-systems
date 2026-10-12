// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Private host operating-system services used by the DDI core and drivers.
//!
//! A build selects one OS implementation for host allocation, page-size
//! discovery, and I/O error classification. Owned files use [`std::fs::File`]
//! directly. Linux descriptor transfer and fixed-address mappings remain in
//! the Linux implementation.
//! A host allocation reports its numeric extent and retains ownership when
//! explicit release fails, so its DDI owner can retry cleanup.
//!
//! Host mappings do not require device activation.
//! Public runtime policy stays in the memory and session modules.

#[cfg(target_os = "linux")]
pub(crate) mod linux;

#[cfg(target_os = "linux")]
use linux as platform;

#[cfg(target_os = "linux")]
pub(crate) use platform::{HostAllocation, host_page_size, native_error};
