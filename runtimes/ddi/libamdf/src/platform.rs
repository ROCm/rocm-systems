// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Platform services used to translate the AMDF C ABI.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub(crate) use linux::{host_numa_node, native_identity, render_supported};

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub(crate) use windows::{host_numa_node, native_identity, render_supported};

pub(crate) mod external_memory;
