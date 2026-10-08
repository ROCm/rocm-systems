// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Selects the native platform implementation used by the rocddi core.
//!
//! Higher layers depend only on the private `Driver` contract and the exported
//! native owner aliases below. Operating-system APIs, handles, and cleanup
//! details remain inside the selected backend. The explicit unsupported-target
//! error prevents an accidental build from appearing portable before a backend
//! with equivalent ownership and recovery guarantees exists.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod linux_kfd;
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub(crate) use linux_kfd::{LinuxKfdDriver as PlatformDriver, NativeSignalEvent};

/// Private selector carried by passive endpoints. Public endpoint records
/// remain independent of the backend's OS handles and discovery cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum EndpointSelector {
    LinuxKfd(linux_kfd::LinuxSelector),
    #[allow(dead_code)]
    Opaque(u64),
}

#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
compile_error!("rocddi currently supports Linux x86-64 and AArch64");
