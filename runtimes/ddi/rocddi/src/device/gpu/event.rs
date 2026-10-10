// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU notification services that depend on a platform driver.

/// Linux KFD signal-event interoperability.
#[cfg(target_os = "linux")]
pub mod linux;
