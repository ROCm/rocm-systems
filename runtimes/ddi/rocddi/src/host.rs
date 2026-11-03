// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host operations shared by runtime frontends without a session or device.
//!
//! Clock reads and cache maintenance are live operations with their own
//! capability and failure contracts. Passive host topology is in
//! [`crate::topology::host`].

pub mod cache;
pub mod clock;
