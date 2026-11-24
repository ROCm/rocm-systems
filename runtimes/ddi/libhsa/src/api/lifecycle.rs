// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! C entry points for the HSA generation lifecycle.

use crate::ffi::Status;
use crate::lifecycle::{initialize, shutdown};
use crate::status::plain_boundary;

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_init")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
/// Acquires one HSA initialization reference.
pub extern "C" fn hsa_init() -> Status {
    plain_boundary(initialize)
}

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_shut_down")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
/// Releases one HSA reference or resumes pending generation retirement.
pub extern "C" fn hsa_shut_down() -> Status {
    plain_boundary(shutdown)
}
