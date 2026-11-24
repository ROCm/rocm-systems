// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA image extension entry points.

mod create;
mod query;
mod sampler;
mod transfer;

#[cfg(any(test, not(hsa_tool_dispatch)))]
pub(crate) use create::*;
#[cfg(any(test, not(hsa_tool_dispatch)))]
pub(crate) use query::*;
#[cfg(any(test, not(hsa_tool_dispatch)))]
pub(crate) use sampler::*;
#[cfg(any(test, not(hsa_tool_dispatch)))]
pub(crate) use transfer::*;

#[cfg(any(test, not(hsa_tool_dispatch)))]
mod table;
#[cfg(any(test, not(hsa_tool_dispatch)))]
pub(crate) use table::image_extension_table;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
