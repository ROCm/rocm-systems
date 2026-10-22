// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Public HSA entry points grouped by the contract they implement.

pub(crate) mod agent;
pub(crate) mod amd;
pub(crate) mod isa;
pub(crate) mod lifecycle;
mod output;
pub(crate) mod system;

/// Host DMA-BUF exposure requires an activated GPU in this HSA generation.
pub(crate) fn host_alloc_dmabuf_supported(gpu_count: usize) -> bool {
    gpu_count != 0
}
