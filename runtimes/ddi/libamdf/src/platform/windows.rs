// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Windows AMDF identity adapter before a native GPU provider is installed.

use crate::generated::amdf::amdf_endpoint_native_identity_t;
use rocddi::topology::Endpoint;

pub(crate) fn render_supported(_endpoint: &Endpoint) -> bool {
    false
}
pub(crate) fn host_numa_node(_endpoint: &Endpoint) -> u32 {
    u32::MAX
}
pub(crate) fn native_identity(_endpoint: &Endpoint) -> amdf_endpoint_native_identity_t {
    amdf_endpoint_native_identity_t::default()
}
