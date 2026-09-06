// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocprofsys::rocprofiler_sdk
{
struct client_data;

namespace spm
{
/// Validate the SPM configuration using the active runtime settings and configure
/// the SDK SPM runtime service.
///
/// Returns false for invalid user configuration or an SDK context conflict. Other
/// SDK/hardware/runtime SPM setup failures warn and allow tool initialization to
/// continue without SPM.
[[nodiscard]] bool
configure_runtime(client_data* data);

/// Release SPM runtime state owned by the SDK integration.
void
finalize_runtime(client_data* data) noexcept;
}  // namespace spm
}  // namespace rocprofsys::rocprofiler_sdk
