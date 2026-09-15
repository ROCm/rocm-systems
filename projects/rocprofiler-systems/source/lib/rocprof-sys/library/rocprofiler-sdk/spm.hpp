// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocprofsys::rocprofiler_sdk
{
struct client_data;

namespace spm
{
/// Release SPM runtime state owned by the SDK integration.
void
finalize_runtime(client_data* data) noexcept;
}  // namespace spm
}  // namespace rocprofsys::rocprofiler_sdk
