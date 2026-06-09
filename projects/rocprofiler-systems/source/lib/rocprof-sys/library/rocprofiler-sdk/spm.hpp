// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace spm
{
/// Captures the SPM beta settings resolved from Systems configuration.
struct beta_request
{
    bool                     enabled              = false;
    bool                     sdk_header_available = false;
    std::vector<std::string> events               = {};
    std::uint64_t            sample_interval      = 0;
    std::string              sample_interval_unit = {};

    /// Returns true when SPM was explicitly enabled or SPM counters were requested.
    [[nodiscard]] bool requested() const noexcept;
};

/// Build an SPM beta request from the current Systems configuration settings.
[[nodiscard]]
beta_request
get_request();

/// Validate PR1 SPM scaffolding constraints.
///
/// Returns true when SPM is not requested. If SPM is requested, PR1 rejects runtime
/// collection until the SDK-backed SPM path is implemented and validates the intended
/// mutual exclusion with normal ROCm dispatch counter collection.
[[nodiscard]]
bool
validate_beta_request(const beta_request&             request,
                      const std::vector<std::string>& dispatch_counter_events);

/// Sets the SDK beta opt-in environment variable for validated SPM requests.
void
prepare_beta_environment(const beta_request& request);
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
