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
struct client_data;

namespace spm
{
/// Captures the SPM counter collection settings resolved from Systems configuration.
struct request
{
    std::vector<std::string> events               = {};
    std::uint64_t            sample_interval      = 0;
    std::string              sample_interval_unit = {};

    /// Returns true when SPM counters were requested.
    [[nodiscard]] bool requested() const noexcept { return !events.empty(); }

    /// Resolve an SPM collection request from the current Systems settings.
    [[nodiscard]] static request from_settings();
};

/// Validate SPM collection request constraints.
///
/// Returns true when SPM is not requested. If SPM is requested, validates the required
/// sample interval settings and mutual exclusion with ROCPROFSYS_ROCM_EVENTS and
/// ROCPROFSYS_GPU_PERF_COUNTERS.
[[nodiscard]]
bool
is_config_valid(const request&                  req,
                const std::vector<std::string>& dispatch_counter_events,
                const std::string&              device_counter_events);

/// Returns true when the SDK beta SPM opt-in is explicitly enabled.
[[nodiscard]] bool
sdk_beta_opt_in_enabled(const request& req);

/// Configure the SDK SPM runtime service on the dedicated Systems SPM context.
[[nodiscard]] bool
configure_runtime(client_data* data, const request& req);

/// Report any accumulated SPM runtime diagnostics.
void
report_runtime_summary(client_data* data);
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
