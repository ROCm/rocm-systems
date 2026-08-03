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
using env_bool_reader_t = bool (*)(const char*, bool);

/// Plain value describing an SPM counter collection request.
struct request
{
    std::vector<std::string> events          = {};
    std::uint64_t            sample_interval = 0;

    /// Returns true when the user requested SPM counter collection.
    [[nodiscard]] bool requested() const noexcept;
};

/// Validate SPM collection request constraints.
///
/// Returns true when SPM is not requested because there is no SPM configuration to
/// reject. If SPM is requested, validates the required sample interval settings and
/// mutual exclusion with ROCPROFSYS_ROCM_EVENTS and ROCPROFSYS_GPU_PERF_COUNTERS.
[[nodiscard]]
bool
is_config_valid(const request&                  req,
                const std::vector<std::string>& dispatch_counter_events,
                const std::string&              gpu_perf_counter_events);

/// Returns true when the SDK beta SPM opt-in condition does not block runtime setup.
[[nodiscard]] bool
beta_opt_in_satisfied(const request& req, env_bool_reader_t read_env);

/// Returns true when the SDK beta SPM opt-in condition does not block runtime setup.
[[nodiscard]] bool
beta_opt_in_satisfied(const request& req);

/// Configure the SDK SPM runtime service on the dedicated Systems SPM context.
[[nodiscard]] bool
configure_runtime(client_data* data, const request& req);

/// Report any accumulated SPM runtime diagnostics.
void
report_runtime_summary(client_data* data);
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
