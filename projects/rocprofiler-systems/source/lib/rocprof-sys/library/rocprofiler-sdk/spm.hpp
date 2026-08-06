// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/environment.hpp"
#include "logger/debug.hpp"

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
inline constexpr auto beta_env_name = "ROCPROFILER_SPM_BETA_ENABLED";

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
template <typename Environment = ::rocprofsys::common::environment<>>
[[nodiscard]] bool
is_spm_enabled_for_sdk(const request& req)
{
    if(!req.requested()) return true;

    if(Environment::template get_env<bool>(beta_env_name, false))
    {
        LOG_WARNING(
            "ROCm SPM counter collection is enabled as a beta feature. Kernel dispatches "
            "are serialized while SPM is active, so timings in the trace will differ "
            "from an uninstrumented run. SPM can also affect system stability and in "
            "rare cases trigger a GPU or system reset. See the SPM section of the "
            "Systems Profiler documentation for supported hardware and driver "
            "requirements.");
        return true;
    }

    LOG_WARNING("ROCm SPM counter collection was requested, but SDK beta SPM is "
                "not explicitly enabled. SPM samples will be skipped for this "
                "run. Set ROCPROFILER_SPM_BETA_ENABLED=ON to acknowledge the "
                "beta risk and enable SPM collection.");
    return false;
}

/// Configure the SDK SPM runtime service on the dedicated Systems SPM context.
[[nodiscard]] bool
configure_runtime(client_data* data, const request& req);

/// Report any accumulated SPM runtime diagnostics.
void
report_runtime_summary(client_data* data);
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
