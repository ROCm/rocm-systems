// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file is always compiled. Keep SPM request parsing, validation, SDK beta
// opt-in checks, and no-SDK fallbacks here; runtime wiring that requires
// rocprofiler-sdk/experimental/spm.h belongs in spm.cpp.

#include "common/environment.hpp"
#include "library/rocprofiler-sdk/spm.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <cctype>
#include <string_view>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace spm
{
namespace
{
constexpr auto beta_env_name = "ROCPROFILER_SPM_BETA_ENABLED";

bool
has_non_space_value(std::string_view value)
{
    return std::any_of(value.begin(), value.end(),
                       [](unsigned char itr) { return std::isspace(itr) == 0; });
}

bool
read_posix_env_bool(const char* name, bool fallback)
{
    return ::rocprofsys::common::get_env(name, fallback);
}
}  // namespace

bool
request::requested() const noexcept
{
    return !events.empty();
}

bool
is_config_valid(const request&                  req,
                const std::vector<std::string>& dispatch_counter_events,
                const std::string&              device_counter_events)
{
    // Backstop for direct library load paths. Tool initialization must reject
    // SPM requests when the required interval or mutual-exclusion constraints
    // are not satisfied.
    if(!req.requested()) return true;

    if(!dispatch_counter_events.empty())
    {
        LOG_WARNING("SPM counter collection is mutually exclusive with "
                    "ROCPROFSYS_ROCM_EVENTS");
        return false;
    }

    if(has_non_space_value(device_counter_events))
    {
        LOG_WARNING("SPM counter collection is mutually exclusive with "
                    "ROCPROFSYS_GPU_PERF_COUNTERS");
        return false;
    }

    if(req.sample_interval == 0)
    {
        LOG_WARNING("SPM counter collection requires a positive sample interval. Set "
                    "ROCPROFSYS_ROCM_SPM_SAMPLE_INTERVAL or pass --spm-sample-interval "
                    "(for example, 8192). Supported intervals are hardware-limited and "
                    "can be queried with 'rocprofv3-avail info --spm-config'.");
        return false;
    }

    return true;
}

bool
beta_opt_in_satisfied(const request& req, env_bool_reader_t read_env)
{
    if(!req.requested()) return true;

    if(read_env(beta_env_name, false))
    {
        LOG_WARNING("ROCm SPM counter collection is enabled as a beta feature");
        return true;
    }

    LOG_WARNING("ROCm SPM counter collection was requested, but SDK beta SPM is "
                "not explicitly enabled. SPM samples will be skipped for this "
                "run. Set ROCPROFILER_SPM_BETA_ENABLED=ON to acknowledge the "
                "beta risk and enable SPM collection.");
    return false;
}

bool
beta_opt_in_satisfied(const request& req)
{
    return beta_opt_in_satisfied(req, read_posix_env_bool);
}

#if !ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
bool
configure_runtime(client_data*, const request& req)
{
    if(!req.requested()) return true;

    LOG_WARNING("SPM runtime collection was requested, but this rocprofiler-sdk "
                "build does not provide the experimental SPM API. Build with a "
                "rocprofiler-sdk version that provides "
                "rocprofiler-sdk/experimental/spm.h.");
    return false;
}

// No-SPM fallback for the unconditional finalize path. The SDK-enabled
// implementation reports accumulated SPM data-loss diagnostics.
void
report_runtime_summary(client_data*)
{}
#endif
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
