// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file is always compiled. Keep SPM request parsing, validation, SDK beta
// opt-in checks, and no-SDK fallbacks here; runtime wiring that requires
// rocprofiler-sdk/experimental/spm.h belongs in spm.cpp.

#include "common/environment.hpp"
#include "core/rocprofiler-sdk.hpp"
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
}  // namespace

request
request::from_settings()
{
    auto spm_request                 = request{};
    spm_request.events               = get_events();
    spm_request.sample_interval      = get_sample_interval();
    spm_request.sample_interval_unit = get_sample_interval_unit();

    return spm_request;
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
                    "can be queried with 'rocprofv3-avail list --spm-config'.");
        return false;
    }

    if(req.sample_interval_unit != env_vars::SPM_SAMPLE_INTERVAL_UNIT_SCLK_CYCLES)
    {
        LOG_WARNING("Unsupported SPM sample interval unit '{}'. Supported unit: "
                    "{}",
                    req.sample_interval_unit,
                    env_vars::SPM_SAMPLE_INTERVAL_UNIT_SCLK_CYCLES);
        return false;
    }

    return true;
}

bool
sdk_beta_opt_in_enabled(const request& req)
{
    if(!req.requested()) return true;

    if(::rocprofsys::common::get_env(beta_env_name, false))
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

#if !ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
bool
configure_runtime(client_data* data, const request& req)
{
    (void) data;
    if(!req.requested()) return true;

    LOG_WARNING("SPM runtime collection was requested, but this rocprofiler-sdk "
                "build does not provide the experimental SPM API. Build with a "
                "rocprofiler-sdk version that provides "
                "rocprofiler-sdk/experimental/spm.h.");
    return false;
}

void
report_runtime_summary(client_data* data)
{
    (void) data;
}
#endif
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
