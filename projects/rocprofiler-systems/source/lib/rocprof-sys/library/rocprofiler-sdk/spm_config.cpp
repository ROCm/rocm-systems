// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/environment.hpp"
#include "common/rocm_spm.hpp"
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
constexpr auto beta_env_name  = "ROCPROFILER_SPM_BETA_ENABLED";
constexpr auto beta_env_value = "1";

bool
has_non_space_value(std::string_view value)
{
    return std::any_of(value.begin(), value.end(),
                       [](unsigned char itr) { return std::isspace(itr) == 0; });
}
}  // namespace

bool
beta_request::requested() const noexcept
{
    return (enabled || !events.empty());
}

beta_request
get_request()
{
    auto request    = beta_request{};
    request.events  = ::rocprofsys::rocprofiler_sdk::get_rocm_spm_events();
    request.enabled = ::rocprofsys::rocprofiler_sdk::get_rocm_spm_enabled();
    request.sample_interval =
        ::rocprofsys::rocprofiler_sdk::get_rocm_spm_sample_interval();
    request.sample_interval_unit =
        ::rocprofsys::rocprofiler_sdk::get_rocm_spm_sample_interval_unit();
    if(!request.events.empty()) request.enabled = true;

    return request;
}

bool
validate_beta_request(const beta_request&             request,
                      const std::vector<std::string>& dispatch_counter_events,
                      const std::string&              device_counter_events)
{
    // Backstop for direct library load paths. Tool initialization must validate
    // SPM requests and fail closed when required settings or mutual exclusions
    // are not satisfied.
    if(!request.requested()) return true;

    if(request.events.empty())
    {
        LOG_WARNING("SPM counter collection was enabled, but no counters were requested. "
                    "Set ROCPROFSYS_ROCM_SPM_EVENTS or pass --spm-events.");
        return false;
    }

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

    if(request.sample_interval == 0)
    {
        LOG_WARNING("SPM counter collection requires a positive sample interval. Set "
                    "ROCPROFSYS_ROCM_SPM_SAMPLE_INTERVAL or pass --spm-sample-interval.");
        return false;
    }

    if(request.sample_interval_unit != common::rocm_spm_sample_interval_unit_sclk_cycles)
    {
        LOG_WARNING("Unsupported SPM sample interval unit '{}'. Supported unit: "
                    "{}",
                    request.sample_interval_unit,
                    common::rocm_spm_sample_interval_unit_sclk_cycles);
        return false;
    }

    return true;
}

void
prepare_beta_environment(const beta_request& request)
{
    if(!request.requested()) return;

    ::rocprofsys::common::environment<>::set_env(beta_env_name, beta_env_value, 1);
    LOG_WARNING("ROCm SPM counter collection is enabled as a beta feature");
}
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
