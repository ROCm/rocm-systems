// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/spm.hpp"
#include "common/environment.hpp"
#include "common/rocm_spm.hpp"
#include "core/rocprofiler-sdk.hpp"

#include "logger/debug.hpp"

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace spm
{
namespace
{
#if __has_include(<rocprofiler-sdk/experimental/spm.h>)
constexpr bool sdk_spm_header_available = true;
#else
constexpr bool sdk_spm_header_available = false;
#endif

constexpr auto beta_env_name  = "ROCPROFILER_SPM_BETA_ENABLED";
constexpr auto beta_env_value = "1";

constexpr bool runtime_collection_available = false;
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
    request.sdk_header_available = sdk_spm_header_available;

    if(!request.events.empty()) request.enabled = true;

    return request;
}

bool
validate_beta_request(const beta_request&             request,
                      const std::vector<std::string>& dispatch_counter_events)
{
    // Backstop for direct library load paths. rocprof-sys-run/sample reject SPM
    // earlier in the launcher, but tool_init must also fail closed for PR1.
    if(!request.requested()) return true;

    if(!request.sdk_header_available)
    {
        LOG_WARNING("SPM counter collection was requested, but this build was compiled "
                    "without rocprofiler-sdk experimental SPM header support");
        return false;
    }

    if(!runtime_collection_available)
    {
        LOG_WARNING("SPM counter collection was requested, but Systems Profiler SPM "
                    "runtime collection is not implemented in this build");
        return false;
    }

    // The detailed runtime request checks below become reachable when PR2 enables
    // runtime collection. PR1 intentionally rejects any requested SPM collection above.
    if(request.events.empty())
    {
        LOG_WARNING("SPM counter collection was enabled, but no counters were requested. "
                    "Set ROCPROFSYS_ROCM_SPM_EVENTS or pass --spm-events.");
        return false;
    }

    if(!dispatch_counter_events.empty())
    {
        LOG_WARNING("SPM counter collection is mutually exclusive with "
                    "ROCPROFSYS_ROCM_EVENTS in the beta implementation");
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
