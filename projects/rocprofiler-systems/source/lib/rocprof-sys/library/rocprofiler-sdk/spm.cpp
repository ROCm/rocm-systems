// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/spm.hpp"
#include "common/environment.hpp"
#include "common/rocm_spm.hpp"
#include "core/rocprofiler-sdk.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"

#include "logger/debug.hpp"

#if ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
#    include <rocprofiler-sdk/experimental/spm.h>
#endif

#include <algorithm>
#include <atomic>
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

#if ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
constexpr auto invalid_context_handle = 0UL;
#endif

bool
has_non_space_value(std::string_view value)
{
    return std::any_of(value.begin(), value.end(),
                       [](unsigned char itr) { return std::isspace(itr) == 0; });
}

void
warn_once(std::atomic<bool>& warned, std::string_view message) noexcept
{
    if(!warned.exchange(true)) LOG_WARNING("{}", message);
}

#if ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
std::string_view
status_name(rocprofiler_status_t status)
{
    return rocprofiler_get_status_string(status);
}

void
spm_dispatch_callback(
    const rocprofiler_spm_dispatch_counting_service_data_t* /*dispatch_data*/,
    rocprofiler_counter_config_id_t* config, rocprofiler_user_data_t* user_data,
    void* /*callback_data_args*/) noexcept
{
    if(config) *config = {};
    if(user_data) *user_data = {};

    static auto warned = std::atomic<bool>{ false };
    warn_once(warned, "SPM dispatch service is configured, but per-agent SPM "
                      "counter configuration selection is not implemented yet");
}

void
spm_record_callback(
    const rocprofiler_spm_dispatch_counting_service_data_t* /*dispatch_data*/,
    const rocprofiler_spm_counter_record_t** /*records*/, size_t /*record_count*/,
    rocprofiler_spm_record_flag_t /*flags*/, rocprofiler_user_data_t /*userdata*/,
    void* /*record_callback_args*/) noexcept
{
    static auto warned = std::atomic<bool>{ false };
    warn_once(warned, "SPM records were received, but SPM record translation is "
                      "not implemented yet");
}
#endif
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
    // Backstop for direct library load paths. rocprof-sys-run/sample reject SPM
    // earlier in the launcher, but tool_init must also fail closed for PR1.
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

bool
configure_runtime(client_data* data, const beta_request& request)
{
    if(!request.requested()) return true;

#if !ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
    (void) data;
    static auto warned = std::atomic<bool>{ false };
    warn_once(warned, "SPM runtime collection was requested, but this "
                      "rocprofiler-sdk build does not provide the experimental "
                      "SPM API. Build with a rocprofiler-sdk version that "
                      "provides rocprofiler-sdk/experimental/spm.h.");
    return false;
#else
    if(data == nullptr)
    {
        LOG_WARNING("SPM runtime collection requested but client data is unavailable");
        return false;
    }

    if(data->spm_ctx.handle == invalid_context_handle)
    {
        auto status = rocprofiler_create_context(&data->spm_ctx);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Failed to create SPM context: {} ({})", static_cast<int>(status),
                        status_name(status));
            return false;
        }
    }

    auto status = rocprofiler_spm_configure_callback_dispatch_service(
        data->spm_ctx, spm_dispatch_callback, data, spm_record_callback, data);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to configure SPM callback dispatch service: {} ({})",
                    static_cast<int>(status), status_name(status));
        return false;
    }

    LOG_DEBUG("Configured SPM callback dispatch service on spm_ctx={}",
              data->spm_ctx.handle);
    return true;
#endif
}
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
