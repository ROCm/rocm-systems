// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <optional>

namespace rocprofsys::domains::callback::roctx
{

// roctxProfilerPause is acted on when the call begins, roctxProfilerResume once it has
// returned.
template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_roctx_control_enter(typename SdkBackend::callback_tracing_record_t     record,
                       [[maybe_unused]] typename SdkBackend::user_data_t* user_data,
                       [[maybe_unused]] void*                             callback_data,
                       [[maybe_unused]] typename SdkBackend::timestamp_t  timestamp)
{
    auto* trigger = Externals::get_roctx_trigger();
    if(trigger != nullptr &&
       record.operation == SdkBackend::MARKER_CONTROL_API_ID_roctxProfilerPause)
    {
        trigger->on_pause();
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_roctx_control_exit(typename SdkBackend::callback_tracing_record_t     record,
                      [[maybe_unused]] typename SdkBackend::user_data_t* user_data,
                      [[maybe_unused]] void*                             callback_data,
                      [[maybe_unused]] typename SdkBackend::timestamp_t  timestamp)
{
    auto* trigger = Externals::get_roctx_trigger();
    if(trigger != nullptr &&
       record.operation == SdkBackend::MARKER_CONTROL_API_ID_roctxProfilerResume)
    {
        trigger->on_resume();
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_control_api = callback_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "marker_control_api",
            .id    = SdkBackend::CALLBACK_TRACING_MARKER_CONTROL_API,
            .mode  = collection_mode::callback,
            .group = std::nullopt,
        },
    .on_record = tracing_callback_dispatcher<
        SdkBackend, on_roctx_control_enter<SdkBackend, Externals>,
        on_roctx_control_exit<SdkBackend, Externals>>::callback,
};

}  // namespace rocprofsys::domains::callback::roctx
