// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/stream_stack_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include "logger/debug.hpp"

#include <optional>
#include <string_view>

namespace rocprofsys::domains::callback
{

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_hip_stream_enter(typename SdkBackend::callback_tracing_record_t     record,
                    [[maybe_unused]] typename SdkBackend::user_data_t* user_data,
                    [[maybe_unused]] void*                             callback_data)
{
    const auto* stream_handle_data =
        static_cast<const SdkBackend::tracing_hip_stream_data_t*>(record.payload);
    if(stream_handle_data == nullptr)
    {
        return;
    }

    auto stream_id = stream_handle_data->stream_id;

    if(record.operation == SdkBackend::HIP_STREAM_SET)
    {
        LOG_TRACE(" operation = ROCPROFILER_HIP_STREAM_SET, phase = "
                  "ROCPROFILER_CALLBACK_PHASE_ENTER, stream_id={}",
                  static_cast<unsigned long>(stream_id.handle));
        rocprofiler_sdk::stream_stack_service<SdkBackend>::push(stream_id);
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_hip_stream_exit(typename SdkBackend::callback_tracing_record_t     record,
                   [[maybe_unused]] typename SdkBackend::user_data_t* user_data,
                   [[maybe_unused]] void*                             callback_data)
{
    const auto* stream_handle_data =
        static_cast<const SdkBackend::tracing_hip_stream_data_t*>(record.payload);
    if(stream_handle_data == nullptr)
    {
        return;
    }

    auto stream_id = stream_handle_data->stream_id;

    if(record.operation == SdkBackend::HIP_STREAM_SET)
    {
        LOG_TRACE("operation = ROCPROFILER_HIP_STREAM_SET, phase = "
                  "ROCPROFILER_CALLBACK_PHASE_EXIT, stream_id={}",
                  static_cast<unsigned long>(stream_id.handle));
        rocprofiler_sdk::stream_stack_service<SdkBackend>::pop();
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_hip_stream = callback_domain_definition<SdkBackend>{
    .meta = domain_descriptor{ .name  = "hip_stream",
                               .id    = SdkBackend::CALLBACK_TRACING_HIP_STREAM,
                               .mode  = collection_mode::callback,
                               .group = std::nullopt },
    .on_record =
        tracing_callback_dispatcher<SdkBackend,
                                    on_hip_stream_enter<SdkBackend, Externals>,
                                    on_hip_stream_exit<SdkBackend, Externals>>::callback,
};

}  // namespace rocprofsys::domains::callback
