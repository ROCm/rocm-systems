// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/backend.hpp"
#include "backends/rocprofiler_sdk/wrapper.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include <cstddef>
#include <cstdint>

namespace rocprofsys::domains::buffered
{

template <typename SdkBackend>
inline void
on_kfd_event_unmap_from_gpu(
    typename SdkBackend::buffer_tracing_kfd_event_unmap_from_gpu_record_t* record,
    void*                                                                  data)
{}

template <typename SdkBackend =
              backends::rocprofiler_sdk::backend<rocprofiler_sdk::wrapper>>
inline constexpr auto kfd_event_unmap_from_gpu = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "kfd_event_unmap_from_gpu",
            .id    = SdkBackend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
            .mode  = collection_mode::buffered,
            .group = domain_group{ .name = "kfd_events" },
        },
    .on_records = buffered_callback_dispatcher<
        SdkBackend, typename SdkBackend::buffer_tracing_kfd_event_unmap_from_gpu_record_t,
        on_kfd_event_unmap_from_gpu<SdkBackend>>::callback
};

}  // namespace rocprofsys::domains::buffered
