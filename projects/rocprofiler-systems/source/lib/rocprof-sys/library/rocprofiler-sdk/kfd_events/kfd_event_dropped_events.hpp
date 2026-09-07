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
on_kfd_event_dropped_events(
    typename SdkBackend::buffer_tracing_kfd_event_dropped_events_record_t* record,
    void*                                                                  data)
{}

template <typename SdkBackend =
              backends::rocprofiler_sdk::backend<rocprofiler_sdk::wrapper>>
inline constexpr auto kfd_event_dropped_events = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "kfd_event_dropped_events",
            .id    = SdkBackend::BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
            .mode  = collection_mode::buffered,
            .group = domain_group{ .name = "kfd_events" },
        },
    .on_records = buffered_callback_dispatcher<
        SdkBackend, typename SdkBackend::buffer_tracing_kfd_event_dropped_events_record_t,
        on_kfd_event_dropped_events<SdkBackend>>::callback
};

}  // namespace rocprofsys::domains::buffered
