// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"

#include <cstddef>
#include <cstdint>

namespace rocprofsys::domains::buffered
{

template <typename SdkBackend, typename Externals>
inline void
on_kfd_event_unmap_from_gpu(typename SdkBackend::kfd_event_unmap_record* record,
                            void*                                        data)
{}

template <typename SdkBackend, typename Externals>
inline constexpr auto k_kfd_event_unmap_from_gpu = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "kfd_event_unmap_from_gpu",
            .id    = SdkBackend::BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
            .mode  = collection_mode::buffered,
            .group = domain_group{ .name = "kfd_events" },
        },
    .on_records = buffered_callback_dispatcher<
        SdkBackend, typename SdkBackend::kfd_event_unmap_record,
        on_kfd_event_unmap_from_gpu<SdkBackend, Externals>>::callback
};

}  // namespace rocprofsys::domains::buffered
