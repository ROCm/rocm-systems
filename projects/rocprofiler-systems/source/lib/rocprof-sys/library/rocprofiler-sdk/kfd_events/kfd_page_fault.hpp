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
on_kfd_page_fault(typename SdkBackend::buffer_tracing_kfd_page_fault_record_t* record,
                  void*                                                        data)
{}

template <typename SdkBackend =
              backends::rocprofiler_sdk::backend<rocprofiler_sdk::wrapper>>
inline constexpr auto kfd_page_fault = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "kfd_page_fault",
            .id    = SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT,
            .mode  = collection_mode::buffered,
            .group = domain_group{ .name = "kfd_events" },
        },
    .on_records = buffered_callback_dispatcher<
        SdkBackend, typename SdkBackend::buffer_tracing_kfd_page_fault_record_t,
        on_kfd_page_fault<SdkBackend>>::callback
};

}  // namespace rocprofsys::domains::buffered
