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
on_kfd_page_fault(typename SdkBackend::kfd_page_fault_record* record, void* data)
{}

template <typename SdkBackend, typename Externals>
inline constexpr auto k_kfd_page_fault = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "kfd_page_fault",
            .id    = SdkBackend::BUFFER_TRACING_KFD_PAGE_FAULT,
            .mode  = collection_mode::buffered,
            .group = domain_group{ .name = "kfd_events" },
        },
    .on_records =
        buffered_callback_dispatcher<SdkBackend,
                                     typename SdkBackend::kfd_page_fault_record,
                                     on_kfd_page_fault<SdkBackend, Externals>>::callback
};

}  // namespace rocprofsys::domains::buffered
