// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service.hpp"

namespace rocprofsys::domains::buffered
{
template <policies::rocprofiler_sdk::domain_service_externals Externals>
inline void
on_kfd_event_page_migrate_configure()
{}

template <policies::rocprofiler_sdk::domain_service_backend   SdkBackend,
          policies::rocprofiler_sdk::domain_service_externals Externals>
inline void
on_kfd_event_page_migrate(typename SdkBackend::kfd_event_page_migrate_record* record,
                          void*                                               data)
{
    (void) record;
    (void) data;
}

template <policies::rocprofiler_sdk::domain_service_backend   SdkBackend,
          policies::rocprofiler_sdk::domain_service_externals Externals>
inline constexpr auto k_kfd_event_page_migrate = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "kfd_event_page_migrate",
            .id    = SdkBackend::BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE,
            .mode  = collection_mode::buffered,
            .group = domain_group{ .name = "kfd_events" },
        },
    .on_records = buffered_callback_dispatcher<
        SdkBackend, typename SdkBackend::kfd_event_page_migrate_record,
        on_kfd_event_page_migrate<SdkBackend, Externals>>::callback,
    .on_configure = on_kfd_event_page_migrate_configure<Externals>
};

}  // namespace rocprofsys::domains::buffered
