// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/stream_stack_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <cstdint>
#include <optional>

namespace rocprofsys::domains::buffered
{

template <policies::domain_service::externals Externals>
inline void
on_memory_allocation_configure()
{
    Externals::get_metadata_registry().add_string(
        Externals::k_memory_allocation_category_name);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_memory_allocation(typename SdkBackend::memory_allocation_record_t* record,
                     [[maybe_unused]] void*                           data)
{
    if(record == nullptr)
    {
        return;
    }

    constexpr const char* k_empty_json           = "{}";
    constexpr auto        k_zero_start_timestamp = 0;
    constexpr auto        k_zero_end_timestamp   = 0;

    const std::uint64_t stream_id =
        rocprofiler_sdk::stream_stack_service<SdkBackend>::get_stream_id(record).handle;

    auto& metadata_registry = Externals::get_metadata_registry();
    metadata_registry.ensure_thread(record->thread_id, [&] {
        return typename Externals::thread_info_t{
            .parent_process_id = Externals::get_ppid(),
            .process_id        = Externals::get_pid(),
            .thread_id         = record->thread_id,
            .start             = k_zero_start_timestamp,
            .end               = k_zero_end_timestamp,
            .extdata           = k_empty_json,
        };
    });

    metadata_registry.add_stream(stream_id);

    const auto name =
        SdkBackend::get_buffer_tracing_names().at(record->kind, record->operation);

    Externals::get_buffer_storage().store(typename Externals::memory_allocation_sample_t{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->agent_id.handle, name, static_cast<std::int32_t>(record->operation),
        record->allocation_size, record->correlation_id.internal,
        SdkBackend::get_parent_stack_id(record->correlation_id),
        SdkBackend::get_memory_allocation_address(*record), stream_id });
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_memory_allocation = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "memory_allocation",
            .id    = SdkBackend::BUFFER_TRACING_MEMORY_ALLOCATION,
            .mode  = collection_mode::buffered,
            .group = std::nullopt,
        },
    .on_records = buffered_callback_dispatcher<
        SdkBackend, typename SdkBackend::memory_allocation_record_t,
        on_memory_allocation<SdkBackend, Externals>>::callback,
    .on_configure           = on_memory_allocation_configure<Externals>,
    .correlation_dependency = SdkBackend::EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION,
};

}  // namespace rocprofsys::domains::buffered
