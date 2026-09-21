// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"
#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>

namespace rocprofsys::domains::buffered
{

template <policies::domain_service::externals Externals>
inline void
on_scratch_memory_configure()
{
    Externals::get_metadata_registry().add_string(
        Externals::scratch_memory_category_name);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_scratch_memory(typename SdkBackend::scratch_memory_record_t* record, void* data)
{
    if(record == nullptr)
    {
        return;
    }

    (void) data;

    constexpr const char* k_empty_json           = "{}";
    constexpr auto        k_zero_start_timestamp = 0;
    constexpr auto        k_zero_end_timestamp   = 0;

    const std::uint64_t stream_id = SdkBackend::get_stream_id(record).handle;
    const auto&         agent =
        Externals::get_agent_manager().get_agent_by_handle(record->agent_id.handle);

    Externals::get_metadata_registry().add_thread_info(
        { Externals::get_ppid(), Externals::get_pid(), record->thread_id,
          k_zero_start_timestamp, k_zero_end_timestamp, k_empty_json });

    Externals::get_metadata_registry().add_track(
        { fmt::format("GPU Scratch Memory [{}] Thread {}", agent.device_id,
                      record->thread_id),
          record->thread_id, k_empty_json });

    Externals::get_metadata_registry().add_queue(record->queue_id.handle);
    Externals::get_metadata_registry().add_stream(stream_id);

    Externals::get_buffer_storage().store(typename Externals::scratch_memory_sample_t{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->agent_id.handle, record->queue_id.handle,
        static_cast<std::int32_t>(record->kind),
        static_cast<std::int32_t>(record->operation),
        static_cast<std::int32_t>(record->flags),
        SdkBackend::get_scratch_memory_allocation_size(*record),
        record->correlation_id.internal,
        SdkBackend::get_parent_stack_id(record->correlation_id), stream_id });
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_scratch_memory = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "scratch_memory",
            .id    = SdkBackend::BUFFER_TRACING_SCRATCH_MEMORY,
            .mode  = collection_mode::buffered,
            .group = std::nullopt,
        },
    .on_records =
        buffered_callback_dispatcher<SdkBackend,
                                     typename SdkBackend::scratch_memory_record_t,
                                     on_scratch_memory<SdkBackend, Externals>>::callback,
    .on_configure = on_scratch_memory_configure<Externals>
};

}  // namespace rocprofsys::domains::buffered
