// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/stream_stack_service.hpp"
#include "library/rocprofiler-sdk/types.hpp"

#include "policies/rocprofiler-sdk/domain_service/backend.hpp"
#include "policies/rocprofiler-sdk/domain_service/externals.hpp"

#include "logger/debug.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace rocprofsys::domains::buffered
{

template <policies::domain_service::externals Externals>
inline void
on_scratch_memory_configure()
{
    Externals::get_metadata_registry().add_string(
        Externals::k_scratch_memory_category_name);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_scratch_memory(typename SdkBackend::scratch_memory_record_t* record,
                  [[maybe_unused]] void*                        data)
{
    if(record == nullptr)
    {
        return;
    }

    constexpr const char* k_empty_json           = "{}";
    constexpr auto        k_zero_start_timestamp = 0;
    constexpr auto        k_zero_end_timestamp   = 0;

    const typename Externals::agent_t* agent = nullptr;
    try
    {
        agent =
            &Externals::get_agent_manager().get_agent_by_handle(record->agent_id.handle);
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("dropping record: agent lookup failed for handle {} ({})",
                    record->agent_id.handle, e.what());
        return;
    }

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

    auto const track_name = fmt::format("GPU Scratch Memory [{}] Thread {}",
                                        agent->device_id, record->thread_id);
    metadata_registry.ensure_track(track_name, [&] {
        return typename Externals::track_t{ track_name, record->thread_id, k_empty_json };
    });

    metadata_registry.add_queue(record->queue_id.handle);
    metadata_registry.add_stream(stream_id);

    const auto name =
        SdkBackend::get_buffer_tracing_names().at(record->kind, record->operation);

    Externals::get_buffer_storage().store(typename Externals::scratch_memory_sample_t{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->agent_id.handle, record->queue_id.handle, name,
        static_cast<std::int32_t>(record->operation),
        static_cast<std::int32_t>(record->flags),
        SdkBackend::get_scratch_memory_allocation_size(*record),
        record->correlation_id.internal,
        SdkBackend::get_parent_stack_id(record->correlation_id), stream_id });

    if(Externals::get_use_timemory())
    {
        if(const auto sequent_tid =
               Externals::get_thread_info_sequent_tid(record->thread_id))
        {
            Externals::write_timemory_bundle(
                name, *sequent_tid, record->end_timestamp - record->start_timestamp);
        }
    }
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
