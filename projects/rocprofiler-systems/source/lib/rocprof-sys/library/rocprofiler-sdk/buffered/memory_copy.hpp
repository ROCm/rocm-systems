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
on_memory_copy_configure()
{
    Externals::get_metadata_registry().add_string(Externals::k_memory_copy_category_name);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_memory_copy(typename SdkBackend::memory_copy_record_t* record,
               [[maybe_unused]] void*                     data)
{
    if(record == nullptr)
    {
        return;
    }

    constexpr const char* k_empty_json           = "{}";
    constexpr auto        k_zero_start_timestamp = 0;
    constexpr auto        k_zero_end_timestamp   = 0;

    const typename Externals::agent_t* dst_agent = nullptr;
    try
    {
        dst_agent = &Externals::get_agent_manager().get_agent_by_handle(
            record->dst_agent_id.handle);
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("dropping record: dst_agent lookup failed for handle {} ({})",
                    record->dst_agent_id.handle, e.what());
        return;
    }

    const typename Externals::agent_t* src_agent = nullptr;
    try
    {
        src_agent = &Externals::get_agent_manager().get_agent_by_handle(
            record->src_agent_id.handle);
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("dropping record: src_agent lookup failed for handle {} ({})",
                    record->src_agent_id.handle, e.what());
        return;
    }

    const auto beg_timestamp_ns = record->start_timestamp;
    const auto end_timestamp_ns = record->end_timestamp;

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

    auto const track_name = fmt::format("GPU Memory Copy to Agent [{}] Thread {}",
                                        dst_agent->logical_node_id, record->thread_id);
    metadata_registry.ensure_track(track_name, [&] {
        return typename Externals::track_t{ track_name, record->thread_id, k_empty_json };
    });

    metadata_registry.add_stream(stream_id);

    const auto name =
        SdkBackend::get_buffer_tracing_names().at(record->kind, record->operation);

    Externals::get_buffer_storage().store(typename Externals::memory_copy_sample_t{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->dst_agent_id.handle, record->src_agent_id.handle, name, record->bytes,
        record->correlation_id.internal,
        SdkBackend::get_parent_stack_id(record->correlation_id),
        SdkBackend::get_memory_copy_dst_address(*record),
        SdkBackend::get_memory_copy_src_address(*record), stream_id });

    if(Externals::get_use_timemory())
    {
        if(const auto sequent_tid =
               Externals::get_thread_info_sequent_tid(record->thread_id))
        {
            const auto bundle_name =
                fmt::format("memory_copy: {} -> {}", src_agent->logical_node_id,
                            dst_agent->logical_node_id);

            Externals::write_timemory_bundle(bundle_name, *sequent_tid,
                                             end_timestamp_ns - beg_timestamp_ns);
        }
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr auto k_memory_copy = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "memory_copy",
            .id    = SdkBackend::BUFFER_TRACING_MEMORY_COPY,
            .mode  = collection_mode::buffered,
            .group = std::nullopt,
        },
    .on_records =
        buffered_callback_dispatcher<SdkBackend,
                                     typename SdkBackend::memory_copy_record_t,
                                     on_memory_copy<SdkBackend, Externals>>::callback,
    .on_configure           = on_memory_copy_configure<Externals>,
    .correlation_dependency = SdkBackend::EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
};

}  // namespace rocprofsys::domains::buffered
