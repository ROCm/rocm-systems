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
on_memory_copy_configure()
{
    Externals::get_metadata_registry().add_string(Externals::memory_copy_category_name);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_memory_copy(typename SdkBackend::memory_copy_record_t* record, void* data)
{
    if(record == nullptr)
    {
        return;
    }

    (void) data;

    constexpr const char* k_empty_json           = "{}";
    constexpr auto        k_zero_start_timestamp = 0;
    constexpr auto        k_zero_end_timestamp   = 0;

    const auto  beg_timestamp_ns = record->start_timestamp;
    const auto  end_timestamp_ns = record->end_timestamp;
    const auto& dst_agent =
        Externals::get_agent_manager().get_agent_by_handle(record->dst_agent_id.handle);
    const auto& src_agent =
        Externals::get_agent_manager().get_agent_by_handle(record->src_agent_id.handle);

    const std::uint64_t stream_id = SdkBackend::get_stream_id(record).handle;

    Externals::get_metadata_registry().add_thread_info(
        { Externals::get_ppid(), Externals::get_pid(), record->thread_id,
          k_zero_start_timestamp, k_zero_end_timestamp, k_empty_json });

    Externals::get_metadata_registry().add_track(
        { fmt::format("GPU Memory Copy to Agent [{}] Thread {}",
                      dst_agent.logical_node_id, record->thread_id),
          record->thread_id, k_empty_json });

    Externals::get_metadata_registry().add_stream(stream_id);

    Externals::get_buffer_storage().store(typename Externals::memory_copy_sample_t{
        record->start_timestamp, record->end_timestamp, record->thread_id,
        record->dst_agent_id.handle, record->src_agent_id.handle,
        static_cast<std::int32_t>(record->kind),
        static_cast<std::int32_t>(record->operation), record->bytes,
        record->correlation_id.internal,
        SdkBackend::get_parent_stack_id(record->correlation_id),
        SdkBackend::get_memory_copy_dst_address(*record),
        SdkBackend::get_memory_copy_src_address(*record), stream_id });

    if(Externals::get_use_timemory())
    {
        const auto sequent_tid =
            Externals::get_thread_info_sequent_tid(record->thread_id);
        auto name = fmt::format("memory_copy: {} -> {}", src_agent.logical_node_id,
                                dst_agent.logical_node_id);

        Externals::write_timemory_bundle(name, sequent_tid,
                                         end_timestamp_ns - beg_timestamp_ns);
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr external_correlation_domain_definition<SdkBackend>
    k_memory_copy_stream_correlation = external_correlation_domain_definition<SdkBackend>{
        .kind       = SdkBackend::EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
        .on_request = SdkBackend::request_stream_correlation_id,
    };

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr buffered_domain_definition<SdkBackend> k_memory_copy =
    buffered_domain_definition<SdkBackend>{
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
        .correlation_dependency = &k_memory_copy_stream_correlation<SdkBackend, Externals>
    };

}  // namespace rocprofsys::domains::buffered
