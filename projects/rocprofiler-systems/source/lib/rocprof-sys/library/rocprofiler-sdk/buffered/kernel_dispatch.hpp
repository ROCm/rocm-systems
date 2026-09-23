// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/demangler.hpp"

#include "library/rocprofiler-sdk/stream_stack_service.hpp"
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
on_kernel_dispatch_configure()
{
    Externals::get_metadata_registry().add_string(
        Externals::kernel_dispatch_category_name);
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline void
on_kernel_dispatch(typename SdkBackend::kernel_dispatch_record_t* record, void* data)
{
    if(record == nullptr)
    {
        return;
    }

    (void) data;

    constexpr const char* k_empty_json           = "{}";
    constexpr auto        k_zero_start_timestamp = 0;
    constexpr auto        k_zero_end_timestamp   = 0;

    auto name = rocprofsys::utility::demangle(
        Externals::get_kernel_symbol_name(record->dispatch_info.kernel_id));
    auto        beg_timestamp_ns = record->start_timestamp;
    auto        end_timestamp_ns = record->end_timestamp;
    auto        queue_id         = record->dispatch_info.queue_id;
    const auto& agent            = Externals::get_agent_manager().get_agent_by_handle(
        record->dispatch_info.agent_id.handle);

    std::uint64_t stream_id =
        rocprofiler_sdk::stream_stack_service<SdkBackend>::get_stream_id(record).handle;

    {
        Externals::get_metadata_registry().add_thread_info(
            { Externals::get_ppid(), Externals::get_pid(), record->thread_id,
              k_zero_start_timestamp, k_zero_end_timestamp, k_empty_json });

        Externals::get_metadata_registry().add_track(
            { fmt::format("GPU Kernel Dispatch [{}] Queue {}", agent.device_id,
                          queue_id.handle),
              record->thread_id, k_empty_json });

        Externals::get_metadata_registry().add_queue(queue_id.handle);
        Externals::get_metadata_registry().add_stream(stream_id);

        Externals::get_buffer_storage().store(
            typename Externals::kernel_dispatch_sample_t{
                record->start_timestamp, record->end_timestamp, record->thread_id,
                record->dispatch_info.agent_id.handle, record->dispatch_info.kernel_id,
                record->dispatch_info.dispatch_id, record->dispatch_info.queue_id.handle,
                record->correlation_id.internal,
                SdkBackend::get_parent_stack_id(record->correlation_id),
                record->dispatch_info.private_segment_size,
                record->dispatch_info.group_segment_size,
                record->dispatch_info.workgroup_size.x,
                record->dispatch_info.workgroup_size.y,
                record->dispatch_info.workgroup_size.z, record->dispatch_info.grid_size.x,
                record->dispatch_info.grid_size.y, record->dispatch_info.grid_size.z,
                stream_id });
    }

    if(Externals::get_use_timemory())
    {
        const auto sequent_tid =
            Externals::get_thread_info_sequent_tid(record->thread_id);

        Externals::write_timemory_bundle(name, sequent_tid,
                                         end_timestamp_ns - beg_timestamp_ns);
    }
}

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr external_correlation_domain_definition<SdkBackend>
    k_kernel_dispatch_stream_correlation =
        external_correlation_domain_definition<SdkBackend>{
            .kind       = SdkBackend::EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
            .on_request = rocprofiler_sdk::stream_stack_service<
                SdkBackend>::request_stream_correlation_id,
        };

template <policies::domain_service::backend   SdkBackend,
          policies::domain_service::externals Externals>
inline constexpr buffered_domain_definition<SdkBackend> k_kernel_dispatch =
    buffered_domain_definition<SdkBackend>{
        .meta =
            domain_descriptor{
                .name  = "kernel_dispatch",
                .id    = SdkBackend::BUFFER_TRACING_KERNEL_DISPATCH,
                .mode  = collection_mode::buffered,
                .group = std::nullopt,
            },
        .on_records = buffered_callback_dispatcher<
            SdkBackend, typename SdkBackend::kernel_dispatch_record_t,
            on_kernel_dispatch<SdkBackend, Externals>>::callback,
        .on_configure = on_kernel_dispatch_configure<Externals>,
        .correlation_dependency =
            &k_kernel_dispatch_stream_correlation<SdkBackend, Externals>
    };

}  // namespace rocprofsys::domains::buffered
