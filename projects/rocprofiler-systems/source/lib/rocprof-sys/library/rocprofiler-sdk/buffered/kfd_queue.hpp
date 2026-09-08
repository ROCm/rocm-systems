// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/rocprofiler-sdk/types.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>

namespace rocprofsys::domains::buffered
{

template <typename Externals>
inline void
on_kfd_queue_configure()
{
    Externals::add_string(Externals::kfd_queue_category_name);

    auto& agent_mgr  = Externals::get_agent_manager();
    auto  gpu_agents = agent_mgr.get_agents_by_type(Externals::AGENT_TYPE_GPU);
    for(const auto& gpu : gpu_agents)
    {
        const auto dev_idx = static_cast<std::uint32_t>(gpu->device_type_index);
        constexpr std::size_t k_event_code  = 0;
        constexpr std::size_t k_instance_id = 0;
        constexpr auto*       k_component   = "rocm";
        constexpr auto*       k_block       = "KFD";
        constexpr auto*       k_expression  = "";
        const std::string     value_type_absolute{ Externals::pmc_value_type_absolute };

        Externals::add_pmc_info(typename Externals::pmc_info_t{
            .type             = Externals::AGENT_TYPE_GPU,
            .agent_type_index = dev_idx,
            .target_arch      = "GPU",
            .event_code       = k_event_code,
            .instance_id      = k_instance_id,
            .name             = std::string{ Externals::kfd_queue_category_name },
            .symbol           = "KFD Queue Events",
            .description      = std::string{ Externals::kfd_queue_category_description },
            .long_description = "KFD queue eviction/restore paired records",
            .component        = k_component,
            .units            = "events",
            .value_type       = value_type_absolute,
            .block            = k_block,
            .expression       = k_expression,
            .is_constant      = 0,
            .is_derived       = 0,
            .extdata          = "{}",
        });
    }
}

template <typename SdkBackend, typename Externals>
inline void
on_kfd_queue(typename SdkBackend::kfd_queue_record* record, void* data)
{
    if(!record)
    {
        return;
    }

    const auto  name = std::string{ SdkBackend::get_buffer_tracing_names().at(
        SdkBackend::BUFFER_TRACING_KFD_QUEUE, record->operation) };
    const auto  tid  = static_cast<std::uint64_t>(record->pid);
    const auto* agent =
        &Externals::get_agent_manager().get_agent_by_handle(record->agent_id.handle);

    Externals::add_thread_info(typename Externals::thread_info_t{
        Externals::get_ppid(), Externals::get_pid(), tid, 0, 0, "{}" });

    auto agent_label = [](const auto* agent_ptr) {
        if(!agent_ptr)
        {
            return std::string{ "?" };
        }

        const bool is_gpu = (agent_ptr->type == Externals::AGENT_TYPE_GPU);
        return fmt::format("{} {}", is_gpu ? "GPU" : "CPU", agent_ptr->device_type_index);
    };

    auto track_name = fmt::format("KFD Queue [{}]", agent_label(agent));
    Externals::add_track(typename Externals::track_t{ track_name, tid, "{}" });

    constexpr double pmc_value = 1.0;
    Externals::buffer_storage_store(typename Externals::kfd_sample_t{
        tid, name, record->start_timestamp, record->end_timestamp, "" /*empty args*/,
        std::string{ Externals::kfd_queue_category_name }, std::move(track_name), "{}",
        static_cast<std::uint32_t>(agent ? agent->device_type_index : 0),
        static_cast<std::uint8_t>(Externals::AGENT_TYPE_GPU),
        std::string{ Externals::kfd_queue_category_name }, pmc_value,
        std::optional<std::int64_t>(record->pid) });
}

template <typename SdkBackend, typename Externals>
inline constexpr auto k_kfd_queue = buffered_domain_definition<SdkBackend>{
    .meta =
        domain_descriptor{
            .name  = "kfd_queue",
            .id    = SdkBackend::BUFFER_TRACING_KFD_QUEUE,
            .mode  = collection_mode::buffered,
            .group = domain_group{ .name = "kfd_events" },
        },
    .on_records =
        buffered_callback_dispatcher<SdkBackend, typename SdkBackend::kfd_queue_record,
                                     on_kfd_queue<SdkBackend, Externals>>::callback,
    .on_configure = on_kfd_queue_configure<Externals>
};

}  // namespace rocprofsys::domains::buffered
