// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "policies/agent_manager_policy.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofsys::policies::domain_service
{

/// @brief External dependencies required by rocprofsys::domain_service and its
/// buffered/callback KFD event domains: agent lookup, PMC/thread/track reporting, and
/// the KFD event category name/description constants. Also covers the region-sample
/// reporting, timemory push/pop, and backtrace API used by
/// domains::callback::on_tracing_api_enter/exit (the hip/hsa callback domains).
template <typename Externals>
concept externals =
    requires {
        typename Externals::pmc_info_t;
        typename Externals::thread_info_t;
        typename Externals::track_t;
        typename Externals::kfd_sample_t;
        typename Externals::agent_t;
        typename Externals::agent_type_t;
        typename Externals::agent_manager_t;
        typename Externals::region_sample;
        typename Externals::rocm_hip_api_category;
        typename Externals::rocm_hsa_api_category;
        typename Externals::rocm_rocjpeg_api_category;
        typename Externals::rocm_rocdecode_api_category;
        typename Externals::rocm_rocshmem_api_category;
        typename Externals::rocm_hipfile_api_category;
        requires agent_manager_policy<typename Externals::agent_manager_t,
                                      typename Externals::agent_t,
                                      typename Externals::agent_type_t>;
        {
            Externals::k_agent_type_gpu
        } -> std::convertible_to<typename Externals::agent_type_t>;
        {
            Externals::k_agent_type_cpu
        } -> std::convertible_to<typename Externals::agent_type_t>;
        { Externals::k_pmc_value_type_absolute } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_dropped_events_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_dropped_events_category_description
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_queue_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_queue_category_description
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_unmap_from_gpu_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_unmap_from_gpu_category_description
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_page_fault_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_page_fault_category_description
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_page_migrate_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_page_migrate_category_description
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_page_fault_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_page_fault_category_description
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_page_migrate_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_event_page_migrate_category_description
        } -> std::convertible_to<std::string_view>;
        { Externals::k_kfd_queue_category_name } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_queue_category_description
        } -> std::convertible_to<std::string_view>;
        {
            Externals::rocm_hip_api_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::rocm_hsa_api_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::rocm_rocjpeg_api_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::rocm_rocdecode_api_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::rocm_rocshmem_api_category_name
        } -> std::convertible_to<std::string_view>;
        {
            Externals::rocm_hipfile_api_category_name
        } -> std::convertible_to<std::string_view>;
        {
            typename Externals::pmc_info_t{
                .type             = typename Externals::agent_type_t{},
                .agent_type_index = std::size_t{},
                .target_arch      = std::string{},
                .event_code       = std::size_t{},
                .instance_id      = std::size_t{},
                .name             = std::string{},
                .symbol           = std::string{},
                .description      = std::string{},
                .long_description = std::string{},
                .component        = std::string{},
                .units            = std::string{},
                .value_type       = std::string{},
                .block            = std::string{},
                .expression       = std::string{},
                .is_constant      = std::uint32_t{},
                .is_derived       = std::uint32_t{},
                .extdata          = std::string{},
            }
        };
        {
            typename Externals::thread_info_t{ std::int32_t{},  std::int32_t{},
                                               std::uint64_t{}, std::uint32_t{},
                                               std::uint32_t{}, std::string{} }
        };
        { typename Externals::track_t{ std::string{}, std::uint64_t{}, std::string{} } };
        {
            typename Externals::kfd_sample_t{ std::uint64_t{},
                                              std::string{},
                                              std::uint64_t{},
                                              std::uint64_t{},
                                              std::string{},
                                              std::string{},
                                              std::string{},
                                              std::string{},
                                              std::uint32_t{},
                                              std::uint8_t{},
                                              std::string{},
                                              double{},
                                              std::optional<std::int64_t>{} }
        };
    } &&
    requires(std::string_view text, Externals::thread_info_t thread_info,
             Externals::track_t track, Externals::pmc_info_t pmc_info,
             Externals::kfd_sample_t sample) {
        { Externals::add_string(text) };
        { Externals::add_thread_info(thread_info) };
        { Externals::add_track(track) };
        { Externals::add_pmc_info(pmc_info) };
        { Externals::buffer_storage_store(std::move(sample)) };
        { Externals::get_pid() } -> std::convertible_to<std::int32_t>;
        { Externals::get_ppid() } -> std::convertible_to<std::int32_t>;
        {
            Externals::get_agent_manager()
        } -> std::convertible_to<typename Externals::agent_manager_t&>;
    }
    // ─── Members required by domains::callback::on_tracing_api_enter/exit (the
    // hip/hsa callback domains) ─────────────────────────────────────────────────────
    //
    // check_backtrace_operations()/get_backtrace_data()/get_backtrace_json() are
    // deliberately NOT checked here: their kind/operation parameters are the SDK's
    // own enum types (e.g. rocprofiler_callback_tracing_kind_t, a real C enum with no
    // implicit int conversion), which this Externals-only concept has no way to name
    // generically. They stay duck-typed, exactly as on_tracing_api_enter/exit already
    // uses them.
    && requires(std::string_view text, Externals::thread_info_t thread_info,
                Externals::region_sample         sample,
                Externals::rocm_hip_api_category hip_category) {
           { Externals::is_active() } -> std::convertible_to<bool>;
           { Externals::get_use_timemory() } -> std::convertible_to<bool>;
           { Externals::tracing_push_timemory(hip_category, text) };
           { Externals::tracing_pop_timemory(hip_category, text) };
           { Externals::metadata_add_string(text) };
           { Externals::metadata_add_thread_info(thread_info) };
           { Externals::buffer_storage_store(std::move(sample)) };
       };

}  // namespace rocprofsys::policies::domain_service
