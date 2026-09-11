// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "policies/agent_manager_policy.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>

namespace rocprofsys::policies::rocprofiler_sdk
{

template <typename Backend>
concept domain_service_backend =
    requires {
        typename Backend::context_id_t;
        typename Backend::buffer_id_t;
        typename Backend::record_header_t;
        typename Backend::callback_thread_id_t;
        typename Backend::user_data_t;
        typename Backend::callback_tracing_record_t;
        typename Backend::tracing_operation_t;
        typename Backend::buffer_tracing_kind_t;
        typename Backend::callback_tracing_kind_t;
        typename Backend::on_records_cb_t;
        typename Backend::on_record_cb_t;
        typename Backend::buffer_policy_t;
        { Backend::compile_time_version } -> std::convertible_to<std::uint32_t>;
        {
            Backend::BUFFER_POLICY_LOSSLESS
        } -> std::convertible_to<typename Backend::buffer_policy_t>;
    } && requires(Backend::context_id_t context, Backend::context_id_t* context_ptr,
                  Backend::buffer_id_t buffer, Backend::buffer_id_t* buffer_ptr,
                  Backend::buffer_tracing_kind_t   buffer_kind,
                  Backend::callback_tracing_kind_t callback_kind,
                  Backend::tracing_operation_t*    operations,
                  Backend::callback_thread_id_t*   thread_ptr,
                  Backend::callback_thread_id_t    thread,
                  Backend::on_records_cb_t on_records, Backend::on_record_cb_t on_record,
                  std::size_t num_operations, std::uint32_t operation,
                  void* callback_data, Backend::buffer_policy_t policy) {
        { Backend::create_context(context_ptr) };
        { Backend::start_context(context) };
        {
            Backend::create_buffer(context, num_operations, num_operations, policy,
                                   on_records, callback_data, buffer_ptr)
        };
        {
            Backend::configure_buffer_tracing_service(context, buffer_kind, operations,
                                                      num_operations, buffer)
        };
        { Backend::create_callback_thread(thread_ptr) };
        { Backend::assign_callback_thread(buffer, thread) };
        { Backend::flush_buffer(buffer) };
        { Backend::destroy_buffer(buffer) };
        {
            Backend::configure_callback_tracing_service(context, callback_kind,
                                                        operations, num_operations,
                                                        on_record, callback_data)
        };
        { Backend::get_buffer_tracing_names() } -> std::ranges::range;
        { Backend::get_callback_tracing_names() } -> std::ranges::range;
        {
            Backend::get_buffer_tracing_names().at(buffer_kind, operation)
        } -> std::convertible_to<std::string_view>;
        {
            Backend::get_callback_tracing_names().at(callback_kind, operation)
        } -> std::convertible_to<std::string_view>;
        // Each entry of a tracing-name table maps a domain to its name, the
        // human-readable names of its operations, and the raw domain id used to key
        // the per-domain descriptor registry.
        requires requires(
            std::ranges::range_value_t<decltype(Backend::get_buffer_tracing_names())>
                buffered_entry) {
            { buffered_entry.name } -> std::convertible_to<std::string_view>;
            { buffered_entry.operations } -> std::ranges::range;
            { buffered_entry.value } -> std::convertible_to<std::size_t>;
        };
        requires requires(
            std::ranges::range_value_t<decltype(Backend::get_callback_tracing_names())>
                callback_entry) {
            { callback_entry.name } -> std::convertible_to<std::string_view>;
            { callback_entry.operations } -> std::ranges::range;
            { callback_entry.value } -> std::convertible_to<std::size_t>;
        };
    };

template <typename Externals>
concept domain_service_externals =
    requires {
        typename Externals::pmc_info_t;
        typename Externals::thread_info_t;
        typename Externals::track_t;
        typename Externals::kfd_sample_t;
        typename Externals::agent_t;
        typename Externals::agent_type_t;
        typename Externals::agent_manager_t;
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
        { Externals::k_kfd_queue_category_name } -> std::convertible_to<std::string_view>;
        {
            Externals::k_kfd_queue_category_description
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
    } && requires(std::string_view text, Externals::thread_info_t thread_info,
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
    };

}  // namespace rocprofsys::policies::rocprofiler_sdk
