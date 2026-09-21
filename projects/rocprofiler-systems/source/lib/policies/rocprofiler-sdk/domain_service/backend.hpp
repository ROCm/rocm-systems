// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string_view>

namespace rocprofsys::policies::domain_service
{

/// @brief Contract required of the rocprofiler-sdk backend by
/// rocprofsys::domain_service and its buffered_domain/callback_domain/registry
/// collaborators: the buffer/callback tracing-service API used by the KFD event
/// domains, plus the timestamp/backtrace/arg-iteration API used by
/// domains::callback::on_tracing_api_enter/exit (the hip/hsa callback domains). Does
/// not cover the device-counting (GPU perf-counter) API — see
/// policies::gpu_perf_counters::backend for that.
template <typename Backend>
concept backend =
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
        typename Backend::timestamp_t;
        typename Backend::correlation_id_t;
        typename Backend::callback_tracing_operation_args_cb_t;
        { Backend::compile_time_version } -> std::convertible_to<std::uint32_t>;
        {
            Backend::BUFFER_POLICY_LOSSLESS
        } -> std::convertible_to<typename Backend::buffer_policy_t>;
    } &&
    requires(Backend::context_id_t context, Backend::context_id_t* context_ptr,
             Backend::buffer_id_t buffer, Backend::buffer_id_t* buffer_ptr,
             Backend::buffer_tracing_kind_t   buffer_kind,
             Backend::callback_tracing_kind_t callback_kind,
             Backend::tracing_operation_t*    operations,
             Backend::callback_thread_id_t*   thread_ptr,
             Backend::callback_thread_id_t thread, Backend::on_records_cb_t on_records,
             Backend::on_record_cb_t on_record, std::size_t num_operations,
             std::uint32_t operation, void* callback_data,
             Backend::buffer_policy_t policy, Backend::callback_tracing_record_t record,
             Backend::callback_tracing_operation_args_cb_t args_callback,
             Backend::correlation_id_t correlation_id, std::int32_t max_deref,
             Backend::user_data_t user_data, Backend::timestamp_t timestamp) {
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
        // ─── Members required by domains::callback::on_tracing_api_enter/exit ──────
        {
            Backend::get_timestamp()
        } -> std::convertible_to<typename Backend::timestamp_t>;
        {
            Backend::get_parent_stack_id(correlation_id)
        } -> std::convertible_to<std::uint64_t>;
        {
            Backend::iterate_callback_tracing_kind_operation_args(
                record, args_callback, max_deref, callback_data)
        };
        { user_data.value = timestamp };
    };

}  // namespace rocprofsys::policies::domain_service
