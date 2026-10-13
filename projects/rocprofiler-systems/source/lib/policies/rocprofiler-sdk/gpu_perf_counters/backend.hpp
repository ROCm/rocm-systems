// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocprofsys::policies::gpu_perf_counters
{

/// @brief Contract required of the rocprofiler-sdk backend by
/// rocprofsys::pmc::collectors::gpu_perf_counter::device and
/// rocprofsys::pmc::device_providers::rocprofiler_sdk::provider: context lifecycle plus
/// the device-counting (GPU perf-counter) API. Does not cover the buffer/callback
/// tracing-service API used by the KFD event domains — see
/// policies::domain_service::backend for that.
template <typename Backend>
concept backend =
    requires {
        typename Backend::context_id_t;
        typename Backend::agent_id_t;
        typename Backend::buffer_id_t;
        typename Backend::user_data_t;
        typename Backend::status_t;
        typename Backend::counter_flag_t;
        typename Backend::counter_record_t;
        typename Backend::counter_id_t;
        typename Backend::counter_config_id_t;
        typename Backend::counter_metadata_t;
        typename Backend::available_counters_cb_t;
        typename Backend::device_counting_service_cb_t;
        { Backend::flag_none } -> std::convertible_to<typename Backend::counter_flag_t>;
        { Backend::status_success } -> std::convertible_to<typename Backend::status_t>;
        {
            Backend::status_hsa_not_loaded
        } -> std::convertible_to<typename Backend::status_t>;
    } && requires(Backend::context_id_t context, Backend::context_id_t* context_ptr,
                  Backend::agent_id_t agent, Backend::buffer_id_t buffer,
                  Backend::user_data_t user_data, Backend::counter_flag_t flags,
                  Backend::counter_record_t record, Backend::counter_record_t* records,
                  Backend::counter_id_t* counter_id, Backend::counter_id_t counter,
                  Backend::counter_id_t* counters, Backend::counter_config_id_t* config,
                  Backend::available_counters_cb_t      counter_cb,
                  Backend::device_counting_service_cb_t service_cb, void* callback_data,
                  std::size_t counter_count, std::size_t* record_count) {
        { Backend::create_context(context_ptr) };
        { Backend::start_context(context) };
        { Backend::stop_context(context) };
        {
            Backend::make_agent_id(std::uint64_t{})
        } -> std::same_as<typename Backend::agent_id_t>;
        {
            Backend::sample_device_counting_service(context, user_data, flags, records,
                                                    record_count)
        } -> std::same_as<typename Backend::status_t>;
        {
            Backend::query_record_counter_id(record, counter_id)
        } -> std::same_as<typename Backend::status_t>;
        {
            Backend::query_counter_details(counter)
        } -> std::same_as<std::vector<typename Backend::counter_metadata_t>>;
        {
            Backend::iterate_agent_supported_counters(agent, counter_cb, callback_data)
        } -> std::same_as<typename Backend::status_t>;
        {
            Backend::create_counter_config(agent, counters, counter_count, config)
        } -> std::same_as<typename Backend::status_t>;
        {
            Backend::configure_device_counting_service(context, buffer, agent, service_cb,
                                                       callback_data)
        } -> std::same_as<typename Backend::status_t>;
    };

}  // namespace rocprofsys::policies::gpu_perf_counters
