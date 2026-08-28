// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/agent.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

// Contract the GPU perf-counter collector requires of its backend (rocprofiler-sdk).
template <typename Backend>
concept backend_contract = requires(
    Backend backend, typename Backend::context_id_t context,
    typename Backend::agent_id_t agent, typename Backend::buffer_id_t buffer,
    typename Backend::user_data_t user_data, typename Backend::counter_flag_t flags,
    typename Backend::counter_record_t  record,
    typename Backend::counter_record_t* records,
    typename Backend::counter_id_t* counter_id, typename Backend::counter_id_t counter,
    typename Backend::counter_id_t*                counters,
    typename Backend::counter_config_id_t*         config,
    typename Backend::available_counters_cb_t      counter_cb,
    typename Backend::device_counting_service_cb_t service_cb, size_t* record_count,
    void* data) {
    {
        Backend::make_agent_id(std::uint64_t{})
    } -> std::same_as<typename Backend::agent_id_t>;
    { backend.create_context(&context) } -> std::same_as<typename Backend::status_t>;
    { backend.start_context(context) } -> std::same_as<typename Backend::status_t>;
    { backend.stop_context(context) } -> std::same_as<typename Backend::status_t>;
    {
        backend.sample_device_counting_service(context, user_data, flags, records,
                                               record_count)
    } -> std::same_as<typename Backend::status_t>;
    {
        backend.query_record_counter_id(record, counter_id)
    } -> std::same_as<typename Backend::status_t>;
    {
        backend.query_counter_details(counter)
    } -> std::same_as<std::vector<counter_metadata>>;
    {
        backend.iterate_agent_supported_counters(agent, counter_cb, data)
    } -> std::same_as<typename Backend::status_t>;
    {
        backend.create_counter_config(agent, counters, size_t{}, config)
    } -> std::same_as<typename Backend::status_t>;
    {
        backend.configure_device_counting_service(context, buffer, agent, service_cb,
                                                  data)
    } -> std::same_as<typename Backend::status_t>;
    // Status/flag constants read directly by device<Backend> during sampling.
    { Backend::flag_none } -> std::convertible_to<typename Backend::counter_flag_t>;
    { Backend::status_success } -> std::convertible_to<typename Backend::status_t>;
    { Backend::status_hsa_not_loaded } -> std::convertible_to<typename Backend::status_t>;
};

template <backend_contract Backend>
class device
{
public:
    device(std::shared_ptr<Backend> backend, typename Backend::context_id_t context,
           std::shared_ptr<rocprofsys::agent>    agent,
           typename Backend::counter_config_id_t counter_config,
           std::vector<counter_metadata>         counter_meta)
    : m_backend_api{ std::move(backend) }
    , m_context{ context }
    , m_agent{ std::move(agent) }
    , m_counter_config{ counter_config }
    , m_counter_meta{ std::move(counter_meta) }
    {
        // Each counter may produce multiple dimension instances (e.g. per-WGP);
        // the headroom factor gives slack beyond the enumerated metadata count.
        // The minimum size guards against an under-sized buffer on sparse devices.
        m_record_buffer.resize(
            std::max<size_t>(m_counter_meta.size() * k_record_buffer_headroom_factor,
                             k_record_buffer_min_size));
    }

    [[nodiscard]] bool is_supported() const noexcept { return !m_counter_meta.empty(); }

    [[nodiscard]] size_t get_index() const noexcept { return m_agent->device_type_index; }

    [[nodiscard]] const std::string& get_name() const noexcept { return m_agent->name; }

    [[nodiscard]] const std::string& get_product_name() const noexcept
    {
        return m_agent->product_name;
    }

    [[nodiscard]] const std::string& get_vendor_name() const noexcept
    {
        return m_agent->vendor_name;
    }

    [[nodiscard]] const std::vector<counter_metadata>& get_counter_metadata()
        const noexcept
    {
        return m_counter_meta;
    }

    [[nodiscard]] const metrics& sample_metrics(
        [[maybe_unused]] const enabled_metrics& enabled,
        [[maybe_unused]] std::uint64_t          timestamp)
    {
        m_result_cache.clear();

        // start_context requires HSA to be live. It is called eagerly in start(), but
        // tool_init runs before the application's hsa_init, so the first call may fail.
        // Retry here until it succeeds; once m_context_started is true this is a single
        // branch-predicted branch with no further work.
        if(!m_context_started) start();
        if(!m_context_started) return m_result_cache;

        auto rec_count = m_record_buffer.size();

        const auto status = m_backend_api->sample_device_counting_service(
            m_context, {}, Backend::flag_none, m_record_buffer.data(), &rec_count);

        if(status == Backend::status_hsa_not_loaded)
        {
            LOG_DEBUG("HSA not loaded for device {} (status={}). Ignoring error.",
                      m_agent->device_type_index, static_cast<int>(status));
            return m_result_cache;
        }

        if(status != Backend::status_success)
        {
            LOG_WARNING("Sample failed for device {} (status={})",
                        m_agent->device_type_index, static_cast<int>(status));
            return m_result_cache;
        }

        // SDK writes back the actual number of records filled; it must not exceed
        // the buffer capacity we provided.
        assert(rec_count <= m_record_buffer.size());

        m_result_cache.reserve(rec_count);
        for(size_t idx = 0; idx < rec_count; ++idx)
        {
            const auto& record = m_record_buffer[idx];

            typename Backend::counter_id_t config_id{};
            m_backend_api->query_record_counter_id(record, &config_id);
            auto         id     = config_id.handle;
            const double raw    = record.counter_value;
            auto [it, inserted] = m_prev_values.try_emplace(id, raw);
            const double delta  = inserted ? raw : raw - it->second;
            if(!inserted) it->second = raw;
            m_result_cache.push_back({ id, delta });
        }

        return m_result_cache;
    }

    void start()
    {
        if(m_context_started) return;

        auto status = m_backend_api->start_context(m_context);
        if(status == Backend::status_success)
        {
            m_context_started = true;
            m_prev_values.clear();
            LOG_DEBUG("GPU PMC context started for device {}.",
                      m_agent->device_type_index);
        }
        else
        {
            // HSA may not be initialized yet at this call site. The hsa_init callback
            // registered in tool_init will call start() again once HSA is live.
            LOG_DEBUG("GPU PMC context start deferred for device {} (status={}).",
                      m_agent->device_type_index, static_cast<int>(status));
        }
    }

    void stop()
    {
        auto status = m_backend_api->stop_context(m_context);
        if(status != Backend::status_success)
        {
            LOG_WARNING("Failed to stop context for device {} (status={})",
                        m_agent->device_type_index, static_cast<int>(status));
        }
    }

private:
    static constexpr size_t k_record_buffer_headroom_factor = 2;
    static constexpr size_t k_record_buffer_min_size        = 256;

    std::shared_ptr<Backend>                        m_backend_api;
    typename Backend::context_id_t                  m_context;
    std::shared_ptr<rocprofsys::agent>              m_agent;
    typename Backend::counter_config_id_t           m_counter_config;
    std::vector<counter_metadata>                   m_counter_meta;
    std::vector<typename Backend::counter_record_t> m_record_buffer;
    metrics                                         m_result_cache;
    std::unordered_map<counter_id_t, double>        m_prev_values;
    bool                                            m_context_started = false;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
