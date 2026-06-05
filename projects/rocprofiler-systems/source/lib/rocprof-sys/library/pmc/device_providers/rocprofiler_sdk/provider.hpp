// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "library/pmc/collectors/gpu_perf_counter/device.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
{

template <typename DriverFactory>
class provider
{
public:
    using driver_t = typename DriverFactory::driver_t;
    using device_t = collectors::gpu_perf_counter::device<driver_t>;

    provider(const std::vector<std::shared_ptr<agent>>&           agent_list,
             const collectors::gpu_perf_counter::enabled_metrics& enabled)
    : m_driver_api{ DriverFactory::create_driver() }
    {
        configure_agents(agent_list, enabled);
    }

    void start()
    {
        for(const auto& device : m_devices)
        {
            device->start();
        }
    }

    void stop()
    {
        for(const auto& device : m_devices)
        {
            device->stop();
        }
    }

    void shutdown() { stop(); }

    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices(device_type type)
    {
        if(type != device_type::GPU) return {};
        return { m_devices.begin(), m_devices.end() };
    }

private:
    void configure_agents(const std::vector<std::shared_ptr<agent>>&           agent_list,
                          const collectors::gpu_perf_counter::enabled_metrics& enabled)
    {
        for(const auto& gpu_agent : agent_list)
        {
            const auto agent_id = driver_t::make_agent_id(gpu_agent->handle);

            auto supported_ids = query_supported_counters(agent_id);
            LOG_INFO("Agent {} (device {}): {} supported counters", gpu_agent->name,
                     gpu_agent->device_id, supported_ids.size());
            if(supported_ids.empty())
            {
                continue;
            }

            auto [filtered_ids, counter_meta] = filter_and_resolve_counter_details(
                supported_ids, enabled, gpu_agent->device_type_index);
            LOG_INFO("Agent {}: {} counters after filtering", gpu_agent->handle,
                     filtered_ids.size());
            if(filtered_ids.empty())
            {
                continue;
            }

            auto profile = typename driver_t::counter_config_id_t{};
            auto status  = m_driver_api->create_counter_config(
                agent_id, filtered_ids.data(), filtered_ids.size(), &profile);
            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                LOG_WARNING("Failed to create profile config for agent {} (status={})",
                            gpu_agent->handle, static_cast<int>(status));
                continue;
            }

            m_profile_configs[gpu_agent->handle] = profile;

            typename driver_t::context_id_t counter_context{};
            status = m_driver_api->create_context(&counter_context);
            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                LOG_WARNING("Failed to create context for agent {} (status={})",
                            gpu_agent->handle, static_cast<int>(status));
                continue;
            }

            status = m_driver_api->configure_device_counting_service(
                counter_context, typename driver_t::buffer_id_t{ 0 }, agent_id,
                [](typename driver_t::context_id_t               ctx,
                   typename driver_t::agent_id_t                 agent_cb,
                   typename driver_t::device_counting_agent_cb_t set_config,
                   void*                                         user_data) {
                    auto* configs = static_cast<std::unordered_map<
                        std::uint64_t, typename driver_t::counter_config_id_t>*>(
                        user_data);
                    auto iter = configs->find(agent_cb.handle);
                    if(iter != configs->end()) set_config(ctx, iter->second);
                },
                &m_profile_configs);
            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                LOG_WARNING(
                    "Failed to configure device counting for agent {} (status={})",
                    gpu_agent->handle, static_cast<int>(status));
                continue;
            }

            m_devices.push_back(std::make_shared<device_t>(m_driver_api, counter_context,
                                                           gpu_agent, profile,
                                                           std::move(counter_meta)));
        }
    }

    [[nodiscard]] std::vector<typename driver_t::counter_id_t> query_supported_counters(
        typename driver_t::agent_id_t agent_id) const
    {
        const auto collect_counters =
            [](typename driver_t::agent_id_t, typename driver_t::counter_id_t* counters,
               size_t num_counters, void* user_data) -> typename driver_t::status_t {
            auto* out =
                static_cast<std::vector<typename driver_t::counter_id_t>*>(user_data);
            out->insert(out->end(), counters, counters + num_counters);
            return ROCPROFILER_STATUS_SUCCESS;
        };

        auto       result = std::vector<typename driver_t::counter_id_t>{};
        const auto status = m_driver_api->iterate_agent_supported_counters(
            agent_id, collect_counters, &result);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_DEBUG("No counters found for agent {} (status={})", agent_id.handle,
                      static_cast<int>(status));
            return {};
        }
        return result;
    }

    [[nodiscard]] std::pair<std::vector<typename driver_t::counter_id_t>,
                            std::vector<collectors::gpu_perf_counter::counter_metadata>>
    filter_and_resolve_counter_details(
        const std::vector<typename driver_t::counter_id_t>&  supported,
        const collectors::gpu_perf_counter::enabled_metrics& enabled, size_t device_index)
    {
        auto ids  = std::vector<typename driver_t::counter_id_t>{};
        auto meta = std::vector<collectors::gpu_perf_counter::counter_metadata>{};
        ids.reserve(supported.size());
        meta.reserve(supported.size());

        for(const auto& counter_id : supported)
        {
            auto details = m_driver_api->query_counter_details(counter_id);
            if(details.empty()) continue;
            if(!enabled.is_counter_enabled({ details.front().name, device_index }))
                continue;
            ids.push_back(counter_id);
            meta.insert(meta.end(), std::make_move_iterator(details.begin()),
                        std::make_move_iterator(details.end()));
        }

        return { std::move(ids), std::move(meta) };
    }

    std::shared_ptr<driver_t>              m_driver_api;
    std::vector<std::shared_ptr<device_t>> m_devices;
    // Must outlive the context: the SDK callback set via
    // configure_device_counting_service fires during start_context(), reading from this
    // map via the user_data pointer.
    std::unordered_map<std::uint64_t, typename driver_t::counter_config_id_t>
        m_profile_configs;
};

}  // namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
