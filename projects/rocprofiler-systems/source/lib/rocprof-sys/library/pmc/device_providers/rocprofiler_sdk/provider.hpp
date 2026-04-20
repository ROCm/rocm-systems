// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
{

struct counter_instance_info
{
    uint64_t    instance_id = 0;
    std::string qualified_name;
};

struct counter_metadata
{
    std::string name;
    std::string description;
    std::string block;
    std::string expression;
    bool        is_constant = false;
    bool        is_derived  = false;
};

struct agent_info
{
    rocprofiler_agent_id_t             agent_id       = {};
    rocprofiler_counter_config_id_t    profile_config = {};
    size_t                             device_index   = 0;
    std::vector<counter_instance_info> instance_infos = {};
    std::vector<counter_metadata>      counter_meta   = {};
};

/**
 * @brief Input for provider construction: agent handle + logical device index.
 */
struct agent_handle
{
    uint64_t handle       = 0;
    size_t   device_index = 0;
};

/**
 * @brief Rocprofiler-SDK device provider for GPU hardware counter sampling.
 *
 * Queries supported counters, intersects with user settings, configures
 * profile and device counting service — all via the Driver abstraction.
 * Called during tool_init so SDK API timing constraints are satisfied.
 *
 * @tparam DriverFactory Factory for creating rocprofiler-sdk driver instances.
 */
template <typename DriverFactory>
class provider
{
public:
    using driver_t = typename DriverFactory::driver_t;

    provider(rocprofiler_context_id_t                             context,
             const std::vector<agent_handle>&                     agent_handles,
             const collectors::gpu_perf_counter::enabled_metrics& enabled)
    : m_driver_api(DriverFactory::create_driver())
    , m_context(context)
    {
        configure_agents(agent_handles, enabled);
    }

    void start()
    {
        auto status = m_driver_api->start_context(m_context);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Failed to start SDK PMC context (status={})",
                        static_cast<int>(status));
        }
    }

    void stop()
    {
        auto status = m_driver_api->stop_context(m_context);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_DEBUG("Failed to stop SDK PMC context (status={})",
                      static_cast<int>(status));
        }
    }

    void shutdown() { stop(); }

    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices(device_type type)
    {
        if(type != device_type::GPU) return {};

        std::vector<std::shared_ptr<Device>> devices;
        devices.reserve(m_agents.size());

        for(const auto& info : m_agents)
        {
            devices.push_back(std::make_shared<Device>(
                m_driver_api, m_context, info.agent_id, info.profile_config,
                info.device_index, info.instance_infos, info.counter_meta));
        }

        return devices;
    }

private:
    void configure_agents(const std::vector<agent_handle>& agent_handles,
                          const collectors::gpu_perf_counter::enabled_metrics& enabled)
    {
        LOG_INFO("Configuring SDK PMC: {} agents, enabled.value={}, collect_all={}, "
                 "counter_names.size={}",
                 agent_handles.size(), enabled.value, enabled.collect_all,
                 enabled.counter_names.size());

        for(const auto& name : enabled.counter_names)
        {
            LOG_INFO("  requested counter: '{}'", name);
        }

        for(const auto& agent : agent_handles)
        {
            auto agent_id = rocprofiler_agent_id_t{ agent.handle };

            auto supported_ids = query_supported_counters(agent_id);
            LOG_INFO("Agent {} (device {}): {} supported counters", agent.handle,
                     agent.device_index, supported_ids.size());
            if(supported_ids.empty())
            {
                continue;
            }

            auto filtered_ids = filter_counters(supported_ids, enabled);
            LOG_INFO("Agent {}: {} counters after filtering", agent.handle,
                     filtered_ids.size());
            if(filtered_ids.empty())
            {
                continue;
            }

            auto profile = rocprofiler_counter_config_id_t{};
            auto status  = m_driver_api->create_profile_config(
                agent_id, filtered_ids.data(), filtered_ids.size(), &profile);
            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                LOG_WARNING("Failed to create profile config for agent {} (status={})",
                            agent.handle, static_cast<int>(status));
                continue;
            }

            auto instance_infos = std::vector<counter_instance_info>{};
            auto counter_meta   = std::vector<counter_metadata>{};
            resolve_counter_details(filtered_ids, instance_infos, counter_meta);

            m_profile_map[agent.handle] = profile;

            status = m_driver_api->configure_device_counting_service(
                m_context, rocprofiler_buffer_id_t{ 0 }, agent_id, set_profile_callback,
                &m_profile_map);
            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                m_profile_map.erase(agent.handle);
                LOG_WARNING(
                    "Failed to configure device counting for agent {} (status={})",
                    agent.handle, static_cast<int>(status));
                continue;
            }

            m_agents.push_back(agent_info{ agent_id, profile, agent.device_index,
                                           std::move(instance_infos),
                                           std::move(counter_meta) });
        }
    }

    [[nodiscard]] std::vector<rocprofiler_counter_id_t> query_supported_counters(
        rocprofiler_agent_id_t agent_id)
    {
        auto result = std::vector<rocprofiler_counter_id_t>{};

        auto callback = [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters,
                           size_t num_counters, void* user_data) -> rocprofiler_status_t {
            auto* out = static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
            out->assign(counters, counters + num_counters);
            return ROCPROFILER_STATUS_SUCCESS;
        };

        m_driver_api->iterate_agent_supported_counters(agent_id, callback, &result);
        return result;
    }

    [[nodiscard]] std::vector<rocprofiler_counter_id_t> filter_counters(
        const std::vector<rocprofiler_counter_id_t>&         supported,
        const collectors::gpu_perf_counter::enabled_metrics& enabled)
    {
        if(enabled.collect_all) return supported;

        auto result = std::vector<rocprofiler_counter_id_t>{};
        for(const auto& counter_id : supported)
        {
            rocprofiler_counter_info_v0_t info{};
            auto                          status = m_driver_api->query_counter_info(
                counter_id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info);
            if(status != ROCPROFILER_STATUS_SUCCESS || info.name == nullptr) continue;

            if(enabled.is_counter_enabled(info.name))
            {
                result.push_back(counter_id);
            }
        }
        return result;
    }

    void resolve_counter_details(const std::vector<rocprofiler_counter_id_t>& counter_ids,
                                 std::vector<counter_instance_info>& instance_infos,
                                 std::vector<counter_metadata>&      counter_meta)
    {
        for(const auto& cid : counter_ids)
        {
            rocprofiler_counter_info_v1_t cinfo{};
            auto                          status = m_driver_api->query_counter_info(
                cid, ROCPROFILER_COUNTER_INFO_VERSION_1, &cinfo);
            if(status != ROCPROFILER_STATUS_SUCCESS || cinfo.name == nullptr) continue;

            auto base_name = std::string{ cinfo.name };

            counter_meta.push_back(counter_metadata{
                base_name, cinfo.description ? cinfo.description : "",
                cinfo.block ? cinfo.block : "", cinfo.expression ? cinfo.expression : "",
                static_cast<bool>(cinfo.is_constant),
                static_cast<bool>(cinfo.is_derived) });

            for(uint64_t inst = 0; inst < cinfo.dimensions_instances_count; ++inst)
            {
                const auto* dim_inst = cinfo.dimensions_instances[inst];
                auto        dims =
                    std::vector<collectors::gpu_perf_counter::dimension_position>{};
                for(uint64_t dim_idx = 0; dim_idx < dim_inst->dimensions_count; ++dim_idx)
                {
                    dims.push_back(
                        { collectors::gpu_perf_counter::abbreviate_dimension_name(
                              dim_inst->dimensions[dim_idx]->dimension_name),
                          dim_inst->dimensions[dim_idx]->index });
                }
                instance_infos.push_back(
                    { dim_inst->instance_id,
                      collectors::gpu_perf_counter::make_qualified_name(base_name,
                                                                        dims) });
            }
        }
    }

    static void set_profile_callback(rocprofiler_context_id_t               ctx,
                                     rocprofiler_agent_id_t                 agent,
                                     rocprofiler_device_counting_agent_cb_t set_config,
                                     void*                                  user_data)
    {
        using profile_map_t =
            std::unordered_map<uint64_t, rocprofiler_counter_config_id_t>;
        auto* map  = static_cast<profile_map_t*>(user_data);
        auto  iter = map->find(agent.handle);
        if(iter != map->end())
        {
            set_config(ctx, iter->second);
        }
    }

    std::shared_ptr<driver_t>                                     m_driver_api;
    rocprofiler_context_id_t                                      m_context;
    std::vector<agent_info>                                       m_agents;
    std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> m_profile_map;
};

}  // namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
