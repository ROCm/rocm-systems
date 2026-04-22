// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "library/pmc/collectors/gpu_perf_counter/device.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
{

using namespace collectors::gpu_perf_counter;

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
            const auto agent_id = rocprofiler_agent_id_t{ gpu_agent->handle };

            auto supported_ids = query_supported_counters(agent_id);
            LOG_INFO("Agent {} (device {}): {} supported counters", gpu_agent->name,
                     gpu_agent->device_id, supported_ids.size());
            if(supported_ids.empty())
            {
                continue;
            }

            auto filtered_ids =
                filter_counters(supported_ids, enabled, gpu_agent->device_type_index);
            LOG_INFO("Agent {}: {} counters after filtering", gpu_agent->handle,
                     filtered_ids.size());
            if(filtered_ids.empty())
            {
                continue;
            }

            auto profile = rocprofiler_counter_config_id_t{};
            auto status  = m_driver_api->create_counter_config(
                agent_id, filtered_ids.data(), filtered_ids.size(), &profile);
            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                LOG_WARNING("Failed to create profile config for agent {} (status={})",
                            gpu_agent->handle, static_cast<int>(status));
                continue;
            }

            m_profile_configs[gpu_agent->handle] = profile;

            rocprofiler_context_id_t counter_context{};
            status = m_driver_api->create_context(&counter_context);
            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                LOG_WARNING("Failed to create context for agent {} (status={})",
                            gpu_agent->handle, static_cast<int>(status));
                continue;
            }

            status = m_driver_api->configure_device_counting_service(
                counter_context, rocprofiler_buffer_id_t{ 0 }, agent_id,
                [](rocprofiler_context_id_t ctx, rocprofiler_agent_id_t agent_cb,
                   rocprofiler_device_counting_agent_cb_t set_config, void* user_data) {
                    auto* configs = static_cast<
                        std::unordered_map<uint64_t, rocprofiler_counter_config_id_t>*>(
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

            const auto counter_meta = resolve_counter_details(filtered_ids);

            m_devices.push_back(std::make_shared<device_t>(m_driver_api, counter_context,
                                                           gpu_agent, profile,
                                                           std::move(counter_meta)));
        }
    }

    [[nodiscard]] std::vector<rocprofiler_counter_id_t> query_supported_counters(
        rocprofiler_agent_id_t agent_id) const
    {
        static constexpr auto collect_counters =
            [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters,
               size_t num_counters, void* user_data) -> rocprofiler_status_t {
            auto* out = static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
            out->insert(out->end(), counters, counters + num_counters);
            return ROCPROFILER_STATUS_SUCCESS;
        };

        auto       result = std::vector<rocprofiler_counter_id_t>{};
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

    [[nodiscard]] std::vector<rocprofiler_counter_id_t> filter_counters(
        const std::vector<rocprofiler_counter_id_t>&         supported,
        const collectors::gpu_perf_counter::enabled_metrics& enabled, size_t device_index)
    {
        auto result = std::vector<rocprofiler_counter_id_t>{};
        result.reserve(supported.size());
        std::copy_if(supported.begin(), supported.end(), std::back_inserter(result),
                     [&](const rocprofiler_counter_id_t& counter_id) {
                         rocprofiler_counter_info_v1_t info{};
                         const auto status = m_driver_api->query_counter_info(
                             counter_id, ROCPROFILER_COUNTER_INFO_VERSION_1, &info);

                         if(status != ROCPROFILER_STATUS_SUCCESS || info.name == nullptr)
                         {
                             return false;
                         }

                         return enabled.is_counter_enabled(
                             { std::string{ info.name }, device_index });
                     });
        return result;
    }

    [[nodiscard]] std::vector<counter_metadata> resolve_counter_details(
        const std::vector<rocprofiler_counter_id_t>& counter_ids)
    {
        auto result = std::vector<counter_metadata>{};
        result.reserve(counter_ids.size());

        auto safe_str = [](const char* str) {
            return str != nullptr ? std::string{ str } : std::string{};
        };

        auto build_dims =
            [](const rocprofiler_counter_record_dimension_instance_info_t* dim_inst) {
                auto dims = std::vector<dimension_position>{};
                dims.reserve(dim_inst->dimensions_count);
                std::transform(dim_inst->dimensions,
                               dim_inst->dimensions + dim_inst->dimensions_count,
                               std::back_inserter(dims), [](const auto* dim) {
                                   return dimension_position{
                                       std::string{ dim->dimension_name }, dim->index
                                   };
                               });
                return dims;
            };

        for(const auto& cid : counter_ids)
        {
            rocprofiler_counter_info_v1_t cinfo{};
            const auto                    status = m_driver_api->query_counter_info(
                cid, ROCPROFILER_COUNTER_INFO_VERSION_1, &cinfo);
            if(status != ROCPROFILER_STATUS_SUCCESS || cinfo.name == nullptr) continue;

            const auto base_name   = std::string{ cinfo.name };
            const auto description = safe_str(cinfo.description);
            const auto block       = safe_str(cinfo.block);
            const auto expression  = safe_str(cinfo.expression);
            const auto is_constant = static_cast<bool>(cinfo.is_constant);
            const auto is_derived  = static_cast<bool>(cinfo.is_derived);

            auto make_meta = [&](counter_id_t                    counter_id,
                                 std::vector<dimension_position> dims) {
                return counter_metadata{ counter_id, base_name,      description,
                                         block,      expression,     is_constant,
                                         is_derived, std::move(dims) };
            };

            std::transform(cinfo.dimensions_instances,
                           cinfo.dimensions_instances + cinfo.dimensions_instances_count,
                           std::back_inserter(result), [&](const auto* dim_inst) {
                               return make_meta(dim_inst->instance_id,
                                                build_dims(dim_inst));
                           });
        }

        return result;
    }

    std::shared_ptr<driver_t>              m_driver_api;
    std::vector<std::shared_ptr<device_t>> m_devices;
    // Must outlive the context: the SDK callback set via
    // configure_device_counting_service fires during start_context(), reading from this
    // map via the user_data pointer.
    std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> m_profile_configs;
};

}  // namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
