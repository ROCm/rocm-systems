// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/common/types.hpp"
#include "logger/debug.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
{

/**
 * @brief Pre-built mapping from SDK instance_id to qualified counter name.
 *
 * Built during tool_init from rocprofiler_counter_info_v1_t dimension instances.
 */
struct counter_instance_info
{
    uint64_t    instance_id = 0;  ///< rocprofiler_counter_instance_id_t value
    std::string qualified_name;   ///< e.g. "SQC_ICACHE_HITS[WGP=0,SA=0,SE=0]"
};

/**
 * @brief Per-counter metadata extracted from rocprofiler_counter_info_v1_t.
 *
 * Carries SDK counter properties that feed into pmc_info registration
 * (block, expression, is_constant, is_derived). Built once during tool_init,
 * propagated through provider → device → cache_policy.
 */
struct counter_metadata
{
    std::string name;
    std::string description;
    std::string block;
    std::string expression;
    bool        is_constant = false;
    bool        is_derived  = false;
};

/**
 * @brief Per-agent info for device_counting_service.
 */
struct agent_info
{
    rocprofiler_agent_id_t             agent_id       = {};
    rocprofiler_counter_config_id_t    profile_config = {};
    size_t                             device_index   = 0;
    std::vector<std::string>           counter_names  = {};
    std::vector<counter_instance_info> instance_infos = {};
    std::vector<counter_metadata>      counter_meta   = {};
};

/**
 * @brief Rocprofiler-SDK device provider for GPU hardware counter sampling.
 *
 * Receives a pre-configured context and agent list via constructor injection
 * from rocprofiler-sdk.cpp (tool_init). Manages the context lifecycle after
 * creation: start, stop, shutdown.
 *
 * @tparam DriverFactory Factory for creating rocprofiler-sdk driver instances.
 */
template <typename DriverFactory>
class provider
{
public:
    using driver_t = typename DriverFactory::driver_t;

    provider(rocprofiler_context_id_t context, std::vector<agent_info> agents)
    : m_driver_api(DriverFactory::create_driver())
    , m_context(context)
    , m_agents(std::move(agents))
    {}

    /**
     * @brief Start the SDK context. Call before sampling begins.
     */
    void start()
    {
        auto status = m_driver_api->start_context(m_context);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Failed to start SDK PMC context (status={})",
                        static_cast<int>(status));
        }
    }

    /**
     * @brief Stop the SDK context. Call when sampling pauses or ends.
     */
    void stop()
    {
        auto status = m_driver_api->stop_context(m_context);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_DEBUG("Failed to stop SDK PMC context (status={})",
                      static_cast<int>(status));
        }
    }

    /**
     * @brief Shutdown — stops the context.
     */
    void shutdown() { stop(); }

    /**
     * @brief Create device objects from the stored agent list.
     *
     * @tparam Device The device type to create.
     * @param type Device type (only GPU is supported).
     */
    template <typename Device>
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices(device_type type)
    {
        if(type != device_type::GPU)
        {
            return {};
        }

        std::vector<std::shared_ptr<Device>> devices;
        devices.reserve(m_agents.size());

        for(const auto& info : m_agents)
        {
            devices.push_back(std::make_shared<Device>(
                m_driver_api, m_context, info.agent_id, info.profile_config,
                info.device_index, info.counter_names, info.instance_infos,
                info.counter_meta));
        }

        return devices;
    }

private:
    std::shared_ptr<typename DriverFactory::driver_t> m_driver_api;
    rocprofiler_context_id_t                          m_context;
    std::vector<agent_info>                           m_agents;
};

}  // namespace rocprofsys::pmc::device_providers::rocprofiler_sdk
