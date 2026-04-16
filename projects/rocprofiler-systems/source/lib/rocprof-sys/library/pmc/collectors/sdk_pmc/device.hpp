// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/sdk_pmc/types.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/provider.hpp"
#include "logger/debug.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys::pmc::collectors::sdk_pmc
{

/**
 * @brief SDK PMC device wrapping a single GPU agent's device counting service.
 *
 * Each device holds a reference to the shared rocprofiler context and its own
 * per-agent profile configuration. Polling is done via
 * rocprofiler_sample_device_counting_service().
 *
 * An instance_id → qualified_name map is built at construction from pre-resolved
 * dimension info (provided by tool_init via v1 counter info). Sampling performs
 * only a map lookup per record — no SDK queries in the hot path.
 *
 * @tparam Driver The rocprofiler-sdk driver type (real or mock for testing).
 */
template <typename Driver>
class device
{
public:
    using instance_info_vec =
        std::vector<device_providers::rocprofiler_sdk::counter_instance_info>;

    device(std::shared_ptr<Driver> driver, rocprofiler_context_id_t context,
           rocprofiler_agent_id_t          agent_id,
           rocprofiler_counter_config_id_t profile_config, size_t logical_index,
           std::vector<std::string> counter_names, instance_info_vec instance_infos = {})
    : m_driver_api{ std::move(driver) }
    , m_context{ context }
    , m_agent_id{ agent_id }
    , m_profile_config{ profile_config }
    , m_index{ logical_index }
    , m_counter_names{ std::move(counter_names) }
    , m_vendor_name("AMD")
    , m_is_supported(true)
    {
        m_device_name  = fmt::format("GPU {}", m_index);
        m_product_name = fmt::format("GPU {}", m_index);

        m_supported_metrics.value = 1;  // non-zero = has counters

        // Build instance_id → qualified_name lookup from pre-resolved info
        for(auto& info : instance_infos)
        {
            m_instance_map[info.instance_id] = std::move(info.qualified_name);
        }
    }

    [[nodiscard]] bool is_supported() const noexcept { return m_is_supported; }

    [[nodiscard]] enabled_metrics get_supported_metrics() const noexcept
    {
        return m_supported_metrics;
    }

    [[nodiscard]] size_t get_index() const noexcept { return m_index; }

    [[nodiscard]] const std::string& get_name() const noexcept { return m_device_name; }

    [[nodiscard]] const std::string& get_product_name() const noexcept
    {
        return m_product_name;
    }

    [[nodiscard]] const std::string& get_vendor_name() const noexcept
    {
        return m_vendor_name;
    }

    [[nodiscard]] rocprofiler_agent_id_t get_agent_id() const noexcept
    {
        return m_agent_id;
    }

    [[nodiscard]] rocprofiler_counter_config_id_t get_profile_config() const noexcept
    {
        return m_profile_config;
    }

    [[nodiscard]] const std::vector<std::string>& get_counter_names() const noexcept
    {
        return m_counter_names;
    }

    /**
     * @brief Get the list of qualified counter names for all dimension instances.
     *
     * Returns the values from the instance_id map, which are the fully qualified
     * names like "SQC_ICACHE_HITS[WGP=0,SA=0,SE=0]".
     */
    [[nodiscard]] std::vector<std::string> get_qualified_names() const
    {
        std::vector<std::string> names;
        names.reserve(m_instance_map.size());
        for(const auto& [instance_id, counter_name] : m_instance_map)
        {
            names.push_back(counter_name);
        }
        return names;
    }

    /**
     * @brief Poll GPU hardware counters via device_counting_service.
     *
     * Calls rocprofiler_sample_device_counting_service() and looks up each
     * returned record's instance_id in the pre-built map to get the qualified
     * counter name. No SDK queries in the hot path.
     *
     * @param enabled Which counters to collect (currently unused — all configured
     *        counters are always sampled).
     * @param timestamp Current timestamp in nanoseconds.
     * @return Collected counter values.
     */
    [[nodiscard]] metrics get_sdk_pmc_metrics(
        [[maybe_unused]] const enabled_metrics& enabled,
        [[maybe_unused]] uint64_t               timestamp)
    {
        metrics result{};

        static constexpr size_t      MAX_RECORDS = 32768;
        rocprofiler_counter_record_t output_records[MAX_RECORDS];
        size_t                       rec_count = MAX_RECORDS;

        LOG_DEBUG("Sampling device {} (context={}, agent={}, profile={})", m_index,
                  m_context.handle, m_agent_id.handle, m_profile_config.handle);

        const auto status = m_driver_api->sample_device_counting_service(
            m_context, {}, ROCPROFILER_COUNTER_FLAG_NONE, output_records, &rec_count);

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Sample failed for device {} (status={})", m_index,
                        static_cast<int>(status));
            return result;
        }

        LOG_DEBUG("Device {} returned {} counter records", m_index, rec_count);

        result.counters.reserve(rec_count);

        for(size_t i = 0; i < rec_count; ++i)
        {
            auto map_iter = m_instance_map.find(output_records[i].id);
            if(map_iter == m_instance_map.end())
            {
                LOG_DEBUG("Device {} record {} — unknown instance_id {}", m_index, i,
                          output_records[i].id);
                continue;
            }

            LOG_DEBUG("Device {} counter: {} = {}", m_index, map_iter->second,
                      output_records[i].counter_value);

            result.counters.push_back(counter_value{ output_records[i].id,
                                                     map_iter->second,
                                                     output_records[i].counter_value });
        }

        return result;
    }

private:
    std::shared_ptr<Driver>         m_driver_api;
    rocprofiler_context_id_t        m_context;
    rocprofiler_agent_id_t          m_agent_id;
    rocprofiler_counter_config_id_t m_profile_config;
    size_t                          m_index;
    std::vector<std::string>        m_counter_names;
    std::string                     m_device_name;
    std::string                     m_product_name;
    std::string                     m_vendor_name;
    enabled_metrics                 m_supported_metrics;
    bool                            m_is_supported = false;

    // instance_id → qualified counter name (built at construction)
    std::unordered_map<uint64_t, std::string> m_instance_map;
};

}  // namespace rocprofsys::pmc::collectors::sdk_pmc
