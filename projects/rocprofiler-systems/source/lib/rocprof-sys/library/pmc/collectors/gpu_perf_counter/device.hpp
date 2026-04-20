// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/provider.hpp"
#include "logger/debug.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
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
    using counter_meta_vec =
        std::vector<device_providers::rocprofiler_sdk::counter_metadata>;

    device(std::shared_ptr<Driver> driver, rocprofiler_context_id_t context,
           rocprofiler_agent_id_t          agent_id,
           rocprofiler_counter_config_id_t profile_config, size_t logical_index,
           instance_info_vec instance_infos = {}, counter_meta_vec counter_meta = {})
    : m_driver_api{ std::move(driver) }
    , m_context{ context }
    , m_agent_id{ agent_id }
    , m_profile_config{ profile_config }
    , m_index{ logical_index }
    , m_counter_meta{ std::move(counter_meta) }
    {
        m_device_name  = fmt::format("GPU {}", m_index);
        m_product_name = fmt::format("GPU {}", m_index);

        m_supported_metrics.value = 1;

        m_supported_metrics.capabilities.reserve(m_counter_meta.size());
        for(const auto& meta : m_counter_meta)
        {
            m_supported_metrics.capabilities.push_back(
                counter_capability{ meta.name, meta.is_derived, meta.is_constant });
        }

        for(auto& info : instance_infos)
        {
            m_instance_map[info.instance_id] = std::move(info.qualified_name);
        }

        m_record_buffer.resize(m_instance_map.size());
    }

    [[nodiscard]] bool is_supported() const noexcept { return !m_instance_map.empty(); }

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
        static const std::string vendor = "AMD";
        return vendor;
    }

    [[nodiscard]] rocprofiler_agent_id_t get_agent_id() const noexcept
    {
        return m_agent_id;
    }

    [[nodiscard]] rocprofiler_counter_config_id_t get_profile_config() const noexcept
    {
        return m_profile_config;
    }

    [[nodiscard]] const counter_meta_vec& get_counter_metadata() const noexcept
    {
        return m_counter_meta;
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
    [[nodiscard]] metrics get_gpu_perf_counter_metrics(
        [[maybe_unused]] const enabled_metrics& enabled,
        [[maybe_unused]] uint64_t               timestamp)
    {
        metrics result{};

        size_t rec_count = m_record_buffer.size();

        LOG_DEBUG("Sampling device {} (context={}, agent={}, profile={})", m_index,
                  m_context.handle, m_agent_id.handle, m_profile_config.handle);

        const auto status = m_driver_api->sample_device_counting_service(
            m_context, {}, ROCPROFILER_COUNTER_FLAG_NONE, m_record_buffer.data(),
            &rec_count);

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
            auto map_iter = m_instance_map.find(m_record_buffer[i].id);
            if(map_iter == m_instance_map.end())
            {
                LOG_DEBUG("Device {} record {} — unknown instance_id {}", m_index, i,
                          m_record_buffer[i].id);
                continue;
            }

            LOG_DEBUG("Device {} counter: {} = {}", m_index, map_iter->second,
                      m_record_buffer[i].counter_value);

            result.counters.push_back(counter_value{ m_record_buffer[i].id,
                                                     map_iter->second,
                                                     m_record_buffer[i].counter_value });
        }

        return result;
    }

private:
    std::shared_ptr<Driver>         m_driver_api;
    rocprofiler_context_id_t        m_context;
    rocprofiler_agent_id_t          m_agent_id;
    rocprofiler_counter_config_id_t m_profile_config;
    size_t                          m_index;
    counter_meta_vec                m_counter_meta;
    std::string                     m_device_name;
    std::string                     m_product_name;
    enabled_metrics                 m_supported_metrics;

    // instance_id → qualified counter name (built at construction, ordered for
    // deterministic iteration)
    std::map<uint64_t, std::string> m_instance_map;

    // pre-allocated buffer for SDK sample output (sized from m_instance_map at
    // construction)
    std::vector<rocprofiler_counter_record_t> m_record_buffer;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
