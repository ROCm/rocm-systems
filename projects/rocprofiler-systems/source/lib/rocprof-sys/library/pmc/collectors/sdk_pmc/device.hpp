// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/sdk_pmc/types.hpp"
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
 * Counter names are provided at construction time (resolved during tool_init
 * from ROCPROFSYS_ROCM_EVENTS). On the first sample call, a mapping from
 * counter_id → index is built so subsequent samples skip name queries entirely.
 *
 * @tparam Driver The rocprofiler-sdk driver type (real or mock for testing).
 */
template <typename Driver>
class device
{
public:
    device(std::shared_ptr<Driver> driver, rocprofiler_context_id_t context,
           rocprofiler_agent_id_t          agent_id,
           rocprofiler_counter_config_id_t profile_config, size_t logical_index,
           std::vector<std::string> counter_names)
    : m_driver_api{ std::move(driver) }
    , m_context{ context }
    , m_agent_id{ agent_id }
    , m_profile_config{ profile_config }
    , m_index{ logical_index }
    , m_counter_names{ std::move(counter_names) }
    {
        m_device_name  = "GPU" + std::to_string(m_index);
        m_product_name = "GPU " + std::to_string(m_index);
        m_vendor_name  = "AMD";

        m_is_supported            = true;
        m_supported_metrics.value = 1;  // non-zero = has counters
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
     * @brief Poll GPU hardware counters via device_counting_service.
     *
     * Calls rocprofiler_sample_device_counting_service() and decodes the returned
     * counter records into a metrics struct. On the first call, builds a mapping
     * from counter_id → index into m_counter_names so subsequent calls skip
     * name queries entirely.
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

        auto status = m_driver_api->sample_device_counting_service(
            m_context, {}, ROCPROFILER_COUNTER_FLAG_NONE, output_records, &rec_count);

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Sample failed for device {} (status={})", m_index,
                        static_cast<int>(status));
            return result;
        }

        LOG_DEBUG("Device {} returned {} counter records", m_index, rec_count);

        const bool need_name_init = !m_counter_id_map_initialized;

        result.counters.reserve(rec_count);

        for(size_t i = 0; i < rec_count; ++i)
        {
            rocprofiler_counter_id_t counter_id{};
            auto                     query_status =
                m_driver_api->query_record_counter_id(output_records[i].id, &counter_id);

            if(query_status != ROCPROFILER_STATUS_SUCCESS)
            {
                LOG_DEBUG("Device {} record {} — failed to query counter id", m_index, i);
                continue;
            }

            // On first sample, resolve counter_id → name via SDK query
            if(need_name_init && m_counter_id_to_name.count(counter_id.handle) == 0)
            {
                rocprofiler_counter_info_v0_t info{};
                auto info_status = m_driver_api->query_counter_info(
                    counter_id, ROCPROFILER_COUNTER_INFO_VERSION_0,
                    static_cast<void*>(&info));

                if(info_status == ROCPROFILER_STATUS_SUCCESS && info.name != nullptr)
                {
                    m_counter_id_to_name[counter_id.handle] = info.name;
                }
                else
                {
                    m_counter_id_to_name[counter_id.handle] =
                        "counter_" + std::to_string(counter_id.handle);
                }
            }

            auto        name_it = m_counter_id_to_name.find(counter_id.handle);
            const auto& name    = (name_it != m_counter_id_to_name.end())
                                      ? name_it->second
                                      : "counter_" + std::to_string(counter_id.handle);

            LOG_DEBUG("Device {} counter: {} = {}", m_index, name,
                      output_records[i].counter_value);

            result.counters.push_back(counter_value{ counter_id.handle, name,
                                                     output_records[i].counter_value });
        }

        m_counter_id_map_initialized = true;

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
    bool                            m_is_supported               = false;
    bool                            m_counter_id_map_initialized = false;

    // counter_id.handle → counter name (built on first sample)
    std::unordered_map<uint64_t, std::string> m_counter_id_to_name;
};

}  // namespace rocprofsys::pmc::collectors::sdk_pmc
