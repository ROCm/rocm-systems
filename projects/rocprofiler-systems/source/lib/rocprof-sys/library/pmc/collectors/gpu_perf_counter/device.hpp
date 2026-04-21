// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "logger/debug.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

template <typename Driver>
class device
{
public:
    device(std::shared_ptr<Driver> driver, rocprofiler_context_id_t context,
           std::shared_ptr<rocprofsys::agent> agent,
           rocprofiler_counter_config_id_t    profile_config,
           std::vector<counter_metadata>      counter_meta)
    : m_driver_api{ std::move(driver) }
    , m_context{ context }
    , m_agent{ std::move(agent) }
    , m_profile_config{ profile_config }
    , m_counter_meta{ std::move(counter_meta) }
    {
        m_record_buffer.resize(std::max<size_t>(m_counter_meta.size() * 2, 256));
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

    [[nodiscard]] metrics get_gpu_perf_counter_metrics(const enabled_metrics& /*enabled*/,
                                                       uint64_t /*timestamp*/)
    {
        metrics result{};

        auto rec_count = m_record_buffer.size();

        const auto status = m_driver_api->sample_device_counting_service(
            m_context, {}, ROCPROFILER_COUNTER_FLAG_NONE, m_record_buffer.data(),
            &rec_count);

        if(status == ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED)
        {
            LOG_DEBUG("HSA not loaded for device {} (status={}). Ignoring error.",
                      m_agent->device_type_index, static_cast<int>(status));
            return result;
        }

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Sample failed for device {} (status={})",
                        m_agent->device_type_index, static_cast<int>(status));
            return result;
        }

        result.reserve(rec_count);
        for(size_t idx = 0; idx < rec_count; ++idx)
        {
            const auto& record = m_record_buffer.at(idx);
            result.push_back({ record.id, record.counter_value });
        }

        return result;
    }

private:
    std::shared_ptr<Driver>                   m_driver_api;
    rocprofiler_context_id_t                  m_context;
    std::shared_ptr<rocprofsys::agent>        m_agent;
    rocprofiler_counter_config_id_t           m_profile_config;
    std::vector<counter_metadata>             m_counter_meta;
    std::vector<rocprofiler_counter_record_t> m_record_buffer;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
