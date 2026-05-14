// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/agent.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "logger/debug.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <cassert>
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
           std::shared_ptr<rocprofsys::agent>   agent,
           typename Driver::counter_config_id_t profile_config,
           std::vector<counter_metadata>        counter_meta)
    : m_driver_api{ std::move(driver) }
    , m_context{ context }
    , m_agent{ std::move(agent) }
    , m_profile_config{ profile_config }
    , m_counter_meta{ std::move(counter_meta) }
    {
        // *2: each counter may produce multiple dimension instances (e.g. per-WGP).
        // The factor of 2 gives headroom beyond the metadata count, which only
        // covers distinct (name, dimension) combinations enumerated at init time.
        // 256 is the floor to avoid an under-sized buffer for low-counter devices.
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
                                                       std::uint64_t /*timestamp*/)
    {
        m_result_cache.clear();

        auto rec_count = m_record_buffer.size();

        const auto status = m_driver_api->sample_device_counting_service(
            m_context, {}, ROCPROFILER_COUNTER_FLAG_NONE, m_record_buffer.data(),
            &rec_count);

        if(status == ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED)
        {
            LOG_DEBUG("HSA not loaded for device {} (status={}). Ignoring error.",
                      m_agent->device_type_index, static_cast<int>(status));
            return std::move(m_result_cache);
        }

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Sample failed for device {} (status={})",
                        m_agent->device_type_index, static_cast<int>(status));
            return std::move(m_result_cache);
        }

        // SDK writes back the actual number of records filled; it must not exceed
        // the buffer capacity we provided.
        assert(rec_count <= m_record_buffer.size());

        m_result_cache.reserve(rec_count);
        for(size_t idx = 0; idx < rec_count; ++idx)
        {
            const auto& record = m_record_buffer[idx];

            // On ROCm < 7.0, record.id is rocprofiler_counter_instance_id_t (an
            // encoded value combining counter handle + dimension positions). The
            // metadata key stored by query_counter_details is counter_id.handle (a
            // plain counter handle), so we must decode via query_record_counter_id.
            // On ROCm >= 7.0, metadata is keyed by dim_inst->instance_id which equals
            // record.id directly, so no decode is required.
#if ROCPROFSYS_ROCM_VERSION < 70000
            typename Driver::counter_id_t cid{};
            m_driver_api->query_record_counter_id(record.id, &cid);
            m_result_cache.push_back({ cid.handle, record.counter_value });
#else
            m_result_cache.push_back({ record.id, record.counter_value });
#endif
        }

        return std::move(m_result_cache);
    }

    void start()
    {
        auto status = m_driver_api->start_context(m_context);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Failed to start context for device {} (status={})",
                        m_agent->device_type_index, static_cast<int>(status));
        }
    }

    void stop()
    {
        auto status = m_driver_api->stop_context(m_context);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Failed to stop context for device {} (status={})",
                        m_agent->device_type_index, static_cast<int>(status));
        }
    }

private:
    std::shared_ptr<Driver>                        m_driver_api;
    rocprofiler_context_id_t                       m_context;
    std::shared_ptr<rocprofsys::agent>             m_agent;
    typename Driver::counter_config_id_t           m_profile_config;
    std::vector<counter_metadata>                  m_counter_meta;
    std::vector<typename Driver::counter_record_t> m_record_buffer;
    metrics                                        m_result_cache;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
