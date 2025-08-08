// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/fwd.h>
#include <cstdint>
#include <unordered_set>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/controller.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"

extern "C" {
/**
 * @brief Create Profile Configuration.
 *
 * @param [in] agent Agent identifier
 * @param [in] counters_list List of GPU counters
 * @param [in] counters_count Size of counters list
 * @param [in/out] config_id Identifier for GPU counters group. If an existing
                   profile is supplied, that profiles counters will be copied
                   over to a new profile (returned via this id).
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_create_counter_config(rocprofiler_agent_id_t           agent_id,
                                  rocprofiler_counter_id_t*        counters_list,
                                  size_t                           counters_count,
                                  rocprofiler_counter_config_id_t* config_id)
{
    std::unordered_set<uint64_t> already_added;
    const auto*                  agent = ::rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    std::shared_ptr<rocprofiler::counters::counter_config> config =
        std::make_shared<rocprofiler::counters::counter_config>();

    auto        metrics_map = rocprofiler::counters::loadMetrics();
    const auto& id_map      = metrics_map->id_to_metric;

    for(size_t i = 0; i < counters_count; i++)
    {
        auto& counter_id = counters_list[i];

        const auto* metric_ptr = rocprofiler::common::get_val(id_map, counter_id.handle);
        if(!metric_ptr) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        // Don't add duplicates
        if(!already_added.emplace(metric_ptr->id()).second) continue;

        if(!rocprofiler::counters::checkValidMetric(std::string(agent->name), *metric_ptr))
        {
            return ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT;
        }
        config->metrics.push_back(*metric_ptr);
    }

    if(config_id->handle != 0)
    {
        // Copy existing counters from previous config
        if(auto existing = rocprofiler::counters::get_counter_config(*config_id))
        {
            for(const auto& metric : existing->metrics)
            {
                if(!already_added.emplace(metric.id()).second) continue;
                config->metrics.push_back(metric);
            }
        }
    }

    config->agent = agent;
    if(auto status = rocprofiler::counters::create_counter_profile(config);
       status != ROCPROFILER_STATUS_SUCCESS)
    {
        return status;
    }
    *config_id = config->id;

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_destroy_counter_config(rocprofiler_counter_config_id_t config_id)
{
    rocprofiler::counters::destroy_counter_profile(config_id.handle);
    return ROCPROFILER_STATUS_SUCCESS;
}
}
