// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

extern "C" {
/**
 * @brief Query Counter name.
 *
 * @param [in] counter_id
 * @param [out] name if nullptr, size will be returned
 * @param [out] size
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_counter_name(rocprofiler_counter_id_t counter_id, const char** name, size_t* size)
{
    const auto& id_map = *CHECK_NOTNULL(rocprofiler::counters::getMetricIdMap());

    if(const auto* metric_ptr = rocprofiler::common::get_val(id_map, counter_id.handle))
    {
        *name = metric_ptr->name().c_str();
        *size = metric_ptr->name().size();
        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
}

/**
 * @brief This call returns the number of instances specific counter contains.
 *        WARNING: There is a restriction on this call in the alpha/beta release
 *        of rocprof. This call will not return correct instance information in
 *        tool_init and must be called as part of the dispatch callback for accurate
 *        instance counting information. The reason for this restriction is that HSA
 *        is not yet loaded on tool_init.
 *
 * @param [in] agent rocprofiler agent
 * @param [in] counter_id counter id (obtained from iterate_agent_supported_counters)
 * @param [out] instance_count number of instances the counter has
 * @return rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_counter_instance_count(rocprofiler_agent_id_t   agent_id,
                                         rocprofiler_counter_id_t counter_id,
                                         size_t*                  instance_count)
{
    const rocprofiler_agent_t* agent = rocprofiler::agent::get_agent(agent_id);

    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;
    if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) return ROCPROFILER_STATUS_ERROR;

    const auto& id_map     = *CHECK_NOTNULL(rocprofiler::counters::getMetricIdMap());
    const auto* metric_ptr = rocprofiler::common::get_val(id_map, counter_id.handle);
    if(!metric_ptr) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;

    *instance_count = 0;
    // Special counters do not have hardware metrics and will always have an instance
    // count of 1 (i.e. MAX_WAVE_SIZE)
    if(!metric_ptr->special().empty())
    {
        *instance_count = 1;
        return ROCPROFILER_STATUS_SUCCESS;
    }

    // Returns the set of hardware counters needed to evaluate the metric.
    // For derived metrics, this can be more than one counter. In that case,
    // we return the maximum instance count among all underlying counters.
    auto req_counters = rocprofiler::counters::get_required_hardware_counters(
        rocprofiler::counters::get_ast_map(), std::string(agent->name), *metric_ptr);
    if(!req_counters) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;

    for(const auto& counter : *req_counters)
    {
        if(!counter.special().empty())
        {
            *instance_count = std::max(size_t(1), *instance_count);
            continue;
        }

        try
        {
            auto dims = rocprofiler::counters::getBlockDimensions(agent->name, counter);
            for(const auto& dim : dims)
            {
                *instance_count = std::max(static_cast<size_t>(dim.size()), *instance_count);
            }
        } catch(std::runtime_error& err)
        {
            LOG(ERROR) << fmt::format("Could not lookup instance count for counter {}", counter);
            return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in] agent
 * @param [out] counters_list
 * @param [out] counters_count
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                             rocprofiler_available_counters_cb_t cb,
                                             void*                               user_data)
{
    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    auto metrics = rocprofiler::counters::getMetricsForAgent(agent->name);
    std::vector<rocprofiler_counter_id_t> ids;
    ids.reserve(metrics.size());
    for(const auto& metric : metrics)
    {
        ids.push_back({.handle = metric.id()});
    }

    return cb(agent_id, ids.data(), ids.size(), user_data);
}

/**
 * @brief Query counter id information from record_id
 *
 * @param [in] id record id from rocprofiler_record_counter_t
 * @param [out] counter_id counter id associated with the record
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                    rocprofiler_counter_id_t*         counter_id)
{
    // Get counter id from record
    *counter_id = rocprofiler::counters::rec_to_counter_id(id);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_query_record_dimension_position(rocprofiler_counter_instance_id_t  id,
                                            rocprofiler_counter_dimension_id_t dim,
                                            size_t*                            pos)
{
    *pos = rocprofiler::counters::rec_to_dim_pos(
        id, static_cast<rocprofiler::counters::rocprofiler_profile_counter_instance_types>(dim));
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_query_record_dimension_info(rocprofiler_counter_id_t,
                                        rocprofiler_counter_dimension_id_t   dim,
                                        rocprofiler_record_dimension_info_t* info)
{
    if(const auto* ptr = rocprofiler::common::get_val(
           rocprofiler::counters::dimension_map(),
           static_cast<rocprofiler::counters::rocprofiler_profile_counter_instance_types>(dim)))
    {
        info->name = ptr->c_str();
        // TODO: Needs info on the instance size per block to fill in.
        //       counter_id will be used to lookup this information.
        info->instance_size = 0;
        return ROCPROFILER_STATUS_SUCCESS;
    }
    return ROCPROFILER_STATUS_ERROR;
}
}
