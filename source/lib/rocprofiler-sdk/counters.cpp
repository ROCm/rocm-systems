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

#include <rocprofiler-sdk/experimental/counters.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>

#include "lib/common/container/small_vector.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/dimensions.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"

namespace
{
const char*
get_static_string(const std::string& str)
{
    return rocprofiler::common::get_string_entry(rocprofiler::common::add_string_entry(str))
        ->c_str();
}
}  // namespace

extern "C" {
/**
 * @brief Query Counter info such as name or description.
 *
 * @param [in] counter_id counter to get info for
 * @param [in] version Version of struct in info, see @ref rocprofiler_counter_info_version_id_t for
 * available types
 * @param [out] info rocprofiler_counter_info_{version}_t struct to write info to.
 * @return ::rocprofiler_status_t
 * @retval ROCPROFILER_STATUS_SUCCESS if counter found
 * @retval ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND if counter not found
 * @retval ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI Version is not supported
 */
rocprofiler_status_t
rocprofiler_query_counter_info(rocprofiler_counter_id_t              counter_id,
                               rocprofiler_counter_info_version_id_t version,
                               void*                                 info)
{
    if(version != ROCPROFILER_COUNTER_INFO_VERSION_0)
        return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
    auto metrics_map = rocprofiler::counters::loadMetrics();

    const auto& id_map = metrics_map->id_to_metric;

    auto& out_struct = *static_cast<rocprofiler_counter_info_v0_t*>(info);

    if(const auto* metric_ptr = rocprofiler::common::get_val(id_map, counter_id.handle))
    {
        out_struct.id          = counter_id;
        out_struct.is_constant = (metric_ptr->constant().empty()) ? 0 : 1;
        out_struct.is_derived  = (metric_ptr->expression().empty()) ? 0 : 1;
        out_struct.name        = get_static_string(metric_ptr->name());
        out_struct.description = get_static_string(metric_ptr->description());
        out_struct.block       = get_static_string(metric_ptr->block());
        out_struct.expression  = get_static_string(metric_ptr->expression());
        return ROCPROFILER_STATUS_SUCCESS;
    }

    ROCP_ERROR << fmt::format("Could not find counter with id = {}", counter_id.handle);
    return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
}

/**
 * @brief This call returns the number of instances specific counter contains.
 *
 * @param [in] agent rocprofiler agent
 * @param [in] counter_id counter id (obtained from iterate_agent_supported_counters)
 * @param [out] instance_count number of instances the counter has
 * @return rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_counter_instance_count(rocprofiler_agent_id_t,
                                         rocprofiler_counter_id_t counter_id,
                                         size_t*                  instance_count)
{
    *instance_count = 0;
    auto dim_ptr    = rocprofiler::counters::get_dimension_cache();

    const auto* dims = rocprofiler::common::get_val(dim_ptr->id_to_dim, counter_id.handle);
    if(!dims) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;

    for(const auto& metric_dim : *dims)
    {
        if(*instance_count == 0)
            *instance_count = metric_dim.size();
        else if(metric_dim.size() > 0)
            *instance_count = metric_dim.size() * *instance_count;
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
    if(metrics.empty()) return ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED;

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
rocprofiler_iterate_counter_dimensions(rocprofiler_counter_id_t              id,
                                       rocprofiler_available_dimensions_cb_t info_cb,
                                       void*                                 user_data)
{
    auto dim_ptr = rocprofiler::counters::get_dimension_cache();

    const auto* dims = rocprofiler::common::get_val(dim_ptr->id_to_dim, id.handle);
    if(!dims) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;

    // This is likely faster than a map lookup given the limited number of dims.
    rocprofiler::common::container::small_vector<rocprofiler_record_dimension_info_t, 6> user_dims;
    for(const auto& internal_dim : *dims)
    {
        auto& dim         = user_dims.emplace_back();
        dim.name          = get_static_string(internal_dim.name());
        dim.instance_size = internal_dim.size();
        dim.id            = static_cast<rocprofiler_counter_dimension_id_t>(internal_dim.type());
    }

    if(user_dims.empty())
    {
        return ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND;
    }

    info_cb(id, user_dims.data(), user_dims.size(), user_data);

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_load_counter_definition(const char* yaml, size_t size, rocprofiler_counter_flag_t flags)
{
    rocprofiler::counters::CustomCounterDefinition def;
    if(yaml == nullptr && size != 0) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    def.data   = std::string(yaml, size);
    def.append = (flags == ROCPROFILER_COUNTER_FLAG_APPEND_DEFINITION ? true : false);
    def.loaded = false;
    return rocprofiler::counters::setCustomCounterDefinition(def);
}

rocprofiler_status_t
rocprofiler_create_counter(const char*               name,
                           size_t                    name_len,
                           const char*               expr,
                           size_t                    expr_len,
                           const char*               description,
                           size_t                    description_len,
                           rocprofiler_agent_id_t    agent,
                           rocprofiler_counter_id_t* counter_id)
{
    const auto* agent_ptr = rocprofiler::agent::get_agent(agent);
    if(!agent_ptr) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    rocprofiler::counters::Metric new_metric(
        "",
        std::string(name, name_len),
        "",
        "",
        std::string((description ? description : ""), description_len),
        std::string(expr, expr_len),
        "",
        -1);

    // Validate the metric. Checks for duplicate names and invalid expressions.
    if(auto status = rocprofiler::counters::check_ast_generation(agent_ptr->name, new_metric);
       status != ROCPROFILER_STATUS_SUCCESS)
    {
        return status;
    }

    auto add_metric =
        rocprofiler::counters::loadMetrics(true, std::make_pair(agent_ptr->name, new_metric));

    if(add_metric->arch_to_metric.at(agent_ptr->name).back().name() != new_metric.name())
    {
        ROCP_ERROR << fmt::format("Custom metric {} was not added", new_metric.name());
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    counter_id->handle = add_metric->arch_to_metric.at(agent_ptr->name).back().id();
    // Regenerate ASTs and Dimension Cache
    try
    {
        rocprofiler::counters::get_ast_map(true);
        rocprofiler::counters::get_dimension_cache(true);
    } catch(std::exception& e)
    {
        ROCP_FATAL << "Could not regenerate ASTs and Dimension Cache " << e.what();
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
}
