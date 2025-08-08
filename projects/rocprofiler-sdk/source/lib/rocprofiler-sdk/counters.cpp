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

#include "lib/common/container/small_vector.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/dimensions.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/experimental/counters.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/core.h>

namespace rocprofiler
{
namespace counters
{
namespace
{
const char*
get_static_string(std::string_view str)
{
    return common::get_string_entry(str)->c_str();
}

template <class T>
const std::vector<T>*
get_static_ptr(const std::vector<T>& vec)
{
    // The use of std::map is purposeful. Keys can be vectors in map and cannot be in unordered_map.
    // Simplifying the code to create these static objects. Given that they are not created often (
    // or looked up often), the performance difference between map and unordered_map is negligible.
    using static_ptr_map = std::map<std::vector<T>, std::unique_ptr<std::vector<T>>>;
    static auto*& static_ptrs =
        common::static_object<common::Synchronized<static_ptr_map>>::construct();
    return static_ptrs->wlock([&](auto& data) {
        if(auto it = data.find(vec); it != data.end())
        {
            return it->second.get();
        }
        data[vec] = std::make_unique<std::vector<T>>(vec);
        return data[vec].get();
    });
}
}  // namespace
}  // namespace counters
}  // namespace rocprofiler

namespace counters = ::rocprofiler::counters;
namespace common   = ::rocprofiler::common;

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
    auto        metrics_map = counters::loadMetrics();
    const auto& id_map      = metrics_map->id_to_metric;

    auto base_info = [&](auto& out_struct) {
        if(const auto* metric_ptr = common::get_val(id_map, counter_id.handle))
        {
            out_struct.id          = counter_id;
            out_struct.is_constant = (metric_ptr->constant().empty()) ? 0 : 1;
            out_struct.is_derived  = (metric_ptr->expression().empty()) ? 0 : 1;
            out_struct.name        = counters::get_static_string(metric_ptr->name());
            out_struct.description = counters::get_static_string(metric_ptr->description());
            out_struct.block       = counters::get_static_string(metric_ptr->block());
            out_struct.expression  = counters::get_static_string(metric_ptr->expression());
            return true;
        }
        return false;
    };

    auto dim_info = [&](auto& out_struct) {
        auto dim_ptr = counters::get_dimension_cache();

        const auto* dims = common::get_val(dim_ptr->id_to_dim, counter_id.handle);
        if(!dims) return false;

        auto _dim_info = std::vector<rocprofiler_counter_record_dimension_info_t>{};
        for(const auto& metric_dim : *dims)
        {
            _dim_info.emplace_back(rocprofiler_counter_record_dimension_info_t{
                .name          = counters::get_static_string(metric_dim.name()),
                .instance_size = metric_dim.size(),
                .id = static_cast<rocprofiler_counter_dimension_id_t>(metric_dim.type())});
        }

        if(_dim_info.empty())
        {
            // Can be 0 if the counter is not known by AQLProfile. This is the case
            // if it was added in a later version of AQLProfile.
            out_struct.dimensions       = nullptr;
            out_struct.dimensions_count = 0;
            return true;
        }

        out_struct.dimensions       = counters::get_static_ptr(_dim_info)->data();
        out_struct.dimensions_count = _dim_info.size();
        return true;
    };

    // Construct all possible permutations of instance ids. This is every instance
    // that can be returned by the counter across all dimensions.
    auto dim_permutations = [&](auto& out_struct) {
        auto dim_ptr = counters::get_dimension_cache();

        const auto* dims = common::get_val(dim_ptr->id_to_dim, counter_id.handle);
        if(!dims) return false;

        std::vector<rocprofiler_counter_instance_id_t> instances;

        for(const auto& metric_dim : *dims)
        {
            if(metric_dim.size() == 0) continue;
            std::vector<rocprofiler_counter_instance_id_t> tmp;
            // If no instances are found, create the first set of instances
            if(instances.empty())
            {
                for(size_t i = 0; i < metric_dim.size(); i++)
                {
                    auto& rec = instances.emplace_back();
                    counters::set_dim_in_rec(rec, metric_dim.type(), i);
                    counters::set_counter_in_rec(rec, counter_id);
                }
            }
            else
            {
                // For each instance, create a new set of instances with the new dimension added.
                // This will create all possible permutations of the dimensions.
                for(size_t i = 0; i < metric_dim.size(); i++)
                {
                    for(const auto& instance : instances)
                    {
                        auto& rec = tmp.emplace_back(instance);
                        counters::set_dim_in_rec(rec, metric_dim.type(), i);
                        counters::set_counter_in_rec(rec, counter_id);
                    }
                }
                instances = tmp;
            }
        }
        if(instances.empty())
        {
            out_struct.instance_ids       = nullptr;
            out_struct.instance_ids_count = 0;
            return true;
        }

        out_struct.instance_ids       = counters::get_static_ptr(instances)->data();
        out_struct.instance_ids_count = instances.size();
        return true;
    };

    switch(version)
    {
        case ROCPROFILER_COUNTER_INFO_VERSION_0:
        {
            auto& _out_struct = *static_cast<rocprofiler_counter_info_v0_t*>(info);

            if(base_info(_out_struct)) return ROCPROFILER_STATUS_SUCCESS;
            return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        }
        break;
        case ROCPROFILER_COUNTER_INFO_VERSION_1:
        {
            auto& _out_struct = *static_cast<rocprofiler_counter_info_v1_t*>(info);

            if(!base_info(_out_struct)) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
            if(!dim_info(_out_struct)) return ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND;
            if(!dim_permutations(_out_struct)) return ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND;

            return ROCPROFILER_STATUS_SUCCESS;
        }
        break;
        default:
        {
            return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;
        }
    }

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
    auto dim_ptr    = counters::get_dimension_cache();

    const auto* dims = common::get_val(dim_ptr->id_to_dim, counter_id.handle);
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

    auto metrics = counters::getMetricsForAgent(agent->name);
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
 * @param [in] id record id from rocprofiler_counter_record_t
 * @param [out] counter_id counter id associated with the record
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                    rocprofiler_counter_id_t*         counter_id)
{
    // Get counter id from record
    *counter_id = counters::rec_to_counter_id(id);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_query_record_dimension_position(rocprofiler_counter_instance_id_t  id,
                                            rocprofiler_counter_dimension_id_t dim,
                                            size_t*                            pos)
{
    *pos = counters::rec_to_dim_pos(
        id, static_cast<counters::rocprofiler_profile_counter_instance_types>(dim));
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_iterate_counter_dimensions(rocprofiler_counter_id_t              id,
                                       rocprofiler_available_dimensions_cb_t info_cb,
                                       void*                                 user_data)
{
    auto dim_ptr = counters::get_dimension_cache();

    const auto* dims = common::get_val(dim_ptr->id_to_dim, id.handle);
    if(!dims) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;

    // This is likely faster than a map lookup given the limited number of dims.
    auto user_dims =
        common::container::small_vector<rocprofiler_counter_record_dimension_info_t, 6>{};
    for(const auto& internal_dim : *dims)
    {
        auto& dim         = user_dims.emplace_back();
        dim.name          = counters::get_static_string(internal_dim.name());
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
    counters::CustomCounterDefinition def;
    if(yaml == nullptr && size != 0) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    def.data   = std::string(yaml, size);
    def.append = (flags == ROCPROFILER_COUNTER_FLAG_APPEND_DEFINITION ? true : false);
    def.loaded = false;
    return counters::setCustomCounterDefinition(def);
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

    counters::Metric new_metric("",
                                std::string(name, name_len),
                                "",
                                "",
                                std::string((description ? description : ""), description_len),
                                std::string(expr, expr_len),
                                "",
                                -1);

    // Validate the metric. Checks for duplicate names and invalid expressions.
    if(auto status = counters::check_ast_generation(agent_ptr->name, new_metric);
       status != ROCPROFILER_STATUS_SUCCESS)
    {
        return status;
    }

    auto add_metric = counters::loadMetrics(true, std::make_pair(agent_ptr->name, new_metric));

    if(add_metric->arch_to_metric.at(agent_ptr->name).back().name() != new_metric.name())
    {
        ROCP_ERROR << fmt::format("Custom metric {} was not added", new_metric.name());
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    counter_id->handle = add_metric->arch_to_metric.at(agent_ptr->name).back().id();
    // Regenerate ASTs and Dimension Cache
    try
    {
        counters::get_ast_map(true);
        counters::get_dimension_cache(true);
    } catch(std::exception& e)
    {
        ROCP_FATAL << "Could not regenerate ASTs and Dimension Cache " << e.what();
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
}
