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
#include "lib/common/utility.hpp"
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
#include <rocprofiler-sdk/cxx/constants.hpp>
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

template <typename Tp>
const Tp**
get_static_ptr_array(const std::vector<Tp>& vec)
{
    // The use of std::map is purposeful. Keys can be vectors in map and cannot be in unordered_map.
    // Simplifying the code to create these static objects. Given that they are not created often (
    // or looked up often), the performance difference between map and unordered_map is negligible.
    using static_ptr_map = std::map<std::vector<Tp>, std::vector<const Tp*>>;
    static auto*& static_ptrs =
        common::static_object<common::Synchronized<static_ptr_map>>::construct();

    return static_ptrs->wlock([&](auto& data) -> const Tp** {
        if(auto it = data.find(vec); it != data.end())
        {
            return it->second.data();
        }

        auto [inserted_it, success]       = data.emplace(vec, std::vector<const Tp*>{});
        const std::vector<Tp>& stored_vec = inserted_it->first;

        auto& ptr_vec = inserted_it->second;
        ptr_vec.reserve(stored_vec.size());
        for(const auto& item : stored_vec)
        {
            ptr_vec.push_back(&item);
        }

        return ptr_vec.data();
    });
}

}  // namespace

// Helper function to extract agent_id from agent-encoded counter_id
// Returns agent_id with handle=0 if not found
rocprofiler_agent_id_t
get_agent_id_from_counter_id(rocprofiler_counter_id_t counter_id)
{
    // Extract logical_node_id from counter ID encoding
    auto agent_index = get_agent_from_counter_id(counter_id) - AGENT_ENCODING_OFFSET;

    // Find the agent with matching logical_node_id
    for(const auto* agent : rocprofiler::agent::get_agents())
    {
        if(agent && agent->logical_node_id == agent_index)
        {
            return agent->id;
        }
    }

    // Return invalid agent_id if not found
    return sdk::null_agent_id;
}

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

    // Extract base metric ID from counter_id (handles agent-encoded counter IDs)
    auto base_metric_id = counters::get_base_metric_from_counter_id(counter_id);

    auto base_info = [&](auto& out_struct) {
        if(const auto* metric_ptr = common::get_val(id_map, static_cast<uint64_t>(base_metric_id)))
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

    auto dim_info = [&](auto& out_struct, rocprofiler_agent_id_t agent_id) {
        auto dim_ptr = counters::get_dimension_cache(agent_id);
        if(!dim_ptr) return false;

        // Use base metric ID for dimension lookup
        const auto* dims =
            common::get_val(dim_ptr->id_to_dim, static_cast<uint64_t>(base_metric_id));
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

        out_struct.dimensions       = counters::get_static_ptr_array(_dim_info);
        out_struct.dimensions_count = _dim_info.size();
        return true;
    };

    // Construct all possible permutations of instance ids. This is every instance
    // that can be returned by the counter across all dimensions.
    // Note: Dimensions are agent-specific. This function uses the agent from counter ID encoding.
    auto dim_permutations = [&](auto& out_struct) {
        // Get agent from counter ID encoding
        auto agent_id = counters::get_agent_id_from_counter_id(counter_id);
        if(agent_id == rocprofiler::sdk::null_agent_id) return false;

        auto dim_ptr = counters::get_dimension_cache(agent_id);
        if(!dim_ptr) return false;
        // Use base metric ID for dimension lookup
        const auto* dims =
            common::get_val(dim_ptr->id_to_dim, static_cast<uint64_t>(base_metric_id));
        if(!dims) return false;

        auto instances = std::vector<rocprofiler_counter_record_dimension_instance_info_t>{};
        auto _dim_info = std::vector<rocprofiler_counter_record_dimension_info_t>{};

        constexpr auto rocprofiler_counter_record_dimension_instance_v1_info_t_rt_size =
            common::compute_runtime_sizeof<rocprofiler_counter_record_dimension_instance_info_t>();

        constexpr auto rocprofiler_counter_dimension_info_v1_t_rt_size =
            common::compute_runtime_sizeof<rocprofiler_counter_dimension_info_t>();

        for(const auto& metric_dim : *dims)
        {
            if(metric_dim.size() == 0) continue;

            _dim_info.emplace_back(rocprofiler_counter_record_dimension_info_t{
                .name          = counters::get_static_string(metric_dim.name()),
                .instance_size = metric_dim.size(),
                .id = static_cast<rocprofiler_counter_dimension_id_t>(metric_dim.type())});

            auto tmp = std::vector<rocprofiler_counter_record_dimension_instance_info_t>{};
            // If no instances are found, create the first set of instances
            if(instances.empty())
            {
                for(size_t i = 0; i < metric_dim.size(); i++)
                {
                    auto& rec = instances.emplace_back();
                    counters::set_dim_in_rec(rec.instance_id, metric_dim.type(), i);
                    // Store full agent-encoded counter ID in instance record
                    counters::set_counter_in_rec(rec.instance_id, counter_id);
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
                        counters::set_dim_in_rec(rec.instance_id, metric_dim.type(), i);
                        // Store full agent-encoded counter ID in instance record
                        counters::set_counter_in_rec(rec.instance_id, counter_id);
                    }
                }
                instances = tmp;
            }
        }
        if(instances.empty())
        {
            out_struct.dimensions_instances       = nullptr;
            out_struct.dimensions_instances_count = 0;
            return true;
        }

        for(auto& instance : instances)
        {
            auto dimensions  = std::vector<rocprofiler_counter_dimension_info_t>{};
            auto instance_id = instance.instance_id;
            for(const auto& dimension : _dim_info)
            {
                auto& curr = dimensions.emplace_back(rocprofiler_counter_dimension_info_t{});
                curr.index = counters::rec_to_dim_pos(
                    instance_id,
                    static_cast<counters::rocprofiler_profile_counter_instance_types>(
                        dimension.id));
                curr.dimension_name = dimension.name;
                curr.size           = rocprofiler_counter_dimension_info_v1_t_rt_size;
            }
            instance.dimensions       = counters::get_static_ptr_array(dimensions);
            instance.dimensions_count = std::size(_dim_info);
            // Extract the counter_id from the instance_id
            instance.counter_id = counters::rec_to_counter_id(instance.instance_id).handle;
            instance.size       = rocprofiler_counter_record_dimension_instance_v1_info_t_rt_size;
        }
        out_struct.dimensions_instances       = counters::get_static_ptr_array(instances);
        out_struct.dimensions_instances_count = instances.size();
        out_struct.size                       = sizeof(rocprofiler_counter_info_v1_t);
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

            // Get agent from counter ID encoding
            auto agent_id = counters::get_agent_id_from_counter_id(counter_id);
            if(agent_id.handle == 0) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

            if(!dim_info(_out_struct, agent_id)) return ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND;
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
rocprofiler_query_counter_instance_count(rocprofiler_agent_id_t   agent_id,
                                         rocprofiler_counter_id_t counter_id,
                                         size_t*                  instance_count)
{
    *instance_count = 0;
    auto dim_ptr    = counters::get_dimension_cache(agent_id);
    if(!dim_ptr) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    // Extract base metric ID from counter_id (handles agent-encoded counter IDs)
    auto        base_metric_id = counters::get_base_metric_from_counter_id(counter_id);
    const auto* dims = common::get_val(dim_ptr->id_to_dim, static_cast<uint64_t>(base_metric_id));
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

    auto metrics = counters::getMetricsForAgent(agent);
    if(metrics.empty()) return ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED;

    std::vector<rocprofiler_counter_id_t> ids;
    ids.reserve(metrics.size());
    for(const auto& metric : metrics)
    {
        // Create agent-encoded counter ID using the agent's logical_node_id
        rocprofiler_counter_id_t counter_id{.handle = 0};
        counters::set_base_metric_in_counter_id(counter_id, metric.id());
        counters::set_agent_in_counter_id(counter_id, agent->logical_node_id);
        ids.push_back(counter_id);
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
    // Get base metric ID from instance record (bits 63-48)
    uint16_t base_metric = static_cast<uint16_t>(id >> counters::DIM_BIT_LENGTH);

    // Try to get agent encoding from ROCPROFILER_DIMENSION_AGENT dimension field
    uint8_t agent_encoded =
        static_cast<uint8_t>(counters::rec_to_dim_pos(id, counters::ROCPROFILER_DIMENSION_AGENT));

    // Reconstruct full agent-encoded counter ID
    // Note: agent_encoded includes the offset, but set_agent_in_counter_id() adds the offset,
    // so we need to subtract it first to get the raw logical_node_id
    counter_id->handle = 0;
    counters::set_base_metric_in_counter_id(*counter_id, base_metric);
    counters::set_agent_in_counter_id(
        *counter_id, agent_encoded > 0 ? agent_encoded - counters::AGENT_ENCODING_OFFSET : 0);

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
    // Extract base metric ID from counter_id (handles agent-encoded counter IDs)
    auto base_metric_id = counters::get_base_metric_from_counter_id(id);

    // Get agent from counter ID encoding
    auto agent_id = counters::get_agent_id_from_counter_id(id);
    if(agent_id.handle == 0) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    auto dim_ptr = counters::get_dimension_cache(agent_id);
    if(!dim_ptr) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;
    // Use base metric ID for dimension lookup
    const auto* dims = common::get_val(dim_ptr->id_to_dim, static_cast<uint64_t>(base_metric_id));
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
    // Regenerate ASTs and Dimension Cache for this agent
    try
    {
        counters::get_ast_map(true);
        counters::get_dimension_cache(agent, true);
    } catch(std::exception& e)
    {
        ROCP_FATAL << "Could not regenerate ASTs and Dimension Cache " << e.what();
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
}
