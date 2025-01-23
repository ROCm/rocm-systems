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

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

auto destructors = new std::vector<std::function<void()>>{};

namespace common = ::rocprofiler::common;

namespace
{
auto pc_sampling_method = std::deque<std::string>{
    "ROCPROFILER_PC_SAMPLING_METHOD_NONE",
    "ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC",
    "ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP",
    "ROCPROFILER_PC_SAMPLING_METHOD_LAST",
};

auto pc_sampling_unit = std::deque<std::string>{
    "ROCPROFILER_PC_SAMPLING_UNIT_NONE",
    "ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS",
    "ROCPROFILER_PC_SAMPLING_UNIT_CYCLES",
    "ROCPROFILER_PC_SAMPLING_UNIT_TIME",
    "ROCPROFILER_PC_SAMPLING_UNIT_LAST",
};
}  // namespace

using counter_info_t      = std::vector<std::vector<std::string>>;
using pc_sample_info_t    = std::vector<std::vector<std::string>>;
auto agent_counter_info   = std::unordered_map<uint64_t, counter_info_t>{};
auto agent_pc_sample_info = std::unordered_map<uint64_t, pc_sample_info_t>{};
// auto agent_configs_info = std::unordered_map<uint64_t, config_info_t>{};
using counter_dimension_info_t =
    std::unordered_map<uint64_t, std::vector<std::vector<std::string>>>;
auto                  counter_dim_info = counter_dimension_info_t{};
std::vector<uint64_t> agent_node_ids;

constexpr size_t pc_config_fields = 4, method_idx = 0, unit_idx = 1, min_interval_idx = 2,
                 max_interval_idx  = 3;
constexpr size_t dimensions_fields = 3, dim_id_idx = 0, dim_name_idx = 1, size_idx = 2;
constexpr size_t counter_fields = 5, counter_id_idx = 0, name_idx = 1, description_idx = 2,
                 is_derived_idx = 3, block_idx = 4, expression_idx = 4;

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) = result;                 \
        if(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) != ROCPROFILER_STATUS_SUCCESS)              \
        {                                                                                          \
            std::string status_msg =                                                               \
                rocprofiler_get_status_string(ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__));        \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << ROCPROFILER_VARIABLE(CHECKSTATUS, __LINE__) \
                      << ": " << status_msg << "\n"                                                \
                      << std::flush;                                                               \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

using counter_vec_t = std::vector<rocprofiler_counter_id_t>;

ROCPROFILER_EXTERN_C_INIT
void
avail_tool_init() ROCPROFILER_EXPORT;
size_t
get_number_of_agents() ROCPROFILER_EXPORT;
uint64_t
get_agent_node_id(int idx) ROCPROFILER_EXPORT;
int
get_number_of_counters(uint64_t node_id) ROCPROFILER_EXPORT;
void
get_counters_info(uint64_t     node_id,
                  int          idx,
                  uint64_t*    counter_id,
                  const char** counter_name,
                  const char** counter_description,
                  uint8_t*     is_derived) ROCPROFILER_EXPORT;
void
get_counter_expression(uint64_t node_id, int idx, const char** counter_expr) ROCPROFILER_EXPORT;

void
get_counter_block(uint64_t node_id, int idx, const char** counter_block) ROCPROFILER_EXPORT;

int
get_number_of_dimensions(int counter_id) ROCPROFILER_EXPORT;

void
get_counter_dimension(uint64_t     counter_id,
                      uint64_t     dimension_idx,
                      uint64_t*    dimension_id,
                      const char** dimension_name,
                      uint64_t*    dimension_instance) ROCPROFILER_EXPORT;

int
get_number_of_pc_sample_configs(uint64_t node_id) ROCPROFILER_EXPORT;

void
get_pc_sample_config(uint64_t     node_id,
                     int          idx,
                     const char** method,
                     const char** unit,
                     uint64_t*    min_interval,
                     uint64_t*    max_interval) ROCPROFILER_EXPORT;
ROCPROFILER_EXTERN_C_FINI

void
initialize_logging()
{
    auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
    common::init_logging("ROCPROF", logging_cfg);
    FLAGS_colorlogtostderr = true;
}

rocprofiler_status_t
pc_configuration_callback(const rocprofiler_pc_sampling_configuration_t* configs,
                          long unsigned int                              num_config,
                          void*                                          user_data)
{
    auto* avail_configs = static_cast<std::vector<std::vector<std::string>>*>(user_data);

    for(size_t i = 0; i < num_config; i++)
    {
        auto config = std::vector<std::string>{};
        config.reserve(pc_config_fields);
        auto it = config.begin();
        config.insert(it + method_idx, pc_sampling_method.at(configs[i].method));
        config.insert(it + unit_idx, pc_sampling_unit.at(configs[i].unit));
        config.insert(it + min_interval_idx, std::to_string(configs[i].min_interval));
        config.insert(it + max_interval_idx, std::to_string(configs[i].max_interval));
        avail_configs->push_back(config);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
dimensions_info_callback(rocprofiler_counter_id_t /*id*/,
                         const rocprofiler_record_dimension_info_t* dim_info,
                         long unsigned int                          num_dims,
                         void*                                      user_data)
{
    auto* dimensions_info =
        static_cast<std::vector<rocprofiler_record_dimension_info_t>*>(user_data);
    dimensions_info->reserve(num_dims);
    for(size_t j = 0; j < num_dims; j++)
        dimensions_info->emplace_back(dim_info[j]);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
iterate_agent_counters_callback(rocprofiler_agent_id_t,
                                rocprofiler_counter_id_t* counters,
                                size_t                    num_counters,
                                void*                     user_data)
{
    auto* _counters_info = static_cast<std::vector<std::vector<std::string>>*>(user_data);
    for(size_t i = 0; i < num_counters; i++)
    {
        rocprofiler_counter_info_v0_t _info;
        auto dimensions_data = std::vector<rocprofiler_record_dimension_info_t>{};
        ROCPROFILER_CALL(
            rocprofiler_iterate_counter_dimensions(
                counters[i], dimensions_info_callback, static_cast<void*>(&dimensions_data)),
            "iterate_dimension_info");
        auto dimensions_info = std::vector<std::vector<std::string>>{};
        dimensions_info.reserve(dimensions_data.size());
        for(auto& dim : dimensions_data)
        {
            auto dimensions = std::vector<std::string>{};
            dimensions.reserve(dimensions_fields);
            auto it = dimensions.begin();
            dimensions.insert(it + dim_id_idx, std::to_string(dim.id));
            dimensions.insert(it + dim_name_idx, std::string(dim.name));
            dimensions.insert(it + size_idx, std::to_string(dim.instance_size - 1));
            dimensions_info.emplace_back(dimensions);
        }
        counter_dim_info.emplace(counters[i].handle, dimensions_info);
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counters[i], ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&_info)),
            "Could not query counter_id");

        auto counter = std::vector<std::string>{};

        if(_info.is_derived)
        {
            counter.reserve(counter_fields);
            auto it = counter.begin();
            counter.insert(it + counter_id_idx, std::to_string(_info.id.handle));
            counter.insert(it + name_idx, std::string(_info.name));
            counter.insert(it + description_idx, std::string(_info.description));
            counter.insert(it + is_derived_idx, std::to_string(_info.is_derived));
            counter.insert(it + expression_idx, std::string(_info.expression));
        }
        else
        {
            counter.reserve(counter_fields);
            auto it = counter.begin();
            counter.insert(it + counter_id_idx, std::to_string(_info.id.handle));
            counter.insert(it + name_idx, std::string(_info.name));
            counter.insert(it + description_idx, std::string(_info.description));
            counter.insert(it + is_derived_idx, std::to_string(_info.is_derived));
            counter.insert(it + block_idx, std::string(_info.block));
        }

        _counters_info->emplace_back(counter);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
list_avail_configs(rocprofiler_agent_version_t, const void** agents, size_t num_agents, void*)
{
    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
        if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            auto counters_v = counter_vec_t{};

            // TODO(aelwazir): To be changed back to use node id once ROCR fixes
            // the hsa_agents to use the real node id
            uint32_t                              node_id           = agent->node_id;
            std::vector<std::vector<std::string>> configs           = {};
            std::vector<std::vector<std::string>> _counter_dim_info = {};
            agent_node_ids.emplace_back(node_id);
            rocprofiler_query_pc_sampling_agent_configurations(
                agent->id, pc_configuration_callback, &configs);
            ROCPROFILER_CALL(
                rocprofiler_iterate_agent_supported_counters(
                    agent->id, iterate_agent_counters_callback, (void*) (&_counter_dim_info)),
                "Iterate rocprofiler counters");
            if(!_counter_dim_info.empty()) agent_counter_info.emplace(node_id, _counter_dim_info);
            if(!configs.empty())

            {
                agent_pc_sample_info.emplace(node_id, configs);
            }
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_EXTERN_C_INIT

void
avail_tool_init()
{
    initialize_logging();
    ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                        list_avail_configs,
                                                        sizeof(rocprofiler_agent_t),
                                                        nullptr),
                     "Iterate rocporfiler agents");
}

size_t
get_number_of_agents()
{
    return agent_node_ids.size();
}

uint64_t
get_agent_node_id(int idx)
{
    return agent_node_ids.at(idx);
}

int
get_number_of_counters(uint64_t node_id)
{
    if(agent_counter_info.find(node_id) != agent_counter_info.end())
        return agent_counter_info.at(node_id).size();
    else
        return 0;
}

void
get_counters_info(uint64_t     node_id,
                  int          counter_idx,
                  uint64_t*    counter_id,
                  const char** counter_name,
                  const char** counter_description,
                  uint8_t*     is_derived)
{
    if(agent_counter_info.find(node_id) == agent_counter_info.end()) return;
    *counter_id =
        std::stoull(agent_counter_info.at(node_id).at(counter_idx).at(0).c_str(), nullptr, 10);
    *counter_name        = agent_counter_info.at(node_id).at(counter_idx).at(1).c_str();
    *counter_description = agent_counter_info.at(node_id).at(counter_idx).at(2).c_str();
    *is_derived          = std::stoi(agent_counter_info.at(node_id).at(counter_idx).at(3).c_str());
}

void
get_counter_block(uint64_t node_id, int counter_idx, const char** counter_block)
{
    if(agent_counter_info.find(node_id) == agent_counter_info.end()) return;
    *counter_block = agent_counter_info.at(node_id).at(counter_idx).at(4).c_str();
}

void
get_counter_expression(uint64_t node_id, int idx, const char** counter_expr)
{
    if(agent_counter_info.find(node_id) == agent_counter_info.end()) return;
    *counter_expr = agent_counter_info.at(node_id).at(idx).at(4).c_str();
}

int
get_number_of_dimensions(int counter_id)
{
    if(counter_dim_info.find(counter_id) == counter_dim_info.end()) return 0;
    return counter_dim_info.at(counter_id).size();
}
void
get_counter_dimension(uint64_t     counter_id,
                      uint64_t     dimension_idx,
                      uint64_t*    dimension_id,
                      const char** dimension_name,
                      uint64_t*    dimension_instance)
{
    if(counter_dim_info.find(counter_id) == counter_dim_info.end()) return;
    *dimension_id =
        std::stoull(counter_dim_info.at(counter_id).at(dimension_idx).at(0).c_str(), nullptr, 10);
    *dimension_name = counter_dim_info.at(counter_id).at(dimension_idx).at(1).c_str();
    *dimension_instance =
        std::stoull(counter_dim_info.at(counter_id).at(dimension_idx).at(2).c_str(), nullptr, 10);
}

int
get_number_of_pc_sample_configs(uint64_t node_id)
{
    if(agent_pc_sample_info.find(node_id) == agent_pc_sample_info.end()) return 0;
    return agent_pc_sample_info.at(node_id).size();
}

void
get_pc_sample_config(uint64_t     node_id,
                     int          config_idx,
                     const char** method,
                     const char** unit,
                     uint64_t*    min_interval,
                     uint64_t*    max_interval)
{
    if(agent_pc_sample_info.find(node_id) == agent_pc_sample_info.end()) return;
    *method = agent_pc_sample_info.at(node_id).at(config_idx).at(0).c_str();
    *unit   = agent_pc_sample_info.at(node_id).at(config_idx).at(1).c_str();
    *min_interval =
        std::stoull(agent_pc_sample_info.at(node_id).at(config_idx).at(2).c_str(), nullptr, 10);
    *max_interval =
        std::stoull(agent_pc_sample_info.at(node_id).at(config_idx).at(3).c_str(), nullptr, 10);
}

ROCPROFILER_EXTERN_C_FINI
