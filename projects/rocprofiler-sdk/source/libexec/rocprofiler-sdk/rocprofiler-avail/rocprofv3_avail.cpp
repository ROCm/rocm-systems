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

#include "rocprofv3_avail.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/agent_info.hpp"
#include "lib/output/counter_info.hpp"
#include "lib/output/metadata.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <fmt/core.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace common = ::rocprofiler::common;
namespace tool   = ::rocprofiler::tool;

using JSONOutputArchive = cereal::MinimalJSONOutputArchive;

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
            ROCP_FATAL << errmsg.str()                                                             \
        }                                                                                          \
    }

void
initialize_logging()
{
    auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
    common::init_logging("ROCPROF", logging_cfg);
    FLAGS_colorlogtostderr = true;
}

tool::metadata&
get_metadata()
{
    static auto tool_metadata = std::make_unique<tool::metadata>(tool::metadata::inprocess{});
    auto        _once         = std::once_flag{};
    std::call_once(_once, [&]() {
        initialize_logging();
        tool_metadata->init(tool::metadata::inprocess{});
    });
    return *tool_metadata;
}

tool::agent_info_map_t
get_agents()
{
    return get_metadata().agents_map;
}

tool::agent_counter_info_map_t
get_agent_counters()
{
    return get_metadata().agent_counter_info;
}

const tool::tool_counter_info*
get_counter_info(rocprofiler_counter_id_t counter_id)
{
    const auto* counter_info = get_metadata().get_counter_info(counter_id);
    if(!counter_info) ROCP_FATAL << "Invalid counter handle";
    return counter_info;
}

auto agent_json = std::map<rocprofiler_agent_id_t, std::string>{};

ROCPROFILER_EXTERN_C_INIT

size_t
get_number_of_agents()
{
    return get_agents().size();
}

void
agent_handles(uint64_t* agent_handles, size_t num_agents)
{
    auto agent_info = get_agents();
    if(num_agents != agent_info.size()) ROCP_FATAL << "Incorrect number of agents";
    auto agent_ids = std::vector<uint64_t>{};
    std::for_each(agent_info.begin(), agent_info.end(), [&agent_ids](auto& agent) {
        agent_ids.emplace_back(agent.first.handle);
    });
    std::copy(agent_ids.begin(), agent_ids.end(), agent_handles);
}

size_t
get_number_of_agent_counters(uint64_t agent_handle)
{
    auto agent_id           = rocprofiler_agent_id_t{agent_handle};
    auto agent_counter_info = get_agent_counters();
    if(agent_counter_info.find(agent_id) != agent_counter_info.end())
        return agent_counter_info.at(agent_id).size();
    return 0;
}

void
agent_counter_handles(uint64_t* counter_handles,
                      uint64_t  agent_handle,
                      size_t    number_of_agent_counters)
{
    auto agent_counter_info = get_agent_counters();
    auto itr                = agent_counter_info.find(rocprofiler_agent_id_t{agent_handle});

    if(itr == agent_counter_info.end()) return;

    if(number_of_agent_counters != itr->second.size()) ROCP_FATAL << "Incorrect number of agents";
    auto counter_ids = std::vector<uint64_t>{};

    std::for_each(itr->second.begin(), itr->second.end(), [&counter_ids](auto& counter) {
        counter_ids.emplace_back(counter.id.handle);
    });
    std::copy(counter_ids.begin(), counter_ids.end(), counter_handles);
}

void
counter_info(uint64_t     counter_handle,
             const char** counter_name,
             const char** counter_description,
             uint8_t*     is_derived,
             uint8_t*     is_hw_constant)
{
    const auto* counter_info = get_counter_info(rocprofiler_counter_id_t{counter_handle});

    *counter_name        = counter_info->name;
    *counter_description = counter_info->description;
    *is_derived          = counter_info->is_derived;
    *is_hw_constant      = counter_info->is_constant;
}

void
counter_block(uint64_t counter_handle, const char** counter_block)
{
    const auto* counter_info = get_counter_info(rocprofiler_counter_id_t{counter_handle});
    *counter_block           = counter_info->block;
}

void
counter_expression(uint64_t counter_handle, const char** counter_expr)
{
    const auto* counter_info = get_counter_info(rocprofiler_counter_id_t{counter_handle});
    *counter_expr            = counter_info->expression;
}

size_t
get_number_of_dimensions(uint64_t counter_handle)
{
    const auto* counter_info = get_counter_info(rocprofiler_counter_id_t{counter_handle});
    return counter_info->dimensions_count;
}

void
counter_dimension_ids(uint64_t counter_handle, uint64_t* dimension_ids, size_t num_dimensions)
{
    const auto* counter_info = get_counter_info(rocprofiler_counter_id_t{counter_handle});
    if(num_dimensions != counter_info->dimension_ids.size()) ROCP_FATAL << "Invalid counter handle";
    auto dimensions = std::vector<uint64_t>{};
    std::for_each(counter_info->dimension_ids.begin(),
                  counter_info->dimension_ids.end(),
                  [&dimensions](auto& dimension) { dimensions.emplace_back(dimension); });
    std::copy(dimensions.begin(), dimensions.end(), dimension_ids);
}

void
counter_dimension(uint64_t     counter_handle,
                  uint64_t     dimension_handle,
                  const char** dimension_name,
                  uint64_t*    dimension_instance)
{
    const auto* counter_info = get_counter_info(rocprofiler_counter_id_t{counter_handle});

    rocprofiler::tool::counter_dimension_info_vec_t dimensions = counter_info->dimensions;

    for(auto dim : dimensions)
    {
        if(dim.id == dimension_handle)
        {
            *dimension_name     = dim.name;
            *dimension_instance = dim.instance_size;
            return;
        }
    }
}

size_t
get_number_of_pc_sample_configs(uint64_t agent_handle)
{
    auto pc_sampling_config =
        get_metadata().get_pc_sample_config_info(rocprofiler_agent_id_t{agent_handle});

    return pc_sampling_config.size();
}

void
pc_sample_config(uint64_t  agent_handle,
                 uint64_t  config_idx,
                 uint64_t* method,
                 uint64_t* unit,
                 uint64_t* min_interval,
                 uint64_t* max_interval,
                 uint64_t* flags)
{
    auto pc_sampling_config =
        get_metadata().get_pc_sample_config_info(rocprofiler_agent_id_t{agent_handle});
    if(config_idx >= pc_sampling_config.size()) ROCP_FATAL << "Invalid config idx";
    *method       = pc_sampling_config.at(config_idx).method;
    *unit         = pc_sampling_config.at(config_idx).unit;
    *min_interval = pc_sampling_config.at(config_idx).min_interval;
    *max_interval = pc_sampling_config.at(config_idx).max_interval;
    *flags        = pc_sampling_config.at(config_idx).flags;
}

bool
is_counter_set(uint64_t* counter_handles, uint64_t agent_handle, size_t num_counters)
{
    rocprofiler_profile_config_id_t cfg_id = {.handle = 0};
    for(size_t itr = 0; itr < num_counters; itr++)
    {
        auto counter_id = rocprofiler_counter_id_t{counter_handles[itr]};
        if(rocprofiler_create_profile_config(
               rocprofiler_agent_id_t{agent_handle}, &counter_id, 1, &cfg_id) !=
           ROCPROFILER_STATUS_SUCCESS)
            return false;
    }
    rocprofiler_destroy_profile_config(cfg_id);
    return true;
}

void
agent_info(uint64_t agent_handle, const char** agent_info_str)
{
    auto agents = get_agents();

    if(agent_json.empty())
    {
        for(auto& [agent_id, value] : agents)
        {
            auto ss = std::stringstream{};
            {
                constexpr auto json_prec   = 16;
                constexpr auto json_indent = JSONOutputArchive::Options::IndentChar::space;
                auto           json_opts   = JSONOutputArchive::Options{json_prec, json_indent, 0};
                auto           json_ar     = JSONOutputArchive{ss, json_opts};
                cereal::save(json_ar, value);
            }
            agent_json.emplace(agent_id, ss.str());
        }
    }
    *agent_info_str = agent_json.at(rocprofiler_agent_id_t{agent_handle}).c_str();
}

ROCPROFILER_EXTERN_C_FINI
