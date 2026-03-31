// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "hw_counter_query.hpp"

#include "core/agent_manager.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ranges.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
namespace avail
{
namespace
{
// Local counter info struct — avoids depending on library/rocprofiler-sdk/fwd.hpp
struct counter_info
{
    rocprofiler_agent_id_t                           agent_id       = {};
    rocprofiler_counter_info_v0_t                    info           = {};
    std::vector<rocprofiler_record_dimension_info_t> dimension_info = {};
};

using counter_info_map_t =
    std::unordered_map<rocprofiler_agent_id_t, std::vector<counter_info>>;

rocprofiler_status_t
dimensions_info_callback(rocprofiler_counter_id_t /*id*/,
                         const rocprofiler_record_dimension_info_t* dim_info,
                         long unsigned int num_dims, void* user_data)
{
    auto* dims =
        static_cast<std::vector<rocprofiler_record_dimension_info_t>*>(user_data);
    dims->reserve(num_dims);
    for(size_t j = 0; j < num_dims; j++)
        dims->emplace_back(dim_info[j]);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
counters_supported_callback(rocprofiler_agent_id_t    agent_id,
                            rocprofiler_counter_id_t* counters, size_t num_counters,
                            void* user_data)
{
    auto* data = static_cast<counter_info_map_t*>(user_data);
    auto& vec  = (*data)[agent_id];

    for(size_t i = 0; i < num_counters; ++i)
    {
        auto ci     = counter_info{};
        ci.agent_id = agent_id;
        auto status = rocprofiler_query_counter_info(
            counters[i], ROCPROFILER_COUNTER_INFO_VERSION_0, &ci.info);
        if(status != ROCPROFILER_STATUS_SUCCESS) continue;

        rocprofiler_iterate_counter_dimensions(counters[i], dimensions_info_callback,
                                               &ci.dimension_info);

        if(!ci.info.is_constant) vec.emplace_back(std::move(ci));
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

counter_info_map_t
get_agent_counter_info(const std::vector<std::pair<size_t, const agent*>>& gpu_agents)
{
    auto data = counter_info_map_t{};

    for(const auto& [dev_idx, agnt] : gpu_agents)
    {
        auto aid    = rocprofiler_agent_id_t{ agnt->handle };
        auto status = rocprofiler_iterate_agent_supported_counters(
            aid, counters_supported_callback, &data);

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            fprintf(stderr,
                    "[rocprof-sys-avail] Warning: "
                    "rocprofiler_iterate_agent_supported_counters failed for agent "
                    "0x%lx (status=%d). Agent HW architecture may not be supported.\n",
                    static_cast<unsigned long>(aid.handle), static_cast<int>(status));
            continue;
        }

        auto it = data.find(aid);
        if(it != data.end())
        {
            std::sort(it->second.begin(), it->second.end(),
                      [](const counter_info& lhs, const counter_info& rhs) {
                          return lhs.info.id.handle < rhs.info.id.handle;
                      });
            for(auto& ci : it->second)
            {
                std::sort(
                    ci.dimension_info.begin(), ci.dimension_info.end(),
                    [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });
            }
        }
    }

    return data;
}
}  // namespace

std::vector<tim::hardware_counters::info>
query_gpu_hw_counters()
{
    using hardware_counter_info = tim::hardware_counters::info;
    using qualifier_t           = tim::hardware_counters::qualifier;
    using qualifier_vec_t       = std::vector<qualifier_t>;

    auto result = std::vector<hardware_counter_info>{};

    auto& agent_mngr   = get_agent_manager_instance();
    auto  gpu_agents_v = agent_mngr.get_agents_by_type(agent_type::GPU);
    if(gpu_agents_v.empty()) return result;

    // Build the (device_type_index, agent*) pairs that get_agent_counter_info expects
    auto gpu_agent_pairs = std::vector<std::pair<size_t, const agent*>>{};
    gpu_agent_pairs.reserve(gpu_agents_v.size());
    for(const auto& a : gpu_agents_v)
        gpu_agent_pairs.emplace_back(a->device_type_index, a.get());

    auto agent_counters = get_agent_counter_info(gpu_agent_pairs);

    for(const auto& [dev_idx, agnt] : gpu_agent_pairs)
    {
        auto aid                  = rocprofiler_agent_id_t{ agnt->handle };
        auto device_qualifier_sym = fmt::format(":device={}", dev_idx);
        auto device_qualifier =
            qualifier_t{ true, static_cast<int>(dev_idx), device_qualifier_sym,
                         fmt::format("Device {}", dev_idx) };

        auto it = agent_counters.find(aid);
        if(it == agent_counters.end()) continue;

        auto counters = it->second;
        std::sort(counters.begin(), counters.end(),
                  [](const counter_info& lhs, const counter_info& rhs) {
                      if(lhs.info.is_constant && rhs.info.is_constant)
                          return lhs.info.id < rhs.info.id;
                      else if(lhs.info.is_constant)
                          return true;
                      else if(rhs.info.is_constant)
                          return false;

                      if(!lhs.info.is_derived && !rhs.info.is_derived)
                          return lhs.info.id < rhs.info.id;
                      else if(!lhs.info.is_derived)
                          return true;
                      else if(!rhs.info.is_derived)
                          return false;

                      return lhs.info.id < rhs.info.id;
                  });

        for(const auto& ci : counters)
        {
            auto long_desc = std::string{ ci.info.description };
            auto units     = std::string{};
            auto pysym     = std::string{};

            if(ci.info.is_constant)
            {
                continue;
            }
            else if(ci.info.is_derived)
            {
                auto sym        = fmt::format("{}:device={}", ci.info.name, dev_idx);
                auto short_desc = fmt::format("Derived counter: {}", ci.info.expression);
                result.emplace_back(hardware_counter_info(
                    true, tim::hardware_counters::api::rocm, result.size(), 0, sym, pysym,
                    short_desc, long_desc, units, qualifier_vec_t{ device_qualifier }));
            }
            else
            {
                auto dim_info_strs = std::vector<std::string>{};
                for(const auto& d : ci.dimension_info)
                {
                    if(d.instance_size > 1)
                    {
                        dim_info_strs.emplace_back(
                            fmt::format("{}[0:{}]", d.name, d.instance_size - 1));
                    }
                }

                auto sym        = fmt::format("{}:device={}", ci.info.name, dev_idx);
                auto short_desc = fmt::format("{} on device {}", ci.info.name, dev_idx);
                if(!dim_info_strs.empty())
                {
                    short_desc += fmt::format("{}", fmt::join(dim_info_strs, ". "));
                }
                result.emplace_back(hardware_counter_info(
                    true, tim::hardware_counters::api::rocm, result.size(), 0, sym, pysym,
                    short_desc, long_desc, units, qualifier_vec_t{ device_qualifier }));
            }
        }
    }

    return result;
}
}  // namespace avail
}  // namespace rocprofsys
