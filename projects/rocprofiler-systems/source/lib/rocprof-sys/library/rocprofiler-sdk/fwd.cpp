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

#include "library/rocprofiler-sdk/fwd.hpp"
#include "core/debug.hpp"
#include "core/state.hpp"

#include <timemory/utility/join.hpp>

#include <exception>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <utility>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace
{
using tool_agent_vec_t = std::vector<tool_agent>;

rocprofiler_status_t
dimensions_info_callback(rocprofiler_counter_id_t /*id*/,
                         const rocprofiler_record_dimension_info_t* dim_info,
                         long unsigned int num_dims, void* user_data)
{
    auto* dimensions_info =
        static_cast<std::vector<rocprofiler_record_dimension_info_t>*>(user_data);
    dimensions_info->reserve(num_dims);
    for(size_t j = 0; j < num_dims; j++)
        dimensions_info->emplace_back(dim_info[j]);

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
counters_supported_callback(rocprofiler_agent_id_t    agent_id,
                            rocprofiler_counter_id_t* counters, size_t num_counters,
                            void* user_data)
{
    using value_type = typename agent_counter_info_map_t::mapped_type;

    auto* data_v = static_cast<agent_counter_info_map_t*>(user_data);
    data_v->emplace(agent_id, value_type{});
    for(size_t i = 0; i < num_counters; ++i)
    {
        auto _info     = rocprofiler_counter_info_v0_t{};
        auto _dim_info = std::vector<rocprofiler_record_dimension_info_t>{};

        ROCPROFILER_CALL(rocprofiler_query_counter_info(
            counters[i], ROCPROFILER_COUNTER_INFO_VERSION_0, &_info));

        // populate local vector
        ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions(
            counters[i], dimensions_info_callback, &_dim_info));

        if(!_info.is_constant)
            data_v->at(agent_id).emplace_back(agent_id, _info, std::move(_dim_info));
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

agent_counter_info_map_t
get_agent_counter_info(const tool_agent_vec_t& _agents)
{
    auto _data = agent_counter_info_map_t{};

    for(auto itr : _agents)
    {
        const auto& _agent_id = rocprofiler_agent_id_t{ itr.agent->handle };

        ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
            _agent_id, counters_supported_callback, &_data));

        std::sort(_data.at(_agent_id).begin(), _data.at(_agent_id).end(),
                  [](const auto& lhs, const auto& rhs) {
                      return (lhs.id.handle < rhs.id.handle);
                  });

        for(auto& citr : _data.at(_agent_id))
        {
            std::sort(citr.dimension_info.begin(), citr.dimension_info.end(),
                      [](const auto& lhs, const auto& rhs) { return (lhs.id < rhs.id); });
        }
    }

    return _data;
}
}  // namespace

rocprofiler_tool_counter_info_t::rocprofiler_tool_counter_info_t(
    rocprofiler_agent_id_t _agent_id, parent_type _info, dimension_info_vec_t&& _dim_info)
: parent_type{ _info }
, agent_id{ _agent_id }
, dimension_info{ std::move(_dim_info) }
{}

void
client_data::initialize()
{
    buffered_tracing_info = rocprofiler::sdk::get_buffer_tracing_names();
    callback_tracing_info = rocprofiler::sdk::get_callback_tracing_names();

    set_agents();
}

void
client_data::initialize_event_info()
{
    if(agent_manager::get_instance().get_agents().empty())
    {
        initialize();
    }

    if(agent_counter_info.size() != gpu_agents.size())
        agent_counter_info = get_agent_counter_info(gpu_agents);

    try
    {
        using qualifier_t     = tim::hardware_counters::qualifier;
        using qualifier_vec_t = std::vector<qualifier_t>;

        for(const auto& aitr : gpu_agents)
        {
            auto        _dev_index = aitr.device_id;
            const auto& _agent_id  = rocprofiler_agent_id_t{ aitr.agent->handle };
            auto        _device_qualifier_sym = JOIN("", ":device=", _dev_index);
            auto        _device_qualifier =
                tim::hardware_counters::qualifier{ true, static_cast<int>(_dev_index),
                                                   _device_qualifier_sym,
                                                   JOIN(" ", "Device", _dev_index) };

            auto _counter_info = agent_counter_info.at(_agent_id);
            std::sort(_counter_info.begin(), _counter_info.end(),
                      [](const rocprofiler_tool_counter_info_t& lhs,
                         const rocprofiler_tool_counter_info_t& rhs) {
                          if(lhs.is_constant && rhs.is_constant)
                              return lhs.id < rhs.id;
                          else if(lhs.is_constant)
                              return true;
                          else if(rhs.is_constant)
                              return false;

                          if(!lhs.is_derived && !rhs.is_derived)
                              return lhs.id < rhs.id;
                          else if(!lhs.is_derived)
                              return true;
                          else if(!rhs.is_derived)
                              return false;

                          return lhs.id < rhs.id;
                      });

            for(const auto& ditr : _counter_info)
            {
                auto _long_desc = std::string{ ditr.description };
                auto _units     = std::string{};
                auto _pysym     = std::string{};
                if(ditr.is_constant)
                {
                    continue;
                }
                else if(ditr.is_derived)
                {
                    auto _sym        = JOIN("", ditr.name, _device_qualifier_sym);
                    auto _short_desc = JOIN("", "Derived counter: ", ditr.expression);
                    events_info.emplace_back(hardware_counter_info(
                        true, tim::hardware_counters::api::rocm, events_info.size(), 0,
                        _sym, _pysym, _short_desc, _long_desc, _units,
                        qualifier_vec_t{ _device_qualifier }));
                }
                else
                {
                    auto _dim_info = std::vector<std::string>{};

                    for(const auto& itr : ditr.dimension_info)
                    {
                        auto _info = (itr.instance_size > 1)
                                         ? JOIN("", itr.name, "[", 0, ":",
                                                itr.instance_size - 1, "]")
                                         : std::string{};
                        if(!_info.empty()) _dim_info.emplace_back(_info);
                    }

                    auto _sym        = JOIN("", ditr.name, _device_qualifier_sym);
                    auto _short_desc = JOIN("", ditr.name, " on device ", _dev_index);
                    if(!_dim_info.empty())
                    {
                        namespace join = ::timemory::join;
                        _short_desc += JOIN(
                            "", ". ",
                            join::join(join::array_config{ ", ", "", "" }, _dim_info));
                    }
                    events_info.emplace_back(hardware_counter_info(
                        true, tim::hardware_counters::api::rocm, events_info.size(), 0,
                        _sym, _pysym, _short_desc, _long_desc, _units,
                        qualifier_vec_t{ _device_qualifier }));
                }
            }
        }
    } catch(std::exception& _e)
    {
        ROCPROFSYS_WARNING_F(1, "Constructing ROCm event info failed: %s\n", _e.what());
    }
}

void
client_data::set_agents()
{
    auto& agent_mngr = agent_manager::get_instance();

    auto fill_agents = [&](agent_type type, std::vector<tool_agent>& out) {
        const auto& _agents = agent_mngr.get_agents_by_type(type);
        for(const auto& agent : _agents)
        {
            out.emplace_back(tool_agent{ agent->handle, agent.get() });
        }
    };

    fill_agents(agent_type::GPU, gpu_agents);
    fill_agents(agent_type::CPU, cpu_agents);
}
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
