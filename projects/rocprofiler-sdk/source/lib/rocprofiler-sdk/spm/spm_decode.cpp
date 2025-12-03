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
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/spm/spm_decode.hpp"
#include "lib/rocprofiler-sdk/spm/spm_dlsym.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

#define CHECK_HSA(fn, message)                                                                     \
    {                                                                                              \
        auto _status = (fn);                                                                       \
        if(_status != HSA_STATUS_SUCCESS)                                                          \
        {                                                                                          \
            ROCP_ERROR << "HSA Err: " << _status << '\n';                                          \
            throw std::runtime_error(message);                                                     \
        }                                                                                          \
    }

namespace rocprofiler
{
namespace SPM
{
namespace
{
struct buffer_decode_callback_data_t
{
    hsa::SPMPacket* packet{};
    uint64_t        buffer_id{};
};
}  // namespace

void
decode_cb(uint64_t timestamp, uint64_t value, uint64_t index, int shader_engine, void* userdata)
{
    auto& counters = *reinterpret_cast<counter_vec*>(userdata);

    // aqlprofile currently reports shader_engine -1 as global counters
    if(shader_engine < 0)
    {
        counters.at(index).is_global = true;
        shader_engine                = 0;
    }

    if(counters.at(index).shaders.size() <= static_cast<size_t>(shader_engine))
        counters.at(index).shaders.resize(shader_engine + 1);

    auto& instance = counters.at(index).shaders.at(shader_engine);
    instance.timestamps.push_back(timestamp);
    instance.values.push_back(value);
}

void
aql_data_callback(size_t buffer_id, void* data, size_t data_size, int flags, void* userdata)
{
    SPM::counter_vec counters{};
    auto*            spm_packet = static_cast<hsa::SPMPacket*>(userdata);
    if(data_size == 0)
    {
        return;
    }

    auto& desc_v0 = *static_cast<rocprofiler::SPM::spm_desc_v0_t*>(spm_packet->spm_desc.data);
    if(!desc_v0.valid()) return;

    {
        uint64_t count = 0;

        auto status = spm_packet->sym.spm_query_fn(
            spm_packet->aql_desc, AQLPROFILE_SPM_DECODE_QUERY_EVENT_COUNT, &count);
        if(status != HSA_STATUS_SUCCESS) return;
        if(count != desc_v0.num_events) return;
        counters.resize(count);
    }

    for(auto& v : counters)
        v.shaders.resize(4);
    auto status =
        spm_packet->sym.spm_decode_fn(spm_packet->aql_desc, decode_cb, data, data_size, &counters);
    if(status != HSA_STATUS_SUCCESS) return;
    auto records = std::vector<rocprofiler_spm_counter_record_t>{};

    for(size_t i = 0; i < counters.size(); i++)
    {
        auto& event = desc_v0.events()[i];
        for(size_t se = 0; se < counters.at(i).shaders.size(); se++)
        {
            const auto& times  = counters.at(i).shaders.at(se).timestamps;
            const auto& values = counters.at(i).shaders.at(se).values;

            size_t size = std::min(times.size(), values.size());
            if(size == 0u) continue;

            if(counters.at(i).is_global)
            {
                auto instance_id = rocprofiler_counter_instance_id_t{};
                counters::set_dim_in_rec(
                    instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_XCC, buffer_id);
                counters::set_counter_in_rec(instance_id, event.id);
                counters::set_dim_in_rec(
                    instance_id,
                    counters::ROCPROFILER_DIMENSION_AGENT,
                    (rocprofiler::agent::get_rocprofiler_agent(spm_packet->GetAgent()))
                            ->logical_node_id +
                        counters::AGENT_ENCODING_OFFSET);
                for(size_t it = 0; it < size; it++)
                {
                    records.emplace_back(rocprofiler_spm_counter_record_t{
                        .size = sizeof(rocprofiler_spm_counter_record_t),
                        .id   = instance_id,
                        .agent_id =
                            (rocprofiler::agent::get_rocprofiler_agent(spm_packet->GetAgent()))->id,
                        .timestamp = times[it],
                        .value     = values[it]});
                }
            }
            else
            {
                auto instance_id = rocprofiler_counter_instance_id_t{};
                counters::set_dim_in_rec(
                    instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_XCC, buffer_id);
                counters::set_dim_in_rec(
                    instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_SHADER_ENGINE, se);
                counters::set_dim_in_rec(instance_id,
                                         rocprofiler::counters::ROCPROFILER_DIMENSION_INSTANCE,
                                         event.instance);
                counters::set_counter_in_rec(instance_id, event.id);
                counters::set_dim_in_rec(
                    instance_id,
                    counters::ROCPROFILER_DIMENSION_AGENT,
                    (rocprofiler::agent::get_rocprofiler_agent(spm_packet->GetAgent()))
                            ->logical_node_id +
                        counters::AGENT_ENCODING_OFFSET);
                for(size_t it = 0; it < size; it++)
                {
                    records.emplace_back(rocprofiler_spm_counter_record_t{
                        .size = sizeof(rocprofiler_spm_counter_record_t),
                        .id   = instance_id,
                        .agent_id =
                            (rocprofiler::agent::get_rocprofiler_agent(spm_packet->GetAgent()))->id,
                        .timestamp = times[it],
                        .value     = values[it]});
                }
            }
        }
    }

    spm_packet->record_cb(spm_packet->dispatch_data,
                          records.data(),
                          records.size(),
                          1 << ROCPROFILER_SPM_RECORD_FLAG_DATA | flags,
                          spm_packet->user_data,
                          spm_packet->record_callback_args);
}

}  // namespace SPM
}  // namespace rocprofiler
