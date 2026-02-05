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
#include "lib/rocprofiler-sdk/spm/decode.hpp"
#include "lib/rocprofiler-sdk/spm/dlsym.hpp"

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
namespace spm
{
std::mutex&
get_buffer_mut()
{
    static auto*& mut = common::static_object<std::mutex>::construct();
    return *CHECK_NOTNULL(mut);
}

/** @brief Calback for every sample in SPM data buffer
 * [In] timestamp - timestamp of the sample
 * [In] value - value of the sample
 * [In] index - used to index the event map and retrieve the counter id of the sample
 * [In] shader_engine - -1 for global counters or the shader engine number
 */
void
decode_cb(uint64_t timestamp, uint64_t value, uint64_t index, int shader_engine, void* userdata)
{
    auto& counters = *reinterpret_cast<counter_vec*>(userdata);

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

/** @brief  Callback for aqlprofile to return SPM data
 * buffer id - XCC of the data
 * flags - Indicates if there was a data loss
 */
void
aql_data_callback(size_t buffer_id, void* data, size_t data_size, int flags, void* userdata)
{
    spm::counter_vec counters{};
    auto*            spm_packet = static_cast<hsa::SPMPacket*>(userdata);
    if(data_size == 0)
    {
        return;
    }

    auto& desc_v0 = *static_cast<rocprofiler::spm::spm_desc_v0_t*>(spm_packet->spm_desc.data);
    if(!desc_v0.valid()) return;

    {
        uint64_t count = 0;

        auto status = spm_packet->sym.spm_query_fn(
            spm_packet->aql_desc, AQLPROFILE_SPM_DECODE_QUERY_EVENT_COUNT, &count);
        if(status != HSA_STATUS_SUCCESS) return;
        if(count != desc_v0.num_events) return;
        counters.resize(count);
    }

    // Intially size to 4 shaders
    for(auto& v : counters)
        v.shaders.resize(4);

    // Decode SPM data and return vector of instances_t in counters list.
    auto status =
        spm_packet->sym.spm_decode_fn(spm_packet->aql_desc, decode_cb, data, data_size, &counters);

    if(status != HSA_STATUS_SUCCESS) return;

    auto records     = std::vector<const rocprofiler_spm_counter_record_t*>{};
    auto buf_records = std::vector<rocprofiler_spm_counter_record_t>{};

    rocprofiler::buffer::instance* buf = nullptr;
    buf                                = buffer::get_buffer(spm_packet->buffer->handle);

    for(size_t i = 0; i < counters.size(); i++)
    {
        auto& event = desc_v0.events()[i];
        for(size_t se = 0; se < counters.at(i).shaders.size(); se++)
        {
            const auto& times  = counters.at(i).shaders.at(se).timestamps;
            const auto& values = counters.at(i).shaders.at(se).values;

            size_t size = std::min(times.size(), values.size());
            if(size == 0u) continue;
            // Construct instance_id
            auto instance_id = rocprofiler_counter_instance_id_t{};
            counters::set_dim_in_rec(
                instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_XCC, buffer_id);
            counters::set_dim_in_rec(
                instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_INSTANCE, event.instance);
            counters::set_counter_in_rec(instance_id, event.id);
            counters::set_dim_in_rec(
                instance_id,
                counters::ROCPROFILER_DIMENSION_AGENT,
                (rocprofiler::agent::get_rocprofiler_agent(spm_packet->GetAgent()))
                        ->logical_node_id +
                    counters::AGENT_ENCODING_OFFSET);
            if(!counters.at(i).is_global)
                counters::set_dim_in_rec(
                    instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_SHADER_ENGINE, se);
            for(size_t it = 0; it < size; it++)
            {
                if(buf)
                {
                    buf_records.emplace_back(rocprofiler_spm_counter_record_t{
                        sizeof(rocprofiler_spm_counter_record_t),
                        spm_packet->dispatch_data.dispatch_info.dispatch_id,
                        instance_id,
                        (rocprofiler::agent::get_rocprofiler_agent(spm_packet->GetAgent()))->id,
                        times[it],
                        values[it]});
                }
                else

                    // Construct SPM record and add it to the buffer
                    records.emplace_back(new rocprofiler_spm_counter_record_t{
                        .size        = sizeof(rocprofiler_spm_counter_record_t),
                        .dispatch_id = spm_packet->dispatch_data.dispatch_info.dispatch_id,
                        .id          = instance_id,
                        .agent_id =
                            (rocprofiler::agent::get_rocprofiler_agent(spm_packet->GetAgent()))->id,
                        .timestamp = times[it],
                        .value     = values[it]});
            }
        }
    }
    if(buf)
    {
        auto _lk = std::unique_lock{get_buffer_mut()};  // Buffer records need to be in order

        buf->emplace(ROCPROFILER_BUFFER_CATEGORY_COUNTERS,
                     ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER,
                     spm_packet->dispatch_data);
        for(auto itr : buf_records)
        {
            if(itr.dispatch_id != spm_packet->dispatch_data.dispatch_info.dispatch_id)

                buf->emplace(
                    ROCPROFILER_BUFFER_CATEGORY_COUNTERS, ROCPROFILER_COUNTER_RECORD_VALUE, itr);
        }
    }
    else
    {
        // Return the buffer of SPM records to the tool
        spm_packet->record_cb(&(spm_packet->dispatch_data),
                              records.data(),
                              records.size(),
                              1 << ROCPROFILER_SPM_RECORD_FLAG_DATA | flags,
                              spm_packet->user_data,
                              spm_packet->record_callback_args);
        for(const auto* itr : records)
            delete(itr);
        records.clear();
    }
}

}  // namespace spm
}  // namespace rocprofiler
