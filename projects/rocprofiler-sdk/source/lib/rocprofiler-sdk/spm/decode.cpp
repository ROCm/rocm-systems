// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lib/rocprofiler-sdk/spm/interface.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

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

void
decode_cb(uint64_t timestamp, uint64_t value, uint64_t index, int shader_engine, void* userdata)
{
    auto& samples   = *reinterpret_cast<spm_sample_vec*>(userdata);
    bool  is_global = (shader_engine < 0);
    if(is_global) shader_engine = 0;
    samples.emplace_back(spm_sample_t{timestamp, value, index, shader_engine, is_global});
}

/** @brief  Callback for aqlprofile to return SPM data
 * buffer id - XCC of the data
 * flags - Indicates if there was a data loss
 */
void
aql_data_callback(size_t buffer_id, void* data, size_t data_size, int flags, void* userdata)
{
    auto* spm_packet = static_cast<hsa::SPMPacket*>(userdata);
    if(data_size == 0) return;

    auto& desc_v0 = *static_cast<rocprofiler::spm::spm_desc_v0_t*>(spm_packet->spm_desc.data);
    if(!desc_v0.valid()) return;

    {
        uint64_t count  = 0;
        auto     status = spm_packet->sym->spm_decode_query(
            spm_packet->aql_desc, AQLPROFILE_SPM_DECODE_QUERY_EVENT_COUNT, &count);
        if(status != HSA_STATUS_SUCCESS) return;
        if(count != desc_v0.num_events) return;
    }

    auto samples = spm_sample_vec{};
    auto status  = spm_packet->sym->spm_decode_stream_v1(
        spm_packet->aql_desc, decode_cb, data, data_size, &samples);
    if(status != HSA_STATUS_SUCCESS) return;

    auto records     = std::vector<std::unique_ptr<rocprofiler_spm_counter_record_t>>{};
    auto buf_records = std::vector<rocprofiler_spm_counter_record_t>{};

    rocprofiler::buffer::instance* buf = nullptr;
    if(spm_packet->buffer) buf = buffer::get_buffer(spm_packet->buffer->handle);

    auto agent_id    = (rocprofiler::agent::get_rocprofiler_agent(spm_packet->GetAgent()))->id;
    auto dispatch_id = spm_packet->dispatch_data.dispatch_info.dispatch_id;

    for(const auto& s : samples)
    {
        auto& event = desc_v0.events()[s.index];

        auto instance_id = rocprofiler_counter_instance_id_t{};
        counters::set_dim_in_rec(
            instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_XCC, buffer_id);
        counters::set_dim_in_rec(
            instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_INSTANCE, event.instance);
        counters::set_counter_in_rec(instance_id, event.id);
        if(!s.is_global)
            counters::set_dim_in_rec(instance_id,
                                     rocprofiler::counters::ROCPROFILER_DIMENSION_SHADER_ENGINE,
                                     s.shader_engine);

        if(buf)
        {
            buf_records.emplace_back(
                rocprofiler_spm_counter_record_t{sizeof(rocprofiler_spm_counter_record_t),
                                                 dispatch_id,
                                                 instance_id,
                                                 agent_id,
                                                 s.timestamp,
                                                 static_cast<double>(s.value)});
        }
        else
        {
            records.emplace_back(std::make_unique<rocprofiler_spm_counter_record_t>(
                rocprofiler_spm_counter_record_t{.size = sizeof(rocprofiler_spm_counter_record_t),
                                                 .dispatch_id = dispatch_id,
                                                 .id          = instance_id,
                                                 .agent_id    = agent_id,
                                                 .timestamp   = s.timestamp,
                                                 .value       = static_cast<double>(s.value)}));
        }
    }

    if(buf)
    {
        auto _lk = std::unique_lock{get_buffer_mut()};

        buf->emplace(ROCPROFILER_BUFFER_CATEGORY_COUNTERS,
                     ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER,
                     spm_packet->dispatch_data);
        for(const auto& itr : buf_records)
        {
            buf->emplace(
                ROCPROFILER_BUFFER_CATEGORY_COUNTERS, ROCPROFILER_COUNTER_RECORD_VALUE, itr);
        }
    }
    else if(spm_packet->record_cb)
    {
        auto record_ptrs = std::vector<const rocprofiler_spm_counter_record_t*>{};
        record_ptrs.reserve(records.size());
        for(const auto& rec : records)
            record_ptrs.push_back(rec.get());

        spm_packet->record_cb(&(spm_packet->dispatch_data),
                              record_ptrs.data(),
                              record_ptrs.size(),
                              (1 << ROCPROFILER_SPM_RECORD_FLAG_DATA) |
                                  (flags << ROCPROFILER_SPM_RECORD_FLAG_DATA_LOST),
                              spm_packet->user_data,
                              spm_packet->record_callback_args);
    }
}

}  // namespace spm
}  // namespace rocprofiler
