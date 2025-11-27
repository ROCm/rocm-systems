// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "profile_interface.hpp"
#include "perfcounter.hpp"

#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>

#include <cstring>
#include <fstream>

namespace rocprofiler
{
namespace att_wrapper
{
void
get_trace_data(rocprofiler_thread_trace_decoder_record_type_t trace_id,
               void*                                          trace_events,
               size_t                                         trace_size,
               void*                                          userdata)
{
    C_API_BEGIN

    CHECK_NOTNULL(userdata);
    ToolData& tool = *static_cast<ToolData*>(userdata);

    if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO)
    {
        auto* infos = (rocprofiler_thread_trace_decoder_info_t*) trace_events;
        for(size_t i = 0; i < trace_size; i++)
            ROCP_WARNING << rocprofiler_thread_trace_decoder_info_string(tool.decoder, infos[i]);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP)
    {
        tool.config.filemgr->gfxip = reinterpret_cast<size_t>(trace_events);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY)
    {
        for(size_t i = 0; i < trace_size; i++)
            tool.config.occupancy.push_back(static_cast<const occupancy_t*>(trace_events)[i]);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT)
    {
        PerfcounterFile(tool.config, static_cast<perfevent_t*>(trace_events), trace_size);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY)
    {
        if(tool.config.realtime && trace_size != 0)
            tool.config.realtime->frequency = *static_cast<uint64_t*>(trace_events);
    }
    else if(trace_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME)
    {
        if(tool.config.realtime && trace_size != 0)
            tool.config.realtime->add(
                tool.config.shader_engine, static_cast<realtime_t*>(trace_events), trace_size);
    }

    if(trace_id != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE) return;

    bool bInvalid = false;
    for(size_t wave_n = 0; wave_n < trace_size; wave_n++)
    {
        const auto& wave           = static_cast<const wave_t*>(trace_events)[wave_n];
        int64_t     prev_inst_time = wave.begin_time;

        for(size_t j = 0; j < wave.instructions_size; j++)
        {
            const auto& inst = wave.instructions_array[j];
            if(inst.pc.code_object_id == 0 && inst.pc.address == 0) continue;

            try
            {
                auto& line = tool.get(inst.pc);
                line.hitcount += 1;
                line.latency += inst.duration;
                line.stall += inst.stall;
                line.idle += std::max<int64_t>(inst.time - prev_inst_time, 0);
            } catch(...)
            {
                bInvalid = true;
            }
            prev_inst_time = std::max(prev_inst_time, inst.time + inst.duration);
        }

        WaveFile(tool.config, wave);
    }
    if(bInvalid) ROCP_WARNING << "Could not fetch some instructions!";

    C_API_END
}

ToolData::ToolData(std::vector<char>&                    _data,
                   WaveConfig&                           _config,
                   rocprofiler_thread_trace_decoder_id_t _decoder)
: cfile(_config.code)
, config(_config)
, decoder(_decoder)
{
    auto status =
        rocprofiler_trace_decode(decoder, get_trace_data, _data.data(), _data.size(), this);
    ROCP_ERROR_IF(status != ROCPROFILER_STATUS_SUCCESS) << ": " << status;
}

ToolData::~ToolData() = default;

CodeLine& ToolData::get(pcinfo_t _pc) { return cfile->get(_pc); }

}  // namespace att_wrapper
}  // namespace rocprofiler
