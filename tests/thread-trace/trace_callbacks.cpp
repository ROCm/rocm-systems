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
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "trace_callbacks.hpp"

#include <unistd.h>
#include <cassert>
#include <fstream>

namespace Callbacks
{
rocprofiler_thread_trace_decoder_id_t decoder{};
std::atomic<size_t>                   latency{0};

void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t* /* user_data */,
                              void* /* userdata */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_LOAD) return;

    auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);
    if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) return;

    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
    {
        DECODER_CALL(
            rocprofiler_thread_trace_decoder_codeobj_unload(decoder, data->code_object_id));
        return;
    }

    DECODER_CALL(rocprofiler_thread_trace_decoder_codeobj_load(
        decoder,
        data->code_object_id,
        data->load_delta,
        data->load_size,
        reinterpret_cast<const void*>(data->memory_base),
        data->memory_size));
}

typedef void (*rocprofiler_thread_trace_decoder_callback_t)(
    rocprofiler_thread_trace_decoder_record_type_t record_type_id,
    void*                                          trace_events,
    uint64_t                                       trace_size,
    void*                                          userdata);

void
shader_data_callback(rocprofiler_agent_id_t /* agent */,
                     int64_t /* se_id */,
                     void*  se_data,
                     size_t data_size,
                     rocprofiler_user_data_t /* userdata */)
{
    auto parse = [](rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                    void*                                          trace_events,
                    uint64_t                                       trace_size,
                    void* /* userdata */) {
        if(record_type_id != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE) return;

        for(size_t w = 0; w < trace_size; w++)
        {
            auto* wave = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(trace_events);
            for(size_t i = 0; i < wave->instructions_size; i++)
                latency += wave->instructions_array[i].duration;
        }
    };
    DECODER_CALL(rocprofiler_trace_decode(decoder, parse, se_data, data_size, nullptr));
}

void
init()
{
    // const char* decoder_lib = std::getenv("ROCPROF_TRACE_DECODER_PATH");
    DECODER_CALL(rocprofiler_thread_trace_decoder_create(&decoder, "/opt/rocm/lib"));
}

void
finalize(void* /* tool_data */)
{
    rocprofiler_thread_trace_decoder_destroy(decoder);

    if(latency.load() == 0) std::cerr << "Error: No latency was assigned to the trace!";
}

}  // namespace Callbacks
