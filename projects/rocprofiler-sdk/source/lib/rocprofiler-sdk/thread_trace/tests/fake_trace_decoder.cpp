// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Stands in for librocprof-trace-decoder.so so the SDK's decode path can be exercised without
// a GPU, a capture, or a decoder package. Built twice: once exporting the handle-based API of
// a 0.2.3 decoder, and once exporting only what a 0.2.2 decoder had.

#include "lib/rocprofiler-sdk/thread_trace/tests/fake_trace_decoder.hpp"

#include "lib/rocprofiler-sdk/thread_trace/trace_decoder_api.h"

#define FAKE_DECODER_EXPORT __attribute__((visibility("default")))

namespace
{
fake_trace_decoder_log_t  call_log{};
fake_trace_decoder_step_t failing_step = FAKE_TRACE_DECODER_STEP_NONE;

rocprofiler_thread_trace_decoder_wave_t           wave_record{};
rocprofiler_thread_trace_decoder_hidden_latency_t hidden_latency_record{};

#if defined(FAKE_TRACE_DECODER_SUPPORTS_ANALYSIS) && FAKE_TRACE_DECODER_SUPPORTS_ANALYSIS
bool
fails(fake_trace_decoder_step_t step)
{
    return failing_step == step;
}

struct fake_handle_state
{
    rocprof_trace_decoder_isa_callback_t     isa_callback{nullptr};
    void*                                    isa_userdata{nullptr};
    rocprof_trace_decoder_se_data_callback_t se_data_callback{nullptr};
    void*                                    se_data_userdata{nullptr};
    uint64_t                                 analysis_flags{0};
};

fake_handle_state open_handle{};
#endif

// A real decoder pulls shader engine data until the callback reports none left, so do the same
// and record the total. This is what proves the SDK handed over a working data source.
void
drain_se_data(rocprof_trace_decoder_se_data_callback_t callback, void* userdata)
{
    if(callback == nullptr) return;

    uint8_t* buffer = nullptr;
    uint64_t size   = 0;
    while(callback(&buffer, &size, userdata) != 0)
        call_log.se_data_bytes += size;
}

rocprofiler_thread_trace_decoder_status_t
emit_records(rocprof_trace_decoder_trace_callback_t callback, void* userdata, uint64_t flags)
{
    auto status = callback(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE, &wave_record, 1, userdata);
    if(status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) return status;

    if((flags & ROCPROFILER_THREAD_TRACE_DECODER_ANALYSIS_HIDDEN_LATENCY) == 0)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;

    hidden_latency_record.size = sizeof(hidden_latency_record);
    return callback(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_HIDDEN_LATENCY,
                    &hidden_latency_record,
                    1,
                    userdata);
}
}  // namespace

extern "C" {

FAKE_DECODER_EXPORT fake_trace_decoder_log_t*
                    fake_trace_decoder_log(void)
{
    return &call_log;
}

FAKE_DECODER_EXPORT void
fake_trace_decoder_fail_at(fake_trace_decoder_step_t step)
{
    failing_step = step;
}

FAKE_DECODER_EXPORT rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_parse_data(rocprof_trace_decoder_se_data_callback_t se_data_callback,
                                 rocprof_trace_decoder_trace_callback_t   trace_callback,
                                 rocprof_trace_decoder_isa_callback_t     isa_callback,
                                 void*                                    userdata)
{
    (void) isa_callback;
    call_log.parse_data_calls++;
    drain_se_data(se_data_callback, userdata);
    return emit_records(trace_callback, userdata, ROCPROFILER_THREAD_TRACE_DECODER_ANALYSIS_NONE);
}

FAKE_DECODER_EXPORT const char*
rocprof_trace_decoder_get_info_string(rocprofiler_thread_trace_decoder_info_t info)
{
    (void) info;
    return "fake decoder info";
}

FAKE_DECODER_EXPORT const char*
rocprof_trace_decoder_get_status_string(rocprofiler_thread_trace_decoder_status_t status)
{
    (void) status;
    return "fake decoder status";
}

// Present on every decoder from 0.2.2 onwards. Without it the SDK warns that the decoder
// predates event records, which is not the scenario either variant is standing in for.
typedef struct
{
    uint64_t size;
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} rocprof_trace_decoder_version_t;

FAKE_DECODER_EXPORT rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_get_version(rocprof_trace_decoder_version_t* version)
{
    if(version == nullptr) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    version->size  = sizeof(*version);
    version->major = 0;
    version->minor = 2;
#if defined(FAKE_TRACE_DECODER_SUPPORTS_ANALYSIS) && FAKE_TRACE_DECODER_SUPPORTS_ANALYSIS
    version->patch = 3;
#else
    version->patch = 2;
#endif
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

#if defined(FAKE_TRACE_DECODER_SUPPORTS_ANALYSIS) && FAKE_TRACE_DECODER_SUPPORTS_ANALYSIS

FAKE_DECODER_EXPORT rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_create_handle(rocprof_trace_decoder_handle_t* handle)
{
    if(handle == nullptr) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    call_log.create_handle_calls++;
    if(fails(FAKE_TRACE_DECODER_STEP_CREATE_HANDLE))
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;

    open_handle    = fake_handle_state{};
    handle->handle = call_log.create_handle_calls;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

FAKE_DECODER_EXPORT rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_destroy_handle(rocprof_trace_decoder_handle_t handle)
{
    (void) handle;
    call_log.destroy_handle_calls++;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

FAKE_DECODER_EXPORT rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_set_isa_callback(rocprof_trace_decoder_handle_t       handle,
                                       rocprof_trace_decoder_isa_callback_t callback,
                                       void*                                userdata)
{
    (void) handle;
    call_log.set_isa_callback_calls++;
    if(fails(FAKE_TRACE_DECODER_STEP_SET_ISA_CALLBACK))
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    open_handle.isa_callback = callback;
    open_handle.isa_userdata = userdata;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

FAKE_DECODER_EXPORT rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_set_se_data_callback(rocprof_trace_decoder_handle_t           handle,
                                           rocprof_trace_decoder_se_data_callback_t callback,
                                           void*                                    userdata)
{
    (void) handle;
    call_log.set_se_data_callback_calls++;
    if(fails(FAKE_TRACE_DECODER_STEP_SET_SE_DATA_CALLBACK))
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    open_handle.se_data_callback = callback;
    open_handle.se_data_userdata = userdata;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

FAKE_DECODER_EXPORT rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_set_analysis(rocprof_trace_decoder_handle_t handle, uint64_t flags)
{
    (void) handle;
    call_log.set_analysis_calls++;
    if(fails(FAKE_TRACE_DECODER_STEP_SET_ANALYSIS))
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    call_log.analysis_flags    = flags;
    open_handle.analysis_flags = flags;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

FAKE_DECODER_EXPORT rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_parse(rocprof_trace_decoder_handle_t         handle,
                            const void*                            data,
                            uint64_t                               data_size,
                            rocprof_trace_decoder_trace_callback_t trace_callback,
                            void*                                  userdata)
{
    (void) handle;
    (void) data;
    (void) data_size;
    call_log.handle_parse_calls++;
    if(fails(FAKE_TRACE_DECODER_STEP_PARSE))
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA;

    drain_se_data(open_handle.se_data_callback, open_handle.se_data_userdata);
    return emit_records(trace_callback, userdata, open_handle.analysis_flags);
}

#endif
}
