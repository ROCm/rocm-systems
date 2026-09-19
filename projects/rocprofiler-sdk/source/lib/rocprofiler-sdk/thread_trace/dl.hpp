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

#pragma once

#include "lib/aqlprofile/aqlprofile.hpp"
#include "lib/rocprofiler-sdk/thread_trace/trace_decoder_api.h"

namespace rocprofiler
{
namespace thread_trace
{
class DL
{
    using ParseFn  = decltype(rocprof_trace_decoder_parse_data);
    using InfoFn   = decltype(rocprof_trace_decoder_get_info_string);
    using StatusFn = decltype(rocprof_trace_decoder_get_status_string);

    using CreateHandleFn      = decltype(rocprof_trace_decoder_create_handle);
    using DestroyHandleFn     = decltype(rocprof_trace_decoder_destroy_handle);
    using SetIsaCallbackFn    = decltype(rocprof_trace_decoder_set_isa_callback);
    using SetSeDataCallbackFn = decltype(rocprof_trace_decoder_set_se_data_callback);
    using SetAnalysisFn       = decltype(rocprof_trace_decoder_set_analysis);
    using HandleParseFn       = decltype(rocprof_trace_decoder_parse);

public:
    DL(const char* libpath);
    ~DL();
    DL(DL&)        = delete;
    DL(DL&& other) = delete;

    bool valid() const
    {
        return handle != nullptr && att_parse_data_fn != nullptr && att_info_fn != nullptr &&
               att_status_fn != nullptr;
    };

    /// Whether the decoder is new enough to run analyses, which need the handle-based API.
    bool supports_analysis() const
    {
        return att_create_handle_fn != nullptr && att_destroy_handle_fn != nullptr &&
               att_set_isa_callback_fn != nullptr && att_set_se_data_callback_fn != nullptr &&
               att_set_analysis_fn != nullptr && att_handle_parse_fn != nullptr;
    };

    ParseFn*  att_parse_data_fn = nullptr;
    InfoFn*   att_info_fn       = nullptr;
    StatusFn* att_status_fn     = nullptr;
    void*     handle            = nullptr;

    CreateHandleFn*      att_create_handle_fn        = nullptr;
    DestroyHandleFn*     att_destroy_handle_fn       = nullptr;
    SetIsaCallbackFn*    att_set_isa_callback_fn     = nullptr;
    SetSeDataCallbackFn* att_set_se_data_callback_fn = nullptr;
    SetAnalysisFn*       att_set_analysis_fn         = nullptr;
    HandleParseFn*       att_handle_parse_fn         = nullptr;
};

class AQLProfileDL
{
    using GetBufferPacketsFn   = decltype(aqlprofile_att_get_buffer_packets);
    using UpdateBufferStatusFn = decltype(aqlprofile_att_update_buffer_status);

public:
    AQLProfileDL();
    ~AQLProfileDL();

    bool valid() const
    {
        return get_buffer_packets_fn != nullptr && update_buffer_status_fn != nullptr;
    };

    GetBufferPacketsFn*   get_buffer_packets_fn   = nullptr;
    UpdateBufferStatusFn* update_buffer_status_fn = nullptr;
    void*                 handle                  = nullptr;
};

AQLProfileDL*
get_aqlprofile_dl();

}  // namespace thread_trace
}  // namespace rocprofiler
