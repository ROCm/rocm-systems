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

#pragma once

#include "rocprof_trace_decoder/trace_decoder_types.h"

#include <array>
#include <cstdint>

class RecordFilter
{
public:
    RecordFilter() { reset(); }

    void reset()
    {
        enabled.fill(true);
        flags.fill(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REQUEST_FLAGS_NONE);
    }

    void clear()
    {
        enabled.fill(false);
        flags.fill(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REQUEST_FLAGS_NONE);
    }

    bool isEnabled(rocprofiler_thread_trace_decoder_record_type_t type) const
    {
        return valid(type) && enabled[static_cast<size_t>(type)];
    }

    bool isImmediate(rocprofiler_thread_trace_decoder_record_type_t type) const
    {
        return isEnabled(type) && (flags[static_cast<size_t>(type)] &
                                   ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REQUEST_FLAGS_IMMEDIATE) != 0;
    }

    void set(rocprofiler_thread_trace_decoder_record_type_t type, uint32_t value)
    {
        enabled[static_cast<size_t>(type)] = true;
        flags[static_cast<size_t>(type)] = value;
    }

    static bool valid(rocprofiler_thread_trace_decoder_record_type_t type)
    {
        return type >= ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP &&
               type < ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST;
    }

    static bool supportsImmediate(rocprofiler_thread_trace_decoder_record_type_t type)
    {
        return type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA ||
               type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY ||
               type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME ||
               type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_MARKER;
    }

private:
    static constexpr size_t Count = static_cast<size_t>(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST);
    std::array<bool, Count> enabled{};
    std::array<uint32_t, Count> flags{};
};
