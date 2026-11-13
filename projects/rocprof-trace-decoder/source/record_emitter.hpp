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

#include "record_filter.hpp"
#include "rocprof_trace_decoder/rocprof_trace_decoder.h"

#include <vector>

class RecordEmitter
{
public:
    RecordEmitter(rocprof_trace_decoder_trace_callback_t callback, void* userdata, const RecordFilter& filter = {}) :
    callback_(callback), userdata_(userdata), filter_(filter)
    {}

    bool enabled(rocprofiler_thread_trace_decoder_record_type_t type) const { return filter_.isEnabled(type); }

    bool immediate(rocprofiler_thread_trace_decoder_record_type_t type) const { return filter_.isImmediate(type); }

    rocprofiler_thread_trace_decoder_status_t status() const { return status_; }

    rocprofiler_thread_trace_decoder_status_t emit(
        rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t count
    ) const
    {
        if (status_ != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) return status_;
        if (!enabled(type)) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
        return status_ = callback_(type, data, count, userdata_);
    }

    template <typename Type> rocprofiler_thread_trace_decoder_status_t emit(
        rocprofiler_thread_trace_decoder_record_type_t type, Type& value
    ) const
    {
        return emit(type, &value, 1);
    }

    template <typename Type> rocprofiler_thread_trace_decoder_status_t append(
        rocprofiler_thread_trace_decoder_record_type_t type, std::vector<Type>& pooled, const Type& value
    ) const
    {
        if (status_ != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS || !enabled(type)) return status_;
        if (immediate(type))
        {
            Type copy = value;
            return emit(type, copy);
        }
        pooled.push_back(value);
        return status_;
    }

    template <typename Type> rocprofiler_thread_trace_decoder_status_t flush(
        rocprofiler_thread_trace_decoder_record_type_t type, std::vector<Type>& pooled
    ) const
    {
        if (status_ != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS || !enabled(type))
        {
            pooled.clear();
            return status_;
        }
        if (immediate(type))
        {
            for (auto& value : pooled)
                if (emit(type, value) != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) break;
        }
        else if (!pooled.empty())
            emit(type, pooled.data(), pooled.size());
        pooled.clear();
        return status_;
    }

private:
    rocprof_trace_decoder_trace_callback_t callback_ = nullptr;
    void* userdata_ = nullptr;
    RecordFilter filter_{};
    // Most parser call sites emit through void helpers. Preserve the first
    // callback failure so the C API can report it and never call the consumer
    // again after it has asked us to stop.
    mutable rocprofiler_thread_trace_decoder_status_t status_ = ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
};
