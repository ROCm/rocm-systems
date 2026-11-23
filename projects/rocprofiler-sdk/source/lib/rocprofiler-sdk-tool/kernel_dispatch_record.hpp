// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace rocprofiler
{
namespace tool
{
// The record may come from a core of another version: only record->size bytes are read, and
// the members that an older core's shorter record does not cover are default-filled (pipe_id
// becomes ROCPROFILER_KERNEL_DISPATCH_PIPE_ID_NONE).
inline rocprofiler_buffer_tracing_kernel_dispatch_record_t
copy_kernel_dispatch_record(const rocprofiler_buffer_tracing_kernel_dispatch_record_t* record)
{
    using record_t = rocprofiler_buffer_tracing_kernel_dispatch_record_t;

    auto data = record_t{};
    std::memcpy(&data, record, std::min<size_t>(record->size, sizeof(data)));
    if(record->size < offsetof(record_t, pipe_id) + sizeof(data.pipe_id))
        data.pipe_id = ROCPROFILER_KERNEL_DISPATCH_PIPE_ID_NONE;
    return data;
}
}  // namespace tool
}  // namespace rocprofiler
