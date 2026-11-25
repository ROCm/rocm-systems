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

#include "kernel_dispatch_record.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace
{
using record_t = rocprofiler_buffer_tracing_kernel_dispatch_record_t;
using rocprofiler::tool::copy_kernel_dispatch_record;

// a core that predates pipe_id emits the record only up to the end of dispatch_info
constexpr size_t base_record_size = offsetof(record_t, pipe_id);

record_t
make_record(uint64_t size)
{
    auto record                      = record_t{};
    record.size                      = size;
    record.kind                      = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;
    record.operation                 = ROCPROFILER_KERNEL_DISPATCH_COMPLETE;
    record.correlation_id.internal   = 7;
    record.thread_id                 = 11;
    record.start_timestamp           = 100;
    record.end_timestamp             = 200;
    record.dispatch_info.size        = sizeof(rocprofiler_kernel_dispatch_info_t);
    record.dispatch_info.kernel_id   = 13;
    record.dispatch_info.dispatch_id = 17;
    record.dispatch_info.grid_size   = {64, 1, 1};
    record.pipe_id                   = 3;
    return record;
}

// the block is exactly record.size bytes, so AddressSanitizer reports any read past the
// record that the core emitted
std::unique_ptr<std::byte[]>
emit(const record_t& record)
{
    auto buffer = std::make_unique<std::byte[]>(record.size);
    std::memcpy(buffer.get(), &record, std::min<size_t>(record.size, sizeof(record)));
    return buffer;
}

record_t
copy(const std::unique_ptr<std::byte[]>& buffer)
{
    return copy_kernel_dispatch_record(reinterpret_cast<const record_t*>(buffer.get()));
}
}  // namespace

TEST(kernel_dispatch_record, base_layout_from_older_core_reports_no_pipe)
{
    const auto source = make_record(base_record_size);
    const auto data   = copy(emit(source));

    EXPECT_EQ(std::memcmp(&data, &source, base_record_size), 0)
        << "every member of the base layout must be copied";
    EXPECT_EQ(data.pipe_id, ROCPROFILER_KERNEL_DISPATCH_PIPE_ID_NONE);
}

TEST(kernel_dispatch_record, current_layout_keeps_pipe_id)
{
    const auto data = copy(emit(make_record(sizeof(record_t))));

    EXPECT_EQ(data.dispatch_info.dispatch_id, 17u);
    EXPECT_EQ(data.pipe_id, 3);
}

TEST(kernel_dispatch_record, longer_record_from_newer_core_keeps_known_members)
{
    const auto data = copy(emit(make_record(sizeof(record_t) + 16)));

    EXPECT_EQ(data.dispatch_info.dispatch_id, 17u);
    EXPECT_EQ(data.pipe_id, 3);
}
