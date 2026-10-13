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

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include "lib/common/utility.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

namespace
{
using info_t     = rocprofiler_kernel_dispatch_info_t;
using callback_t = rocprofiler_callback_tracing_kernel_dispatch_data_t;
using buffer_t   = rocprofiler_buffer_tracing_kernel_dispatch_record_t;
}  // namespace

// pipe_id lives on the completion records, so the dispatch info shared with counter collection,
// SPM and kernel replay keeps its original layout
TEST(kernel_dispatch_pipe_id, dispatch_info_layout_unchanged)
{
    EXPECT_EQ(sizeof(info_t), 128u);
    EXPECT_EQ(offsetof(info_t, grid_size), 60u);
    EXPECT_EQ(offsetof(info_t, reserved_padding), 72u);
    EXPECT_EQ(rocprofiler::common::compute_runtime_sizeof<info_t>(), 72u);
}

// pipe_id is appended directly after dispatch_info on both completion records
TEST(kernel_dispatch_pipe_id, completion_record_layout)
{
    static_assert(std::is_same<decltype(callback_t::pipe_id), int32_t>::value);
    static_assert(std::is_same<decltype(buffer_t::pipe_id), int32_t>::value);

    EXPECT_EQ(offsetof(callback_t, pipe_id), offsetof(callback_t, dispatch_info) + sizeof(info_t));
    EXPECT_EQ(offsetof(buffer_t, pipe_id), offsetof(buffer_t, dispatch_info) + sizeof(info_t));
    // the 4-byte field plus tail padding to the structs' 8-byte alignment
    EXPECT_EQ(sizeof(callback_t), offsetof(callback_t, pipe_id) + sizeof(uint64_t));
    EXPECT_EQ(sizeof(buffer_t), offsetof(buffer_t, pipe_id) + sizeof(uint64_t));
}
