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

#include "lib/common/demangle.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace rocprofiler
{
namespace collector
{
namespace test
{
// One rocpd_arg row, as rocprofv3 writes it: position, type, name, value.
using arg_row = std::tuple<uint32_t, std::string, std::string, std::string>;

// The rows rocprofiler-sdk's own formatting gives for a record (what rocprofv3's rocpd and
// JSON writers store).
inline std::vector<arg_row>
format_args(const rocprofiler_buffer_tracing_hip_api_ext_record_t& record)
{
    auto rows   = std::vector<arg_row>{};
    auto copy   = record;
    auto header = rocprofiler_record_header_t{};
    header.hash =
        rocprofiler_record_header_compute_hash(ROCPROFILER_BUFFER_CATEGORY_TRACING, record.kind);
    header.payload = &copy;
    auto status    = rocprofiler_iterate_buffer_tracing_record_args(
        header,
        [](rocprofiler_buffer_tracing_kind_t,
           rocprofiler_tracing_operation_t,
           uint32_t arg_number,
           const void* const,
           int32_t,
           const char* arg_type,
           const char* arg_name,
           const char* arg_value,
           void*       data) -> int {
            static_cast<std::vector<arg_row>*>(data)->emplace_back(
                arg_number, rocprofiler::common::cxx_demangle(arg_type), arg_name, arg_value);
            return 0;
        },
        &rows);
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS);
    return rows;
}
}  // namespace test
}  // namespace collector
}  // namespace rocprofiler
