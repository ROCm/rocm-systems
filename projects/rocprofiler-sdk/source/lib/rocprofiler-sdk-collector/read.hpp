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

#include <rocprofiler-sdk-collector/records.h>

#include "lib/rocprofiler-sdk-collector/clock.hpp"
#include "lib/rocprofiler-sdk-collector/hip_records.hpp"

#include <string>

namespace rocprofiler
{
namespace collector
{
struct read_request
{
    std::string                              trace_path       = {};
    std::string                              schema_dir       = {};
    clock_conversion                         clocks           = {};
    rocprofiler_collector_hip_api_callback_t hip_api_callback = nullptr;
    void*                                    user_data        = nullptr;
};

// The probe schema directory: the given one, else ROCPROFILER_COLLECTOR_SCHEMA_DIR, else
// share/rocm-trace in the installation of this library.
std::string
resolve_schema_dir(const char* schema_dir);

// Read a trace directory, join its calls, and deliver the HIP API calls as rocprofiler-sdk
// buffer records. The records' strings are kept in strings.
rocprofiler_collector_status_t
read_hip_api_calls(const read_request&                      request,
                   string_store*                            strings,
                   rocprofiler_collector_read_statistics_t* statistics);
}  // namespace collector
}  // namespace rocprofiler
