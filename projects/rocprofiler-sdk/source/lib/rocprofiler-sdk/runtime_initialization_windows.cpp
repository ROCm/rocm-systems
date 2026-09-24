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

// Windows stand-in for runtime_initialization.cpp and intercept_table.cpp, both of which
// exist to service rocprofiler-register: a runtime announces itself, hands over its
// dispatch table, and rocprofiler overwrites the entries. That is the Linux interception
// model. On Windows HIP reports its activity as an ETW provider and rocprofiler-register
// is not in the tracing path, so no runtime ever announces itself and no table is ever
// handed over. Both APIs stay callable and report that there is nothing to intercept.

// ROCm's amd_hip_vector_types.h spells its members std::int32_t but does not include
// <cstdint> itself, so under MSVC it only compiles if something already has.
#include <cstdint>

#include "lib/rocprofiler-sdk/runtime_initialization.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>

#include <vector>

namespace rocprofiler
{
namespace runtime_init
{
const char* name_by_id(uint32_t) { return nullptr; }

std::vector<uint32_t>
get_ids()
{
    return {};
}
}  // namespace runtime_init
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_at_intercept_table_registration(rocprofiler_intercept_library_cb_t, int, void*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}

rocprofiler_status_t
rocprofiler_query_intercept_table_name(rocprofiler_intercept_table_t, const char**, uint64_t*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE;
}
}
