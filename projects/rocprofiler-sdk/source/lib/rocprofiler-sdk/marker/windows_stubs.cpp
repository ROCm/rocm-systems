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

// Windows stand-in for marker.cpp and range_marker.cpp. roctx is not built on Windows, so
// no roctx API table is ever captured and the marker domains carry no operations.

// ROCm's amd_hip_vector_types.h spells its members std::int32_t but does not include
// <cstdint> itself, so under MSVC it only compiles if something already has.
#include <cstdint>

#include "lib/rocprofiler-sdk/marker/marker.hpp"

#include <rocprofiler-sdk/marker/table_id.h>

#include <vector>

namespace rocprofiler
{
namespace marker
{
template <size_t TableIdx>
const char* name_by_id(uint32_t)
{
    return nullptr;
}

template <size_t TableIdx>
std::vector<uint32_t>
get_ids()
{
    return {};
}

template <size_t TableIdx>
void
iterate_args(uint32_t,
             const rocprofiler_callback_tracing_marker_api_data_t&,
             rocprofiler_callback_tracing_operation_args_cb_t,
             int32_t,
             void*)
{}

using iterate_args_data_t = rocprofiler_callback_tracing_marker_api_data_t;
using iterate_args_cb_t   = rocprofiler_callback_tracing_operation_args_cb_t;

#define INSTANTIATE_MARKER_STUB(TABLE_IDX)                                                         \
    template const char*           name_by_id<TABLE_IDX>(uint32_t);                                \
    template std::vector<uint32_t> get_ids<TABLE_IDX>();                                           \
    template void                  iterate_args<TABLE_IDX>(                                        \
        uint32_t, const iterate_args_data_t&, iterate_args_cb_t, int32_t, void*);

INSTANTIATE_MARKER_STUB(ROCPROFILER_MARKER_TABLE_ID_RoctxCore)
INSTANTIATE_MARKER_STUB(ROCPROFILER_MARKER_TABLE_ID_RoctxControl)
INSTANTIATE_MARKER_STUB(ROCPROFILER_MARKER_TABLE_ID_RoctxName)
INSTANTIATE_MARKER_STUB(ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange)

#undef INSTANTIATE_MARKER_STUB
}  // namespace marker
}  // namespace rocprofiler
