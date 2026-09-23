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

// Windows stand-ins for the HIP sources excluded from this build. names.cpp supplies the
// real operation names, so the HIP domains report their full operation list; what is
// missing here is everything that needs the captured dispatch table. iterate_args walks a
// rocprofiler_hip_api_args_t that only the interception path ever fills in, and the event,
// graph and stream domains are driven by the intercepted HSA queue. Under ETW the runtime
// is observed from outside the process, so none of that exists.

// ROCm's amd_hip_vector_types.h spells its members std::int32_t but does not include
// <cstdint> itself, so under MSVC it only compiles if something already has.
#include <cstdint>

#include "lib/rocprofiler-sdk/hip/event.hpp"
#include "lib/rocprofiler-sdk/hip/graph.hpp"
#include "lib/rocprofiler-sdk/hip/hip.hpp"
#include "lib/rocprofiler-sdk/hip/stream.hpp"

#include <rocprofiler-sdk/hip/table_id.h>

#include <vector>

namespace rocprofiler
{
namespace hip
{
template <size_t TableIdx>
void
iterate_args(uint32_t,
             const rocprofiler_hip_api_args_t&,
             rocprofiler_callback_tracing_operation_args_cb_t,
             int32_t,
             void*)
{}

using iterate_args_data_t = rocprofiler_hip_api_args_t;
using iterate_args_cb_t   = rocprofiler_callback_tracing_operation_args_cb_t;

template void
iterate_args<ROCPROFILER_HIP_TABLE_ID_Runtime>(uint32_t,
                                               const iterate_args_data_t&,
                                               iterate_args_cb_t,
                                               int32_t,
                                               void*);
template void
iterate_args<ROCPROFILER_HIP_TABLE_ID_Compiler>(uint32_t,
                                                const iterate_args_data_t&,
                                                iterate_args_cb_t,
                                                int32_t,
                                                void*);

namespace event
{
const char* name_by_id(uint32_t) { return nullptr; }

std::vector<uint32_t>
get_ids()
{
    return {};
}

void
set_service_configured(bool)
{}
}  // namespace event

namespace graph
{
const char* name_by_id(uint32_t) { return nullptr; }

std::vector<uint32_t>
get_ids()
{
    return {};
}
}  // namespace graph

namespace stream
{
const char* name_by_id(uint32_t) { return nullptr; }

std::vector<uint32_t>
get_ids()
{
    return {};
}
}  // namespace stream
}  // namespace hip
}  // namespace rocprofiler
