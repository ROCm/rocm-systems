// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "helper.hpp"
#include "statistics.hpp"

namespace rocprofiler
{
namespace tool
{
void
write_json(tool_table*                                                      tool_functions,
           uint64_t                                                         pid,
           const domain_stats_vec_t&                                        domain_stats,
           std::vector<rocprofiler_agent_v0_t>                              agent_data,
           std::vector<rocprofiler_tool_counter_info_t>                     counter_data,
           std::deque<rocprofiler_buffer_tracing_hip_api_record_t>*         hip_api_deque,
           std::deque<rocprofiler_buffer_tracing_hsa_api_record_t>*         hsa_api_deque,
           std::deque<rocprofiler_buffer_tracing_kernel_dispatch_record_t>* kernel_dispatch_deque,
           std::deque<rocprofiler_buffer_tracing_memory_copy_record_t>*     memory_copy_deque,
           std::deque<rocprofiler_tool_counter_collection_record_t>*       counter_collection_deque,
           std::deque<rocprofiler_buffer_tracing_marker_api_record_t>*     marker_api_deque,
           std::deque<rocprofiler_buffer_tracing_scratch_memory_record_t>* scratch_memory_deque,
           std::deque<rocprofiler_buffer_tracing_rccl_api_record_t>*       rccl_api_deque);

}  // namespace tool
}  // namespace rocprofiler
