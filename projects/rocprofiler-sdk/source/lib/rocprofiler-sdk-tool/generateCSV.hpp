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

#include <rocprofiler-sdk/agent.h>

namespace rocprofiler
{
namespace tool
{
void
generate_csv(tool_table* tool_functions, std::vector<rocprofiler_agent_v0_t>& data);

void
generate_csv(tool_table* tool_functions, std::vector<kernel_dispatch_ring_buffer_t>& data);

void
generate_csv(tool_table* tool_functions, std::vector<hip_ring_buffer_t>& data);

void
generate_csv(tool_table* tool_functions, std::vector<hsa_ring_buffer_t>& data);

void
generate_csv(tool_table* tool_functions, std::vector<memory_copy_ring_buffer_t>& data);

void
generate_csv(tool_table* tool_functions, std::vector<marker_api_ring_buffer_t>& data);

void
generate_csv(tool_table* tool_functions, std::vector<counter_collection_ring_buffer_t>& data);

void
generate_csv(tool_table* tool_functions, std::vector<scratch_memory_ring_buffer_t>& data);
}  // namespace tool
}  // namespace rocprofiler
