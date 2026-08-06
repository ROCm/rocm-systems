// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace tool
{
namespace graph
{
/// Per-launch attribution captured on EXEC_LAUNCH ENTER.
struct attribution
{
    rocprofiler_graph_exec_id_t graph_exec_id = {.handle = 0};
    rocprofiler_graph_node_id_t node_counter  = {.handle = 0};
};

bool
graph_stack_not_null();

void
push(rocprofiler_graph_exec_id_t graph_exec_id);

void
pop();

/// Top of per-thread stack, or nullptr if no launch is in flight on this thread.
attribution*
current();

bool
graph_stack_empty();
}  // namespace graph
}  // namespace tool
}  // namespace rocprofiler
