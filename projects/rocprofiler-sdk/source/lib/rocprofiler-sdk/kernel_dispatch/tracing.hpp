// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <hsa/hsa.h>

#include <cstdint>

namespace rocprofiler
{
namespace context
{
struct context;
}

namespace kernel_dispatch
{
using context_t              = context::context;
using user_data_map_t        = std::unordered_map<const context_t*, rocprofiler_user_data_t>;
using external_corr_id_map_t = user_data_map_t;

using profiling_time = tracing::profiling_time;

profiling_time
get_dispatch_time(const hsa::queue_info_session& session);

void
dispatch_complete(hsa::queue_info_session&, profiling_time);
}  // namespace kernel_dispatch
}  // namespace rocprofiler
