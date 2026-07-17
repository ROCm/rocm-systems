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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
namespace queue_hooks
{
// Tags identifying the producer subsystem of each inst_pkt_t entry.
// Only distinctness matters; values are fixed for test-order stability.
// Services are migrated off the per-queue callback registry one at a time; the
// unused ids are reserved so the tag values stay stable across those PRs.
constexpr int64_t COUNTERS_CLIENT_ID     = 1;
constexpr int64_t THREAD_TRACE_CLIENT_ID = 2;
constexpr int64_t PC_SAMPLING_CLIENT_ID  = 3;
constexpr int64_t SPM_CLIENT_ID          = 4;
}  // namespace queue_hooks
}  // namespace hsa
}  // namespace rocprofiler
