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

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace hsa
{
namespace queue_hooks
{
// Whether any subsystem wants to see individual dispatch packets on this agent.
// When false, WriteInterceptor forwards the submission untouched: no signals are
// allocated and no packets are rewritten. Every subsystem that installs a write
// hook must be represented here, or its hook will never be reached.
bool
any_consumer_active(rocprofiler_agent_id_t agent);

// Whether a multi-packet submission may be processed as one batch. Subsystems that
// inject per-dispatch AQL packets (counters, ATT, SPM) need per-packet mode; PC
// sampling splices a single marker and tolerates batching.
bool
should_batch_packets();
}  // namespace queue_hooks
}  // namespace hsa
}  // namespace rocprofiler
