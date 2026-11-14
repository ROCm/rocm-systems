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

#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"

#include <hsa/hsa_ext_amd.h>

#include <cstdint>

namespace rocprofiler
{
namespace range_replay
{
// Range replay's WriteInterceptor hook. Range replay never registered with the per-queue callback
// registry, but it follows the convention of the services that were migrated off it: hsa/queue.cpp
// calls this explicitly at one point, and range replay owns what happens there.
//
// Records a submission into the range open on the calling thread, or, when a range is open on
// another thread, lets that range see a foreign dispatch on its agent. Must be called before the
// packets are transformed, while their kernarg blocks still hold this launch's arguments, and
// ahead of any path that forwards them to the ring without instrumentation, so a graph launch
// inside a range is still seen and declines it.
//
// A no-op when no range is open anywhere, and while this thread is re-submitting a range's
// recording, since those packets are the recording rather than new work. queue is nullable on that
// path: the hook returns before it dereferences the queue, so tests can exercise it without an HSA
// runtime.
void
submission_hook(const hsa::Queue*                     queue,
                const hsa::rocprofiler_packet*        packets,
                uint64_t                              pkt_count,
                bool                                  graph_launch_active,
                hsa_amd_queue_intercept_packet_writer writer);
}  // namespace range_replay
}  // namespace rocprofiler
