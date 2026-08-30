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

#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <memory>

namespace rocprofiler
{
namespace pc_sampling
{
// True if the PC sampling service is configured on the given agent. Always
// linkable: returns false when PC sampling HSA support is unavailable. Used by
// the HSA write interceptor gate (in place of the per-queue callback that used
// to keep queue.get_notifiers() > 0) to decide whether a marker packet must be
// injected for this dispatch.
bool
is_configured_on_agent(rocprofiler_agent_id_t agent_id);

// Explicit replacement for the PC-sampling per-queue completion callback that
// used to be registered with the HSA queue controller. Notifies the CID manager
// that the kernel's correlation ID has completed. No-op when PC sampling HSA
// support is unavailable or when the service is not configured on the agent.
void
signal_completion_hook(const ::rocprofiler::hsa::Queue&                           queue,
                       const ::rocprofiler::hsa::rocprofiler_packet&              kernel_packet,
                       std::shared_ptr<::rocprofiler::hsa::queue_info_session_t>& session,
                       ::rocprofiler::hsa::packet_data_t&                         packet,
                       ::rocprofiler::hsa::inst_pkt_t&                            inst_pkt,
                       kernel_dispatch::profiling_time                            dispatch_time);
}  // namespace pc_sampling
}  // namespace rocprofiler
