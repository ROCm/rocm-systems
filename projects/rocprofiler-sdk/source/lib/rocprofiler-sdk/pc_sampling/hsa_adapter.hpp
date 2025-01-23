// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
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

#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/rocprofiler-sdk/context/context.hpp"
#    include "lib/rocprofiler-sdk/hsa/queue.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/types.hpp"
#    include "lib/rocprofiler-sdk/tracing/fwd.hpp"

#    include <hsa/hsa_api_trace.h>

namespace rocprofiler
{
namespace pc_sampling
{
namespace hsa
{
rocprofiler::hsa::rocprofiler_packet
generate_marker_packet_for_kernel(
    context::correlation_id*                      correlation_id,
    const tracing::external_correlation_id_map_t& external_correlation_ids,
    const rocprofiler_dispatch_id_t               dispatch_id);

void
pc_sampling_service_start(context::pc_sampling_service* service);

void
pc_sampling_service_stop(context::pc_sampling_service* service);

void
pc_sampling_service_finish_configuration(context::pc_sampling_service* service);

rocprofiler_status_t
flush_internal_agent_buffers(const PCSAgentSession* agent_session);
}  // namespace hsa
}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
