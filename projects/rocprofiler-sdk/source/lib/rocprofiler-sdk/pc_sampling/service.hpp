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

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/common/synchronized.hpp"
#    include "lib/rocprofiler-sdk/context/context.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/types.hpp"

#    include <rocprofiler-sdk/fwd.h>
#    include <rocprofiler-sdk/pc_sampling.h>
#    include <rocprofiler-sdk/registration.h>

#    include <hsa/hsa_api_trace.h>

#    include <atomic>
#    include <memory>
#    include <unordered_map>

namespace rocprofiler
{
namespace pc_sampling
{
// Global map for O(1) agent ownership checking and lookups
// Maps agent ID to the PC sampling session configured for that agent
using global_pc_sampling_sessions_map_t =
    std::unordered_map<rocprofiler_agent_id_t, std::shared_ptr<PCSAgentSession>>;

common::Synchronized<global_pc_sampling_sessions_map_t>&
get_global_pc_sampling_sessions();

rocprofiler_status_t
start_service(const context::context* ctx);

rocprofiler_status_t
stop_service(const context::context* ctx);

void
post_hsa_init_start_active_service();

rocprofiler_status_t
configure_pc_sampling_service(context::context*                ctx,
                              const rocprofiler_agent_t*       agent,
                              rocprofiler_pc_sampling_method_t method,
                              rocprofiler_pc_sampling_unit_t   unit,
                              uint64_t                         interval,
                              rocprofiler_buffer_id_t          buffer_id);

bool
is_pc_sample_service_configured(rocprofiler_agent_id_t agent_id);

PCSAgentSession*
get_agent_session(rocprofiler_agent_id_t agent_id);

rocprofiler_status_t
flush_internal_agent_buffers(rocprofiler_buffer_id_t buffer_id);

void
service_sync(rocprofiler_client_id_t client_id);

void
service_fini();
}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
