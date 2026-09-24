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

// Windows stand-in for service.cpp. PC sampling delivers its samples through the KFD
// ioctl interface, which Windows reaches via D3DKMT instead; there is no session to
// start, stop, or flush. The public API stays exported so that tools built against the
// headers still load; every entry point reports that the service is unavailable.

#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>

namespace rocprofiler
{
namespace pc_sampling
{
rocprofiler_status_t
start_service(const context::context*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
stop_service(const context::context*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t flush_internal_agent_buffers(rocprofiler_buffer_id_t)
{
    // rocprofiler_flush_buffer() calls this before draining the buffer itself, so returning
    // an error here would break flushing for every tool, not just PC sampling ones. With no
    // service to configure there is nothing to drain, which is success.
    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace pc_sampling
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t,
                                          rocprofiler_agent_id_t,
                                          rocprofiler_pc_sampling_method_t,
                                          rocprofiler_pc_sampling_unit_t,
                                          uint64_t,
                                          rocprofiler_buffer_id_t,
                                          int)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_query_pc_sampling_agent_configurations(
    rocprofiler_agent_id_t,
    rocprofiler_available_pc_sampling_configurations_cb_t,
    void*)
{
    // No agent advertises a configuration, so the callback is never invoked. Reporting
    // success keeps callers that only enumerate configurations from treating this as fatal.
    return ROCPROFILER_STATUS_SUCCESS;
}

const char* rocprofiler_get_pc_sampling_instruction_type_name(
    rocprofiler_pc_sampling_instruction_type_t)
{
    return nullptr;
}

const char* rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
    rocprofiler_pc_sampling_instruction_not_issued_reason_t)
{
    return nullptr;
}
}
