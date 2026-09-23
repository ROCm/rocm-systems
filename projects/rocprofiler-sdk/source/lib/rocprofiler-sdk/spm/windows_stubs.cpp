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

// Windows stand-ins for the SPM sources excluded from this build. SPM streams counter
// data out of aqlprofile-built AQL packets, which the ETW tracing path cannot emit, so
// no SPM context can be started and no aqlprofile stream can be decoded. The public API
// stays exported so that tools built against the headers still load; every entry point
// reports that the service is unavailable.

#include "lib/rocprofiler-sdk/spm/core.hpp"
#include "lib/rocprofiler-sdk/spm/decode.hpp"
#include "lib/rocprofiler-sdk/spm/interface.hpp"

#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstddef>

namespace rocprofiler
{
namespace spm
{
// stands in for core.cpp
rocprofiler_status_t
start_context(const context::context*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

void
stop_context(const context::context*)
{}

// stands in for interface.cpp
const spm_interface*
construct_spm_interface()
{
    return nullptr;
}

// stands in for decode.cpp
void
aql_data_callback(size_t, void*, size_t, int, void*)
{}
}  // namespace spm
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_spm_query_agent_configurations(rocprofiler_agent_id_t,
                                           rocprofiler_spm_available_configurations_cb_t,
                                           void*)
{
    // No agent advertises a configuration, so the callback is never invoked. Reporting
    // success keeps callers that only enumerate configurations from treating this as fatal.
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_spm_create_counter_config(rocprofiler_agent_id_t,
                                      rocprofiler_counter_id_t*,
                                      size_t,
                                      rocprofiler_spm_parameters_t**,
                                      size_t,
                                      rocprofiler_counter_config_id_t*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_spm_configure_callback_dispatch_service(rocprofiler_context_id_t,
                                                    rocprofiler_spm_dispatch_counting_service_cb_t,
                                                    void*,
                                                    rocprofiler_spm_dispatch_counting_record_cb_t,
                                                    void*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}
}
