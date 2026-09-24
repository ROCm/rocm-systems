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

// Windows stand-ins for the thread trace sources excluded from this build. ATT needs
// aqlprofile to build the trace control packets and the trace decoder shared library to
// read them back; neither is available to the ETW tracing path. The public API stays
// exported so that tools built against the headers still load; every entry point reports
// that the service is unavailable.

#include "lib/rocprofiler-sdk/thread_trace/core.hpp"
#include "lib/rocprofiler-sdk/thread_trace/dl.hpp"

#include <rocprofiler-sdk/experimental/thread-trace/agent.h>
#include <rocprofiler-sdk/experimental/thread-trace/dispatch.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstddef>

namespace rocprofiler
{
namespace thread_trace
{
// stands in for core.cpp
void
DispatchThreadTracer::start_context()
{}

void
DispatchThreadTracer::stop_context()
{}

void
DeviceThreadTracer::start_context()
{}

void
DeviceThreadTracer::stop_context()
{}

// stands in for dl.cpp
AQLProfileDL*
get_aqlprofile_dl()
{
    return nullptr;
}
}  // namespace thread_trace
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_configure_device_thread_trace_service(rocprofiler_context_id_t,
                                                  rocprofiler_agent_id_t,
                                                  rocprofiler_thread_trace_parameter_t*,
                                                  size_t,
                                                  rocprofiler_thread_trace_shader_data_callback_t,
                                                  rocprofiler_user_data_t)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_configure_dispatch_thread_trace_service(rocprofiler_context_id_t,
                                                    rocprofiler_agent_id_t,
                                                    rocprofiler_thread_trace_parameter_t*,
                                                    size_t,
                                                    rocprofiler_thread_trace_dispatch_callback_t,
                                                    rocprofiler_thread_trace_shader_data_callback_t,
                                                    void*)
{
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}
}
