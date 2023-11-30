// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>

#include "lib/rocprofiler-sdk/registration.hpp"

namespace
{
template <typename... Tp>
auto
consume_args(Tp&&...)
{}
}  // namespace

extern "C" {
rocprofiler_status_t
rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t         context_id,
                                          rocprofiler_agent_id_t           agent_id,
                                          rocprofiler_pc_sampling_method_t method,
                                          rocprofiler_pc_sampling_unit_t   unit,
                                          uint64_t                         interval,
                                          rocprofiler_buffer_id_t          buffer_id)
{
    if(rocprofiler::registration::get_init_status() > 0)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    consume_args(context_id, agent_id, method, unit, interval, buffer_id);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_pc_sampling_agent_configurations(
    rocprofiler_agent_id_t                                agent_id,
    rocprofiler_available_pc_sampling_configurations_cb_t cb,
    void*                                                 user_data)
{
    consume_args(agent_id, cb, user_data);
    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}
}
