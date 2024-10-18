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

#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/counters/device_counting.hpp"
#include "rocprofiler-sdk/fwd.h"

extern "C" {
rocprofiler_status_t
rocprofiler_configure_device_counting_service(rocprofiler_context_id_t context_id,
                                              rocprofiler_buffer_id_t  buffer_id,
                                              rocprofiler_agent_id_t   agent_id,
                                              rocprofiler_device_counting_service_callback_t cb,
                                              void* user_data)
{
    return rocprofiler::counters::configure_agent_collection(
        context_id, buffer_id, agent_id, cb, user_data);
}

rocprofiler_status_t
rocprofiler_sample_device_counting_service(rocprofiler_context_id_t   context_id,
                                           rocprofiler_user_data_t    user_data,
                                           rocprofiler_counter_flag_t flags)
{
    return rocprofiler::counters::read_agent_ctx(
        rocprofiler::context::get_registered_context(context_id), user_data, flags);
}
}
