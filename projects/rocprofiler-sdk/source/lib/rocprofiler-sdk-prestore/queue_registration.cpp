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

#include "queue_registration.hpp"

namespace rocprofiler {
namespace hsa {

queue_registration_t create_queue_registration(
    hsa_agent_t         agent,
    uint32_t            size,
    hsa_queue_type32_t  type,
    callback_t          callback,
    void*               data,
    uint32_t            private_segment_size,
    uint32_t            group_segment_size,
    CoreApiTable        core_api,
    AmdExtTable         ext_api,
    write_interceptor_t write_interceptor)
{
    (void)core_api; // unused

    hsa_queue_t* queue = nullptr;

    ROCP_HSA_TABLE_CALL(FATAL,
        ext_api.hsa_amd_queue_intercept_create_fn(
            agent,
            size,
            type,
            callback,
            data,
            private_segment_size,
            group_segment_size,
            &queue))
        << "Could not create intercept queue";

    ROCP_HSA_TABLE_CALL(FATAL,
        ext_api.hsa_amd_profiling_set_profiler_enabled_fn(queue, true))
        << "Could not setup intercept profiler";
    
    // Pass hsa_queue_t* as user data, used to identify 
    ROCP_HSA_TABLE_CALL(FATAL,
        ext_api.hsa_amd_queue_intercept_register_fn(queue, write_interceptor, queue))
        << "Could not register interceptor";

    queue_registration_t registration{};
    registration.agent = agent;
    registration.queue = queue;
    registration.user_write_interceptor_func = nullptr;
    registration.user_write_interceptor_data = nullptr;

    return registration;
}

}
}