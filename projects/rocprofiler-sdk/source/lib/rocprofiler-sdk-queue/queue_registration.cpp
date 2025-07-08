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

namespace {
    void WriteInterceptor(
        const void* packets,
        uint64_t    pkt_count,
        uint64_t    unused,
        void*       data,
        hsa_amd_queue_intercept_packet_writer writer)
    {
        ROCP_FATAL_IF(data == nullptr) << "WriteInterceptor was not passed a pointer to the queue";

        auto& queue_registration = *static_cast<QueueRegistration*>(data);
        queue_registration.write_interceptor_shim(packets, pkt_count, unused, data, writer);
    }
}


QueueRegistration::QueueRegistration(
    hsa_agent_t        agent,
    uint32_t           size,
    hsa_queue_type32_t type,
    callback_t         callback,
    void*              data,
    uint32_t           private_segment_size,
    uint32_t           group_segment_size,
    CoreApiTable       core_api,
    AmdExtTable        ext_api,
    hsa_queue_t**      queue)
: _agent(agent) 
{
    (void)core_api; // unused

    ROCP_HSA_TABLE_CALL(FATAL,
                        ext_api.hsa_amd_queue_intercept_create_fn(agent,
                                                                  size,
                                                                  type,
                                                                  callback,
                                                                  data,
                                                                  private_segment_size,
                                                                  group_segment_size,
                                                                  &_intercept_queue))
        << "Could not create intercept queue";

    ROCP_HSA_TABLE_CALL(FATAL,
                        ext_api.hsa_amd_profiling_set_profiler_enabled_fn(_intercept_queue, true))
        << "Could not setup intercept profiler";
    
    ROCP_HSA_TABLE_CALL(
        FATAL,
        ext_api.hsa_amd_queue_intercept_register_fn(_intercept_queue, WriteInterceptor, this))
        << "Could not register interceptor";

    *queue = _intercept_queue;
}

QueueRegistration::~QueueRegistration()
{

}

void QueueRegistration::set_write_interceptor(write_interceptor_t func, void* data)
{
    _write_interceptor_user_func = func;
    _write_interceptor_user_data = data;
}

void QueueRegistration::write_interceptor_shim
    (const void* packets,
    uint64_t     pkt_count,
    uint64_t     unused,
    void*        data,
    hsa_amd_queue_intercept_packet_writer_t writer)
{
    (void)data; // unused, should be this
    if (_write_interceptor_user_func)
    {
        _write_interceptor_user_func(packets, pkt_count, unused, _write_interceptor_user_data, writer);
    } else {
        writer(packets, pkt_count);
    }
}

}
}