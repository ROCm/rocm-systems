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

#pragma once

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>

namespace rocprofiler {
namespace hsa {

// Saved registration for a single specific queue
class QueueRegistration
{
public:
    using hsa_amd_queue_intercept_packet_writer_t = void(*)(const void*, uint64_t);
    using write_interceptor_t = void(*)(const void*, uint64_t, uint64_t, void*, hsa_amd_queue_intercept_packet_writer_t);
    using callback_t = void (*)(hsa_status_t status, hsa_queue_t* source, void* data);

    QueueRegistration(
          hsa_agent_t        agent,
          uint32_t           size,
          hsa_queue_type32_t type,
          callback_t         callback,
          void*              data,
          uint32_t           private_segment_size,
          uint32_t           group_segment_size,
          CoreApiTable       core_api,
          AmdExtTable        ext_api,
          hsa_queue_t**      queue);
    virtual ~QueueRegistration();

    hsa_queue_t* intercept_queue() const { return _intercept_queue; } ROCPROFILER_PUBLIC_API;
    hsa_agent_t agent() const { return _agent; } ROCPROFILER_PUBLIC_API;
    void set_write_interceptor(write_interceptor_t func, void* data) ROCPROFILER_PUBLIC_API;
    
    void write_interceptor_shim(const void*, uint64_t, uint64_t, void*, hsa_amd_queue_intercept_packet_writer_t);
   
private:
    hsa_agent_t _agent = {};
    hsa_queue_t* _intercept_queue = nullptr;
    write_interceptor_t _write_interceptor_user_func = nullptr;
    void* _write_interceptor_user_data = nullptr;
};

}
}