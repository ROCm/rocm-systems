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

#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include <cstdlib>
#include <iostream>

#define CHECK_HSA(fn, message)                                                                     \
    if((fn) != HSA_STATUS_SUCCESS)                                                                 \
    {                                                                                              \
        std::cerr << __FILE__ << ':' << __LINE__ << ' ' << message;                                \
        exit(1);                                                                                   \
    }

namespace rocprofiler
{
namespace hsa
{
CounterAQLPacket::~CounterAQLPacket()
{
    if(!profile.command_buffer.ptr)
    {
        // pass, nothing malloced
    }
    else if(!command_buf_mallocd)
    {
        CHECK_HSA(free_func(profile.command_buffer.ptr), "freeing memory");
    }
    else
    {
        ::free(profile.command_buffer.ptr);
    }

    if(!profile.output_buffer.ptr)
    {
        // pass, nothing malloced
    }
    else if(!output_buffer_malloced)
    {
        CHECK_HSA(free_func(profile.output_buffer.ptr), "freeing memory");
    }
    else
    {
        ::free(profile.output_buffer.ptr);
    }
}

TraceAQLPacket::~TraceAQLPacket() { aqlprofile_att_delete_packets(this->handle); }

hsa_status_t
TraceAQLPacket::Alloc(void** ptr, size_t size, aqlprofile_buffer_desc_flags_t flags, void* data)
{
    if(!data) return HSA_STATUS_ERROR;
    if(!reinterpret_cast<TraceAQLPacket*>(data)->tracepool) return HSA_STATUS_ERROR;

    auto& pool = *reinterpret_cast<TraceAQLPacket*>(data)->tracepool;

    if(!pool.allocate_fn || !pool.free_fn || !pool.allow_access_fn) return HSA_STATUS_ERROR;

    if(flags.host_access)
    {
        hsa_status_t status = pool.allocate_fn(pool.cpu_pool_, size, 0, ptr);
        if(!flags.device_access || status != HSA_STATUS_SUCCESS) return status;
        return pool.allow_access_fn(1, &pool.gpu_agent, nullptr, *ptr);
    }
    return pool.allocate_fn(pool.gpu_pool_, size, 0, ptr);
}

void
TraceAQLPacket::Free(void* ptr, void* data)
{
    auto* pool = reinterpret_cast<TraceAQLPacket*>(data)->tracepool.get();
    if(!pool || !pool->free_fn) return;

    pool->free_fn(ptr);
}

hsa_status_t
TraceAQLPacket::Copy(void* dst, const void* src, size_t size, void* data)
{
    auto* pool = reinterpret_cast<TraceAQLPacket*>(data)->tracepool.get();
    if(!pool || !pool->api_copy_fn) return HSA_STATUS_ERROR;

    return pool->api_copy_fn(dst, src, size);
}

TraceAQLPacket::TraceAQLPacket(std::shared_ptr<TraceMemoryPool>& _tracepool)
: tracepool(_tracepool){};

}  // namespace hsa
}  // namespace rocprofiler
