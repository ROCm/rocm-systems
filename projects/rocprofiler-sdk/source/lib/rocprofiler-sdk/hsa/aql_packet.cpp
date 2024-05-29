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
#include "lib/common/logging.hpp"

#define CHECK_HSA(fn, message)                                                                     \
    if((fn) != HSA_STATUS_SUCCESS)                                                                 \
    {                                                                                              \
        ROCP_ERROR << message;                                                                     \
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

hsa_status_t
BaseTTAQLPacket::Alloc(void** ptr, size_t size, desc_t flags, void* data)
{
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = reinterpret_cast<BaseTTAQLPacket*>(data)->tracepool;

    if(!pool.allocate_fn || !pool.free_fn || !pool.allow_access_fn) return HSA_STATUS_ERROR;

    hsa_status_t status = HSA_STATUS_ERROR;
    if(flags.host_access)
    {
        status = pool.allocate_fn(pool.cpu_pool_, size, 0, ptr);

        if(status == HSA_STATUS_SUCCESS)
            status = pool.allow_access_fn(1, &pool.gpu_agent, nullptr, *ptr);
    }
    else
    {
        // Return page aligned data to avoid cache flush overlap
        status = pool.allocate_fn(pool.gpu_pool_, size + 0x2000, 0, ptr);
        *ptr   = (void*) ((uintptr_t(*ptr) + 0xFFF) & ~0xFFFul);  // NOLINT
    }
    return status;
}

void
BaseTTAQLPacket::Free(void* ptr, void* data)
{
    assert(data);
    auto& pool = reinterpret_cast<BaseTTAQLPacket*>(data)->tracepool;

    if(pool.free_fn) pool.free_fn(ptr);
}

hsa_status_t
BaseTTAQLPacket::Copy(void* dst, const void* src, size_t size, void* data)
{
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = reinterpret_cast<BaseTTAQLPacket*>(data)->tracepool;

    if(!pool.api_copy_fn) return HSA_STATUS_ERROR;

    return pool.api_copy_fn(dst, src, size);
}

TraceControlAQLPacket::TraceControlAQLPacket(const TraceMemoryPool&          _tracepool,
                                             const aqlprofile_att_profile_t& p)
: BaseTTAQLPacket(_tracepool)
{
    auto status = aqlprofile_att_create_packets(&handle, &packets, p, &Alloc, &Free, &Copy, this);
    CHECK_HSA(status, "failed to create ATT packet");

    packets.start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets.stop_packet.header  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packets.start_packet.completion_signal = hsa_signal_t{.handle = 0};
    packets.stop_packet.completion_signal  = hsa_signal_t{.handle = 0};
    this->empty                            = false;
};

void
TraceControlAQLPacket::populate_before()
{
    before_krn_pkt.push_back(packets.start_packet);
    for(auto& [_, codeobj] : loaded_codeobj)
        if(codeobj) before_krn_pkt.push_back(codeobj->packet);
};

CodeobjMarkerAQLPacket::CodeobjMarkerAQLPacket(const TraceMemoryPool& _tracepool,
                                               uint64_t               id,
                                               uint64_t               addr,
                                               uint64_t               size,
                                               bool                   bFromStart,
                                               bool                   bIsUnload)
: BaseTTAQLPacket(_tracepool)
{
    aqlprofile_att_codeobj_data_t codeobj{};
    codeobj.id        = id;
    codeobj.addr      = addr;
    codeobj.size      = size;
    codeobj.agent     = _tracepool.gpu_agent;
    codeobj.isUnload  = bIsUnload;
    codeobj.fromStart = bFromStart;

    auto status = aqlprofile_att_codeobj_marker(&packet, &handle, codeobj, &Alloc, &Free, this);
    CHECK_HSA(status, "failed to create ATT marker");

    packet.header            = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packet.completion_signal = hsa_signal_t{.handle = 0};
    this->empty              = false;
}

}  // namespace hsa
}  // namespace rocprofiler
