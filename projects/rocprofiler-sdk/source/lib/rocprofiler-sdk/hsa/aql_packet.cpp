// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/logging.hpp"

#include <fmt/core.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <mutex>

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
constexpr uint16_t VENDOR_BIT  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
constexpr uint16_t BARRIER_BIT = 1 << HSA_PACKET_HEADER_BARRIER;

// FIX #2: Track original (pre-alignment) GPU allocation pointers so Free()
// can release the correct address back to the HSA allocator.
// Aligned ptr -> original ptr
static std::unordered_map<void*, void*> s_gpu_aligned_to_orig;
static std::mutex                       s_gpu_ptr_mutex;

hsa_status_t
CounterAQLPacket::CounterMemoryPool::Alloc(void** ptr, size_t size, desc_t flags, void* data)
{
    if(size == 0)
    {
        if(ptr != nullptr) *ptr = nullptr;
        return HSA_STATUS_SUCCESS;
    }

    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<CounterAQLPacket::CounterMemoryPool*>(data);

    if(!pool.allocate_fn || !pool.free_fn || !pool.allow_access_fn) return HSA_STATUS_ERROR;
    if(!flags.host_access || pool.kernarg_pool_.handle == 0 || !pool.fill_fn)
        return HSA_STATUS_ERROR;

    hsa_status_t status;
    if(!pool.bIgnoreKernArg && flags.memory_hint == AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED)
        status =
            pool.allocate_fn(pool.kernarg_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);
    else
        status = pool.allocate_fn(pool.cpu_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);

    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_FATAL << "Could not allocate memory";
        return status;
    }

    status = pool.fill_fn(*ptr, 0u, size / sizeof(uint32_t));
    if(status != HSA_STATUS_SUCCESS) return status;

    status = pool.allow_access_fn(1, &pool.gpu_agent, nullptr, *ptr);
    return status;
}

void
CounterAQLPacket::CounterMemoryPool::Free(void* ptr, void* data)
{
    if(ptr == nullptr) return;

    assert(data);
    auto& pool = *reinterpret_cast<CounterAQLPacket::CounterMemoryPool*>(data);
    assert(pool.free_fn);
    pool.free_fn(ptr);
}

hsa_status_t
CounterAQLPacket::CounterMemoryPool::Copy(void* dst, const void* src, size_t size, void* data)
{
    if(size == 0) return HSA_STATUS_SUCCESS;
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<CounterAQLPacket::CounterMemoryPool*>(data);

    if(!pool.api_copy_fn) return HSA_STATUS_ERROR;

    return pool.api_copy_fn(dst, src, size);
}

CounterAQLPacket::CounterAQLPacket(aqlprofile_agent_handle_t                  agent,
                                   CounterAQLPacket::CounterMemoryPool        _pool,
                                   const std::vector<aqlprofile_pmc_event_t>& events)
: pool(_pool)
{
    if(events.empty()) return;

    packets.start_packet = null_amd_aql_pm4_packet;
    packets.stop_packet  = null_amd_aql_pm4_packet;
    packets.read_packet  = null_amd_aql_pm4_packet;

    aqlprofile_pmc_profile_t profile{};
    profile.agent       = agent;
    profile.events      = events.data();
    profile.event_count = static_cast<uint32_t>(events.size());

    ROCP_TRACE << "profile events count: " << profile.event_count;

    hsa_status_t status = aqlprofile_pmc_create_packets(&this->handle,
                                                        &this->packets,
                                                        profile,
                                                        &CounterMemoryPool::Alloc,
                                                        &CounterMemoryPool::Free,
                                                        &CounterMemoryPool::Copy,
                                                        reinterpret_cast<void*>(&pool));
    if(status != HSA_STATUS_SUCCESS)
    {
        std::string event_list;
        for(const auto& event : events)
        {
            event_list += fmt::format("[{},{},{}],",
                                      event.block_index,
                                      event.event_id,
                                      static_cast<int>(event.block_name));
        }
        ROCP_FATAL << "Could not create PMC packets! AQLProfile Return Code: " << status
                   << " Events: " << event_list;
    }

    packets.start_packet.header = VENDOR_BIT;
    packets.stop_packet.header  = VENDOR_BIT | BARRIER_BIT;
    packets.read_packet.header  = VENDOR_BIT | BARRIER_BIT;
    empty                       = false;
}

hsa_status_t
TraceMemoryPool::Alloc(void** ptr, size_t size, desc_t flags, void* data)
{
    if(ptr == nullptr) return HSA_STATUS_ERROR;

    if(size == 0)
    {
        *ptr = nullptr;
        return HSA_STATUS_SUCCESS;
    }

    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<TraceMemoryPool*>(data);

    if(!pool.allocate_fn || !pool.free_fn || !pool.allow_access_fn) return HSA_STATUS_ERROR;

    hsa_status_t status = HSA_STATUS_ERROR;

    ROCP_WARNING << "[ATT_ALLOC] begin"
                 << " host_access=" << flags.host_access
                 << " size=" << size
                 << " gpu_agent=" << pool.gpu_agent.handle;

    if(flags.host_access)
    {
        status = pool.allocate_fn(pool.cpu_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);

        ROCP_WARNING << "[ATT_ALLOC] cpu_pool allocate"
                     << " status=" << status
                     << " ptr_before_align=" << ((ptr && *ptr) ? *ptr : nullptr)
                     << " size=" << size
                     << " gpu_agent=" << pool.gpu_agent.handle;

        if(status == HSA_STATUS_SUCCESS)
        {
            status = pool.allow_access_fn(1, &pool.gpu_agent, nullptr, *ptr);

            ROCP_WARNING << "[ATT_ALLOC] cpu_pool allow_access"
                         << " status=" << status
                         << " ptr=" << *ptr
                         << " gpu_agent=" << pool.gpu_agent.handle;
        }

        if(status != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << "[ATT_ALLOC] cpu_pool failure"
                         << " status=" << status
                         << " ptr=" << ((ptr && *ptr) ? *ptr : nullptr);

            if(ptr && *ptr && pool.free_fn) pool.free_fn(*ptr);
            return status;
        }
    }
    else
    {
        // Allocate extra space for alignment padding
        const size_t alloc_size = size + 0x2000;
        void*        raw_ptr    = nullptr;

        status = pool.allocate_fn(
            pool.gpu_pool_, alloc_size, hsa_amd_memory_pool_executable_flag, &raw_ptr);

        ROCP_WARNING << "[ATT_ALLOC] gpu_pool allocate"
                     << " status=" << status
                     << " ptr_before_align=" << raw_ptr
                     << " size=" << size
                     << " alloc_size=" << alloc_size
                     << " gpu_agent=" << pool.gpu_agent.handle;

        if(status == HSA_STATUS_SUCCESS)
        {
            status = pool.allow_access_fn(1, &pool.gpu_agent, nullptr, raw_ptr);

            ROCP_WARNING << "[ATT_ALLOC] gpu_pool allow_access"
                         << " status=" << status
                         << " ptr_before_align=" << raw_ptr
                         << " gpu_agent=" << pool.gpu_agent.handle;
        }

        if(status != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << "[ATT_ALLOC] gpu_pool failure"
                         << " status=" << status
                         << " ptr_before_align=" << raw_ptr;

            if(raw_ptr && pool.free_fn) pool.free_fn(raw_ptr);
            return status;
        }

        // Align to 4KB boundary
        auto aligned_ptr =
            reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw_ptr) + 0xFFF) & ~0xFFFul);

        ROCP_WARNING << "[ATT_ALLOC] gpu_pool aligned"
                     << " ptr_before_align=" << raw_ptr
                     << " ptr_after_align=" << aligned_ptr
                     << " gpu_agent=" << pool.gpu_agent.handle;

        // FIX #2: Record mapping from aligned ptr -> original raw ptr
        // so that Free() can release the correct address.
        {
            std::lock_guard<std::mutex> lock(s_gpu_ptr_mutex);
            s_gpu_aligned_to_orig[aligned_ptr] = raw_ptr;
        }

        *ptr = aligned_ptr;
    }

    ROCP_WARNING << "[ATT_ALLOC] success"
                 << " host_access=" << flags.host_access
                 << " final_ptr=" << *ptr
                 << " size=" << size
                 << " gpu_agent=" << pool.gpu_agent.handle;

    return HSA_STATUS_SUCCESS;
}

void
TraceMemoryPool::Free(void* ptr, void* data)
{
    if(ptr == nullptr) return;

    assert(data);
    auto& pool = *reinterpret_cast<TraceMemoryPool*>(data);

    // FIX #2: If this was an aligned GPU pointer, free the original raw pointer instead.
    void* free_ptr = ptr;
    {
        std::lock_guard<std::mutex> lock(s_gpu_ptr_mutex);
        auto it = s_gpu_aligned_to_orig.find(ptr);
        if(it != s_gpu_aligned_to_orig.end())
        {
            free_ptr = it->second;
            s_gpu_aligned_to_orig.erase(it);
            ROCP_WARNING << "[ATT_ALLOC] free (aligned->orig)"
                         << " aligned_ptr=" << ptr
                         << " orig_ptr=" << free_ptr
                         << " gpu_agent=" << pool.gpu_agent.handle;
        }
        else
        {
            ROCP_WARNING << "[ATT_ALLOC] free"
                         << " ptr=" << ptr
                         << " gpu_agent=" << pool.gpu_agent.handle;
        }
    }

    if(pool.free_fn) pool.free_fn(free_ptr);
}

hsa_status_t
TraceMemoryPool::Copy(void* dst, const void* src, size_t size, void* data)
{
    if(size == 0) return HSA_STATUS_SUCCESS;
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<TraceMemoryPool*>(data);

    if(!pool.api_copy_fn) return HSA_STATUS_ERROR;

    return pool.api_copy_fn(dst, src, size);
}

TraceControlAQLPacket::TraceControlAQLPacket(const TraceMemoryPool&          _tracepool,
                                             const aqlprofile_att_profile_t& p)
: tracepool(std::make_shared<TraceMemoryPool>(_tracepool))
{
    ROCP_WARNING << "[ATT_CTRL_CREATE] begin"
                 << " gpu_agent=" << tracepool->gpu_agent.handle
                 << " cpu_pool=" << tracepool->cpu_pool_.handle
                 << " gpu_pool=" << tracepool->gpu_pool_.handle;

    // FIX #1: Zero-initialize packets before passing to aqlprofile.
    // Without this, aqlprofile may read garbage from the struct and embed
    // those values into GPU-side control structures. When the GPU later
    // executes and tries to write to a completion signal at one of those
    // garbage addresses, it triggers a memory access fault.
    memset(&packets, 0, sizeof(packets));

    auto status = aqlprofile_att_create_packets(&tracepool->handle,
                                                &packets,
                                                p,
                                                &TraceMemoryPool::Alloc,
                                                &TraceMemoryPool::Free,
                                                &TraceMemoryPool::Copy,
                                                tracepool.get());

    ROCP_WARNING << "[ATT_CTRL_CREATE] created"
                 << " status=" << status
                 << " gpu_agent=" << tracepool->gpu_agent.handle
                 << " start_header=" << packets.start_packet.header
                 << " stop_header=" << packets.stop_packet.header
                 << " start_completion=" << packets.start_packet.completion_signal.handle
                 << " stop_completion=" << packets.stop_packet.completion_signal.handle;

    CHECK_HSA(status, "failed to create ATT packet");

    // Normalize header and completion signal regardless of what aqlprofile wrote.
    packets.start_packet.header            = VENDOR_BIT | BARRIER_BIT;
    packets.stop_packet.header             = VENDOR_BIT | BARRIER_BIT;
    packets.start_packet.completion_signal = hsa_signal_t{.handle = 0};
    packets.stop_packet.completion_signal  = hsa_signal_t{.handle = 0};
    this->empty                            = false;

    clear();

    ROCP_WARNING << "[ATT_CTRL_CREATE] finalized"
                 << " gpu_agent=" << tracepool->gpu_agent.handle
                 << " start_header=" << packets.start_packet.header
                 << " stop_header=" << packets.stop_packet.header
                 << " start_completion=" << packets.start_packet.completion_signal.handle
                 << " stop_completion=" << packets.stop_packet.completion_signal.handle;
};

CodeobjMarkerAQLPacket::CodeobjMarkerAQLPacket(const TraceMemoryPool& _tracepool,
                                               uint64_t               id,
                                               uint64_t               addr,
                                               uint64_t               size,
                                               bool                   bFromStart,
                                               bool                   bIsUnload)
: tracepool(_tracepool)
{
    ROCP_WARNING << "[ATT_CODEOBJ_MARKER] begin"
                 << " id=" << id
                 << " addr=0x" << std::hex << addr
                 << " size=0x" << size
                 << std::dec
                 << " fromStart=" << bFromStart
                 << " isUnload=" << bIsUnload
                 << " gpu_agent=" << tracepool.gpu_agent.handle;

    aqlprofile_att_codeobj_data_t codeobj{};
    codeobj.id        = id;
    codeobj.addr      = addr;
    codeobj.size      = size;
    codeobj.agent     = tracepool.gpu_agent;
    codeobj.isUnload  = bIsUnload;
    codeobj.fromStart = bFromStart;

    // FIX #1: Zero-initialize packet before passing to aqlprofile (same reason as above).
    memset(&packet, 0, sizeof(packet));

    auto status = aqlprofile_att_codeobj_marker(&packet,
                                                &tracepool.handle,
                                                codeobj,
                                                &TraceMemoryPool::Alloc,
                                                &TraceMemoryPool::Free,
                                                &tracepool);

    ROCP_WARNING << "[ATT_CODEOBJ_MARKER] created"
                 << " status=" << status
                 << " id=" << id
                 << " gpu_agent=" << tracepool.gpu_agent.handle
                 << " packet_header=" << packet.header
                 << " completion_signal=" << packet.completion_signal.handle;

    CHECK_HSA(status, "failed to create ATT marker");

    packet.header            = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packet.completion_signal = hsa_signal_t{.handle = 0};
    this->empty              = false;

    clear();

    ROCP_WARNING << "[ATT_CODEOBJ_MARKER] finalized"
                 << " id=" << id
                 << " gpu_agent=" << tracepool.gpu_agent.handle
                 << " packet_header=" << packet.header
                 << " completion_signal=" << packet.completion_signal.handle;
}

}  // namespace hsa
}  // namespace rocprofiler
