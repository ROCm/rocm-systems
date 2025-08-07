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
#include <fmt/core.h>
#include <cstdlib>
#include <iostream>
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

namespace rocprofiler
{
namespace hsa
{
constexpr uint16_t VENDOR_BIT  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
constexpr uint16_t BARRIER_BIT = 1 << HSA_PACKET_HEADER_BARRIER;

AQLMemoryPool::AQLMemoryPool(const AgentCache& agent, const AmdExtTable& ext, copy_fn_t copy_fn)
{
    allocate_fn     = ext.hsa_amd_memory_pool_allocate_fn;
    allow_access_fn = ext.hsa_amd_agents_allow_access_fn;
    free_fn         = ext.hsa_amd_memory_pool_free_fn;
    fill_fn         = ext.hsa_amd_memory_fill_fn;
    api_copy_fn     = copy_fn;

    gpu_agent     = agent.get_hsa_agent();
    cpu_pool_     = agent.cpu_pool();
    gpu_pool_     = agent.gpu_pool();
    kernarg_pool_ = agent.kernarg_pool();
}

void
AQLMemoryPool::Free(void* ptr, void* data)
{
    if(ptr == nullptr) return;
    auto* pool = reinterpret_cast<AQLMemoryPool*>(data);

    ROCP_FATAL_IF(!pool || !pool->free_fn) << "Unable to deallocate from HSA memory pool";
    pool->free_fn(ptr);
}

hsa_status_t
AQLMemoryPool::Copy(void* dst, const void* src, size_t size, void* data)
{
    if(size == 0) return HSA_STATUS_SUCCESS;
    auto* pool = reinterpret_cast<AQLMemoryPool*>(data);
    ROCP_FATAL_IF(!pool || !pool->api_copy_fn) << "Unable to copy HSA memory";

    return pool->api_copy_fn(dst, src, size);
}

hsa_status_t
AQLMemoryPool::Alloc(void** ptr, size_t size, desc_t flags, void* data)
{
    if(size == 0)
    {
        if(ptr != nullptr) *ptr = nullptr;
        return HSA_STATUS_SUCCESS;
    }
    if(!data) return HSA_STATUS_ERROR;
    return static_cast<AQLMemoryPool*>(data)->Alloc(ptr, size, flags);
}

hsa_status_t
CounterAQLPacket::CounterMemoryPool::Alloc(void** ptr, size_t size, desc_t flags)
{
    if(!allocate_fn || !free_fn || !allow_access_fn) return HSA_STATUS_ERROR;
    if(!flags.host_access || kernarg_pool_.handle == 0 || !fill_fn) return HSA_STATUS_ERROR;

    hsa_status_t status;
    if(!bIgnoreKernArg && flags.memory_hint == AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED)
        status = allocate_fn(kernarg_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);
    else
        status = allocate_fn(cpu_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);

    if(status != HSA_STATUS_SUCCESS) return status;

    status = fill_fn(*ptr, 0u, size / sizeof(uint32_t));
    if(status != HSA_STATUS_SUCCESS) return status;

    status = allow_access_fn(1, &gpu_agent, nullptr, *ptr);
    return status;
}

hsa_status_t
TraceMemoryPool::Alloc(void** ptr, size_t size, desc_t flags)
{
    if(!allocate_fn || !free_fn || !allow_access_fn) return HSA_STATUS_ERROR;

    hsa_status_t status = HSA_STATUS_ERROR;
    if(flags.host_access)
    {
        status = allocate_fn(cpu_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);

        if(status == HSA_STATUS_SUCCESS) status = allow_access_fn(1, &gpu_agent, nullptr, *ptr);
    }
    else
    {
        // Return page aligned data to avoid cache flush overlap
        status = allocate_fn(gpu_pool_, size + 0x2000, hsa_amd_memory_pool_executable_flag, ptr);
        *ptr = (void*) ((uintptr_t(*ptr) + 0xFFF) & ~0xFFFul);  // NOLINT(performance-no-int-to-ptr)
    }
    return status;
}

CounterAQLPacket::CounterAQLPacket(aqlprofile_agent_handle_t                  agent,
                                   CounterAQLPacket::CounterMemoryPool        _pool,
                                   const std::vector<aqlprofile_pmc_event_t>& events)
{
    if(events.empty()) return;
    this->pool = std::make_shared<CounterAQLPacket::CounterMemoryPool>(_pool);

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
                                                        &AQLMemoryPool::Alloc,
                                                        &AQLMemoryPool::Free,
                                                        &AQLMemoryPool::Copy,
                                                        pool.get());
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

TraceControlAQLPacket::TraceControlAQLPacket(const TraceMemoryPool&          _tracepool,
                                             const aqlprofile_att_profile_t& p)
{
    this->tracepool = std::make_shared<TraceMemoryPool>(_tracepool);
    this->pool      = tracepool;
    auto status     = aqlprofile_att_create_packets(&tracepool->handle,
                                                &packets,
                                                p,
                                                &AQLMemoryPool::Alloc,
                                                &AQLMemoryPool::Free,
                                                &AQLMemoryPool::Copy,
                                                pool.get());
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to create ATT packet";

    packets.start_packet.header            = VENDOR_BIT | BARRIER_BIT;
    packets.stop_packet.header             = VENDOR_BIT | BARRIER_BIT;
    packets.start_packet.completion_signal = hsa_signal_t{.handle = 0};
    packets.stop_packet.completion_signal  = hsa_signal_t{.handle = 0};

    this->empty  = false;
    this->handle = tracepool->handle;
    clear();
};

CodeobjMarkerAQLPacket::CodeobjMarkerAQLPacket(const TraceMemoryPool& _pool,
                                               uint64_t               id,
                                               uint64_t               addr,
                                               uint64_t               size,
                                               bool                   bFromStart,
                                               bool                   bIsUnload)
{
    auto tracepool = std::make_shared<TraceMemoryPool>(_pool);
    this->pool     = tracepool;

    aqlprofile_att_codeobj_data_t codeobj{};
    codeobj.id        = id;
    codeobj.addr      = addr;
    codeobj.size      = size;
    codeobj.agent     = pool->gpu_agent;
    codeobj.isUnload  = bIsUnload;
    codeobj.fromStart = bFromStart;

    auto status = aqlprofile_att_codeobj_marker(&packet.ext_amd_aql_pm4,
                                                &tracepool->handle,
                                                codeobj,
                                                &AQLMemoryPool::Alloc,
                                                &AQLMemoryPool::Free,
                                                pool.get());
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to create ATT packet";

    packet.ext_amd_aql_pm4.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packet.ext_amd_aql_pm4.completion_signal = hsa_signal_t{.handle = 0};

    this->empty  = false;
    this->handle = tracepool->handle;
    clear();
}

hsa_status_t
SPMMemoryPool::Alloc(void** ptr, size_t size, desc_t flags)
{
    if(!allocate_fn || !free_fn || !allow_access_fn) return HSA_STATUS_ERROR;

    hsa_status_t status = HSA_STATUS_ERROR;

    if(flags.host_access)
        status = allocate_fn(cpu_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);
    else
        status = allocate_fn(kernarg_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);

    if(status == HSA_STATUS_SUCCESS) status = allow_access_fn(1, &gpu_agent, nullptr, *ptr);
    if(status == HSA_STATUS_SUCCESS) status = fill_fn(*ptr, 0u, size / sizeof(uint32_t));

    return status;
}

SPMPacket::SPMPacket(const aqlprofile_spm_profile_t& profile, rocprofiler_agent_id_t _agent_id)
: agent_id(_agent_id)
, sym()
{
    ROCP_FATAL_IF(!sym.valid()) << "Failed to load aqlprofile SPM library";

    auto status = sym.create_packets_fn(&handle, &aql_desc, &packets, profile, 0);
    if(status != HSA_STATUS_SUCCESS) return;

    packets.start_packet.header            = VENDOR_BIT | BARRIER_BIT;
    packets.stop_packet.header             = VENDOR_BIT | BARRIER_BIT;
    packets.start_packet.completion_signal = hsa_signal_t{.handle = 0};
    packets.stop_packet.completion_signal  = hsa_signal_t{.handle = 0};

    status = sym.spm_query_fn(aql_desc, AQLPROFILE_SPM_DECODE_QUERY_SEG_SIZE, &desc.seg_size);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to query SPM seg_size";
    status = sym.spm_query_fn(aql_desc, AQLPROFILE_SPM_DECODE_QUERY_NUM_XCC, &desc.buffer_num);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to query SPM buffer_num";

    is_valid = true;
}

void
SPMPacket::populate_before()
{
    hsa_barrier_and_packet_t barrier{};
    barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.header |= BARRIER_BIT;

    before_krn_pkt.push_back(barrier);
    before_krn_pkt.push_back(barrier);
    before_krn_pkt.push_back(packets.start_packet);
};

void
SPMPacket::populate_after()
{
    after_krn_pkt.push_back(packets.stop_packet);
};

void
SPMPacket::kfd_start()
{
    ROCP_FATAL_IF(!handle.handle) << "Attempt at starting SPM with unitialized packet!";

    if(running.exchange(true))
    {
        ROCP_ERROR << "Double call to KFD start!";
        return;
    }

    auto status = sym.spm_start_fn(this->handle, &SPMPacket::aql_data_callback, this);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Unable to acquire KFD thread";
}

void
SPMPacket::kfd_stop()
{
    if(running.exchange(false))
        sym.spm_stop_fn(this->handle);
    else
        ROCP_WARNING << "Double call to KFD stop!";

    ROCP_FATAL_IF(!data_fn) << "data_fn null";
    data_fn(agent_id, ROCPROFILER_SPM_RECORD_TYPE_DISPATCH_END, nullptr, user_data);
}

void
SPMPacket::aql_data_callback(aqlprofile_spm_buffer_handle_t handle,
                             void*                          data,
                             size_t                         size,
                             int /* flags */,
                             void* userdata)
{
    ROCP_ERROR_IF(!userdata) << "SPM callback passed null!";
    auto& pkt = *static_cast<SPMPacket*>(userdata);

    rocprofiler_spm_data_record_t record{};
    record.buffer_id = handle;
    record.data      = data;
    record.data_size = size;
    pkt.data_fn(pkt.agent_id, ROCPROFILER_SPM_RECORD_TYPE_DATA, &record, pkt.user_data);
}

SPMPacket::~SPMPacket()
{
    if(running.exchange(false) && sym.valid()) sym.spm_stop_fn(this->handle);
}

}  // namespace hsa
}  // namespace rocprofiler
