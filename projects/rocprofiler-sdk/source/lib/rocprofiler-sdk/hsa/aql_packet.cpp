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
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/spm/decode.hpp"
#include "lib/rocprofiler-sdk/spm/interface.hpp"
#include "lib/rocprofiler-sdk/thread_trace/dl.hpp"

#include <fmt/core.h>
#include <cstdlib>
#include <iostream>

#define CHECK_HSA(fn, message)                                                                     \
    {                                                                                              \
        auto _status = (fn);                                                                       \
        ROCP_FATAL_IF(_status != HSA_STATUS_SUCCESS) << message << ": " << _status;                \
    }

namespace rocprofiler
{
namespace hsa
{
constexpr uint16_t VENDOR_BIT  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
constexpr uint16_t BARRIER_BIT = 1 << HSA_PACKET_HEADER_BARRIER;

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
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<TraceMemoryPool*>(data);

    if(!pool.allocate_fn || !pool.free_fn || !pool.allow_access_fn) return HSA_STATUS_ERROR;

    hsa_status_t status = HSA_STATUS_ERROR;
    if(flags.host_access)
    {
        status = pool.allocate_fn(pool.cpu_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);

        if(status == HSA_STATUS_SUCCESS)
            status = pool.allow_access_fn(1, &pool.gpu_agent, nullptr, *ptr);
    }
    else
    {
        // Return page aligned data to avoid cache flush overlap
        status = pool.allocate_fn(
            pool.gpu_pool_, size + 0x2000, hsa_amd_memory_pool_executable_flag, ptr);
        *ptr = (void*) ((uintptr_t(*ptr) + 0xFFF) & ~0xFFFul);  // NOLINT(performance-no-int-to-ptr)
    }
    return status;
}

void
TraceMemoryPool::Free(void* ptr, void* data)
{
    assert(data);
    auto& pool = *reinterpret_cast<TraceMemoryPool*>(data);

    if(pool.free_fn) pool.free_fn(ptr);
}

hsa_status_t
TraceMemoryPool::Copy(void* dst, const void* src, size_t size, void* data)
{
    if(!data) return HSA_STATUS_ERROR;
    auto& pool = *reinterpret_cast<TraceMemoryPool*>(data);

    if(!pool.api_copy_fn) return HSA_STATUS_ERROR;

    return pool.api_copy_fn(dst, src, size);
}

TraceControlAQLPacket::TraceControlAQLPacket(const TraceMemoryPool&          _tracepool,
                                             const aqlprofile_att_profile_t& p)
: tracepool(std::make_shared<TraceMemoryPool>(_tracepool))
{
    auto status = aqlprofile_att_create_packets(&tracepool->handle,
                                                &packets,
                                                p,
                                                &TraceMemoryPool::Alloc,
                                                &TraceMemoryPool::Free,
                                                &TraceMemoryPool::Copy,
                                                tracepool.get());
    CHECK_HSA(status, "failed to create ATT packet");

    packets.start_packet.header            = VENDOR_BIT | BARRIER_BIT;
    packets.stop_packet.header             = VENDOR_BIT | BARRIER_BIT;
    packets.start_packet.completion_signal = hsa_signal_t{.handle = 0};
    packets.stop_packet.completion_signal  = hsa_signal_t{.handle = 0};
    this->empty                            = false;

    clear();
};

SQTTBufferingPackets::SQTTBufferingPackets(aqlprofile_handle_t _handle, int _shader_engine_id)
: handle(_handle)
, shader_engine_id(_shader_engine_id)
{
    auto* aqlprofile_dl = rocprofiler::thread_trace::get_aqlprofile_dl();
    if(!aqlprofile_dl || !aqlprofile_dl->valid())
    {
        ROCP_FATAL << "AQLProfile dynamic library not loaded or missing required symbols. "
                   << "Cannot create SQTT buffering packets.";
    }

    // We sometimes need 2x the number of packets as there are buffers.
    uint64_t num_packets{6};
    buffer_swap.resize(num_packets);

    auto buffer_ptr = std::vector<hsa_ext_amd_aql_pm4_packet_t*>{};
    for(auto& buffer : buffer_swap)
        buffer_ptr.emplace_back(&buffer);

    auto status = aqlprofile_dl->get_buffer_packets_fn(
        &header, &query_status, buffer_ptr.data(), &num_packets, handle, shader_engine_id, 0);
    CHECK_HSA(status, "failed to create ATT double buffer packet");

    buffer_swap.resize(num_packets);  // Discard unused packets

    query_status.header = VENDOR_BIT | BARRIER_BIT;
    for(auto& buffer : buffer_swap)
        buffer.header = VENDOR_BIT | BARRIER_BIT;
}

std::optional<sqtt_buffer_status_t>
SQTTBufferingPackets::query_buffer_status()
{
    auto* aqlprofile_dl = rocprofiler::thread_trace::get_aqlprofile_dl();
    ROCP_FATAL_IF(!aqlprofile_dl || !aqlprofile_dl->valid())
        << "AQLProfile dynamic library not valid. Cannot query buffer status.";

    auto ret = aqlprofile_att_buffer_status_t{};

    auto status = aqlprofile_dl->update_buffer_status_fn(&ret, handle, shader_engine_id, 0);
    CHECK_HSA(status, "failed to query ATT status");

    if(!ret.needs_swap) return {};

    // Ensure aqlprofile and SDK agrees on which is the current buffer
    ROCP_CI_LOG_IF(ERROR, (current_buffer++) != ret.num_swaps)
        << "Mismatch of AQL and SDK buffer states!";

    auto query     = sqtt_buffer_status_t{};
    query.data     = ret.data;
    query.size     = ret.read_size;
    query.gpu_full = ret.is_too_late;
    query.packet   = buffer_swap.at(ret.num_swaps % buffer_swap.size());

    return query;
}

CodeobjMarkerAQLPacket::CodeobjMarkerAQLPacket(const TraceMemoryPool& _tracepool,
                                               uint64_t               id,
                                               uint64_t               addr,
                                               uint64_t               size,
                                               bool                   bFromStart,
                                               bool                   bIsUnload)
: tracepool(_tracepool)
{
    aqlprofile_att_codeobj_data_t codeobj{};
    codeobj.id        = id;
    codeobj.addr      = addr;
    codeobj.size      = size;
    codeobj.agent     = tracepool.gpu_agent;
    codeobj.isUnload  = bIsUnload;
    codeobj.fromStart = bFromStart;

    auto status = aqlprofile_att_codeobj_marker(&packet,
                                                &tracepool.handle,
                                                codeobj,
                                                &TraceMemoryPool::Alloc,
                                                &TraceMemoryPool::Free,
                                                &tracepool);
    CHECK_HSA(status, "failed to create ATT marker");

    packet.header            = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    packet.completion_signal = hsa_signal_t{.handle = 0};
    this->empty              = false;

    clear();
}

SPMMemoryPool::SPMMemoryPool(const AgentCache& agent, const AmdExtTable& ext, copy_fn_t copy_fn)
{
    allocate_fn     = ext.hsa_amd_memory_pool_allocate_fn;
    allow_access_fn = ext.hsa_amd_agents_allow_access_fn;
    free_fn         = ext.hsa_amd_memory_pool_free_fn;
    fill_fn         = ext.hsa_amd_memory_fill_fn;
    api_copy_fn     = copy_fn;

    gpu_agent = agent.get_hsa_agent();
    cpu_pool_ = agent.cpu_pool();
}

void
SPMMemoryPool::Free(void* ptr, void* data)
{
    if(ptr == nullptr) return;
    auto* pool = reinterpret_cast<SPMMemoryPool*>(data);

    ROCP_FATAL_IF(!pool || !pool->free_fn) << "Unable to deallocate from HSA memory pool";
    pool->free_fn(ptr);
}

hsa_status_t
SPMMemoryPool::Copy(void* dst, const void* src, size_t size, void* data)
{
    if(size == 0) return HSA_STATUS_SUCCESS;
    auto* pool = reinterpret_cast<SPMMemoryPool*>(data);
    ROCP_FATAL_IF(!pool || !pool->api_copy_fn) << "Unable to copy HSA memory";

    return pool->api_copy_fn(dst, src, size);
}

hsa_status_t
SPMMemoryPool::Alloc(void** ptr, size_t size, aqlprofile_buffer_desc_flags_t flags, void* data)
{
    hsa_status_t status = HSA_STATUS_ERROR;

    if(size == 0)
    {
        if(ptr != nullptr) *ptr = nullptr;
        return HSA_STATUS_SUCCESS;
    }

    if(!ptr) return HSA_STATUS_ERROR;
    if(!data) return HSA_STATUS_ERROR;

    auto& pool = *reinterpret_cast<SPMMemoryPool*>(data);
    if(!flags.host_access || !pool.allocate_fn || !pool.free_fn || !pool.allow_access_fn ||
       !pool.fill_fn)
        return HSA_STATUS_ERROR;

    status = pool.allocate_fn(pool.cpu_pool_, size, hsa_amd_memory_pool_executable_flag, ptr);
    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_FATAL << "Could not allocate memory";
        return status;
    }

    return status;
}

SPMPacket::SPMPacket(aqlprofile_agent_handle_t                 aql_agent,
                     std::shared_ptr<SPMMemoryPool>            _pool,
                     std::vector<aqlprofile_pmc_event_t>&&     events,
                     std::vector<aqlprofile_spm_parameter_t>&& params)
: pool(std::move(_pool))
{
    profile.aql_events = std::move(events);
    profile.aql_params = std::move(params);

    sym = rocprofiler::spm::construct_spm_interface();
    if(!sym)
    {
        ROCP_ERROR << "Failed to construct SPM interface";
        return;
    }
    aqlprofile_spm_profile_t aql_profile{.aql_agent       = aql_agent,
                                         .hsa_agent       = pool->gpu_agent,
                                         .events          = profile.aql_events.data(),
                                         .event_count     = profile.aql_events.size(),
                                         .parameters      = profile.aql_params.data(),
                                         .parameter_count = profile.aql_params.size(),
                                         .reserved        = 0,
                                         .alloc_cb        = &(hsa::SPMMemoryPool::Alloc),
                                         .dealloc_cb      = &(hsa::SPMMemoryPool::Free),
                                         .memcpy_cb       = &(hsa::SPMMemoryPool::Copy),
                                         .userdata        = pool.get()};

    auto status =
        sym->spm_create_packets(&handle, &profile.aql_desc, &profile.packets, aql_profile, 0);
    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << "spm_create_packets failed with HSA status: " << status
                   << " (event_count=" << profile.aql_events.size() << ")";
        return;
    }
    profile.packets.start_packet.header            = VENDOR_BIT | BARRIER_BIT;
    profile.packets.stop_packet.header             = VENDOR_BIT | BARRIER_BIT;
    profile.packets.start_packet.completion_signal = hsa_signal_t{.handle = 0};
    profile.packets.stop_packet.completion_signal  = hsa_signal_t{.handle = 0};

    status = sym->spm_decode_query(
        profile.aql_desc, AQLPROFILE_SPM_DECODE_QUERY_SEG_SIZE, &profile.spm_desc.seg_size);
    if(status != HSA_STATUS_SUCCESS) return;
    status = sym->spm_decode_query(
        profile.aql_desc, AQLPROFILE_SPM_DECODE_QUERY_NUM_XCC, &profile.spm_desc.buffer_num);
    if(status != HSA_STATUS_SUCCESS) return;

    is_valid = true;
    empty    = false;
}

void
SPMPacket::populate_before()
{
    hsa_barrier_and_packet_t barrier{};
    barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.header |= BARRIER_BIT;

    before_krn_barrier_pkt.push_back(barrier);
    before_krn_barrier_pkt.push_back(barrier);
    before_krn_pkt.push_back(profile.packets.start_packet);
};

void
SPMPacket::populate_after()
{
    after_krn_pkt.push_back(profile.packets.stop_packet);
};

bool
SPMPacket::kfd_start()
{
    ROCP_FATAL_IF(!handle.handle) << "Attempt at starting SPM with uninitialized packet!";

    bool success = false;
    running.wlock([&](auto& _running) {
        if(_running == true)
        {
            ROCP_WARNING << "Double call to KFD start!";
            success = true;
            return;
        }
        auto status = sym->spm_start(this->handle, spm::aql_data_callback, this);
        ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Unable to acquire KFD thread: " << status;
        _running = true;
        success  = true;
    });
    return success;
}

bool
SPMPacket::kfd_stop()
{
    bool success = false;
    running.wlock([&](auto& _running) {
        if(_running == false)
        {
            ROCP_WARNING << "Double call to KFD stop!";
            success = true;
            return;
        }
        auto status = sym->spm_stop(this->handle);
        ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS)
            << "spm_stop failed with HSA status: " << status;
        _running = false;
        success  = true;
    });
    return success;
}

SPMPacket::~SPMPacket()
{
    running.wlock([&](auto& _running) {
        if(_running == false) return;
        auto status = sym->spm_stop(this->handle);
        ROCP_WARNING_IF(status != HSA_STATUS_SUCCESS)
            << "spm_stop failed with HSA status: " << status;
        _running = false;
    });
}
}  // namespace hsa
}  // namespace rocprofiler
