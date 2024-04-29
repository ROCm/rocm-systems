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

#pragma once

#include "lib/common/container/small_vector.hpp"
#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"

#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

namespace rocprofiler
{
namespace aql
{
class CounterPacketConstruct;
class ThreadTraceAQLPacketFactory;
}  // namespace aql

namespace hsa
{
constexpr hsa_ext_amd_aql_pm4_packet_t null_amd_aql_pm4_packet = {
    .header            = 0,
    .pm4_command       = {0},
    .completion_signal = {.handle = 0}};

/**
 * Struct containing AQL packet information. Including start/stop/read
 * packets along with allocated buffers
 */
class AQLPacket
{
public:
    AQLPacket()          = default;
    virtual ~AQLPacket() = default;

    // Keep move constuctors (i.e. std::move())
    AQLPacket(AQLPacket&& other) = default;
    AQLPacket& operator=(AQLPacket&& other) = default;

    // Do not allow copying this class
    AQLPacket(const AQLPacket&) = delete;
    AQLPacket& operator=(const AQLPacket&) = delete;

    aqlprofile_handle_t          pkt_handle = {.handle = 0};
    aqlprofile_pmc_aql_packets_t pkts       = {.start_packet = null_amd_aql_pm4_packet,
                                         .stop_packet  = null_amd_aql_pm4_packet,
                                         .read_packet  = null_amd_aql_pm4_packet};

    bool                             empty   = {true};
    hsa_ven_amd_aqlprofile_profile_t profile = {};
    hsa_ext_amd_aql_pm4_packet_t     start   = null_amd_aql_pm4_packet;
    hsa_ext_amd_aql_pm4_packet_t     stop    = null_amd_aql_pm4_packet;
    hsa_ext_amd_aql_pm4_packet_t     read    = null_amd_aql_pm4_packet;
    common::container::small_vector<hsa_ext_amd_aql_pm4_packet_t, 3> before_krn_pkt = {};
    common::container::small_vector<hsa_ext_amd_aql_pm4_packet_t, 2> after_krn_pkt  = {};

    bool isEmpty() const { return empty; }
};

class CounterAQLPacket : public AQLPacket
{
    friend class rocprofiler::aql::CounterPacketConstruct;
    using memory_pool_free_func_t = decltype(::hsa_amd_memory_pool_free)*;

public:
    CounterAQLPacket(memory_pool_free_func_t func)
    : free_func{func} {};
    ~CounterAQLPacket() override;

protected:
    bool                    command_buf_mallocd    = false;
    bool                    output_buffer_malloced = false;
    memory_pool_free_func_t free_func              = nullptr;
};

struct TraceMemoryPool
{
    hsa_agent_t                             gpu_agent;
    hsa_amd_memory_pool_t                   cpu_pool_;
    hsa_amd_memory_pool_t                   gpu_pool_;
    decltype(hsa_amd_memory_pool_allocate)* allocate_fn;
    decltype(hsa_amd_agents_allow_access)*  allow_access_fn;
    decltype(hsa_amd_memory_pool_free)*     free_fn;
    decltype(hsa_memory_copy)*              api_copy_fn;
};

class TraceAQLPacket : public AQLPacket
{
    friend class rocprofiler::aql::ThreadTraceAQLPacketFactory;

public:
    TraceAQLPacket(std::shared_ptr<TraceMemoryPool>& _tracepool);
    TraceMemoryPool&    GetPool() const { return *tracepool; }
    aqlprofile_handle_t GetHandle() const { return handle; }
    uint64_t            GetAgent() const { return tracepool->gpu_agent.handle; }
    ~TraceAQLPacket() override;

protected:
    std::shared_ptr<TraceMemoryPool>     tracepool;
    aqlprofile_att_control_aql_packets_t packets;
    aqlprofile_handle_t                  handle;

    static hsa_status_t Alloc(void**                         ptr,
                              size_t                         size,
                              aqlprofile_buffer_desc_flags_t flags,
                              void*                          data);
    static void         Free(void* ptr, void* data);
    static hsa_status_t Copy(void* dst, const void* src, size_t size, void* data);
};

}  // namespace hsa
}  // namespace rocprofiler
