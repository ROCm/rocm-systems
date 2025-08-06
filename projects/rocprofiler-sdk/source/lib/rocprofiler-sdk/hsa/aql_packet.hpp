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

#pragma once

#include "lib/common/container/small_vector.hpp"
#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"
#include "lib/rocprofiler-sdk/spm/spm_decode.hpp"
#include "lib/rocprofiler-sdk/spm/spm_dlsym.hpp"

#include <rocprofiler-sdk/experimental/spm/core.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <atomic>

namespace rocprofiler
{
namespace aql
{
class CounterPacketConstruct;
class ThreadTraceAQLPacketFactory;
}  // namespace aql

namespace hsa
{
#define HSA_AMD_INTERFACE_VERSION                                                                  \
    ROCPROFILER_COMPUTE_VERSION(HSA_AMD_INTERFACE_VERSION_MAJOR, HSA_AMD_INTERFACE_VERSION_MINOR, 0)

#if HSA_AMD_INTERFACE_VERSION >= ROCPROFILER_COMPUTE_VERSION(1, 7, 0)
constexpr auto hsa_amd_memory_pool_executable_flag = HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG;
#else
constexpr auto hsa_amd_memory_pool_executable_flag = (1 << 2);
#endif

constexpr hsa_ext_amd_aql_pm4_packet_t null_amd_aql_pm4_packet = {
    .header            = 0,
    .pm4_command       = {0},
    .completion_signal = {.handle = 0}};

struct AQLMemoryPool
{
    using desc_t    = aqlprofile_buffer_desc_flags_t;
    using copy_fn_t = decltype(hsa_memory_copy);

    AQLMemoryPool(const class AgentCache& agent, const class AmdExtTable& ext, copy_fn_t copy_fn);
    explicit AQLMemoryPool() = default;
    virtual ~AQLMemoryPool() = default;

    hsa_agent_t                             gpu_agent{};
    hsa_amd_memory_pool_t                   cpu_pool_{};
    hsa_amd_memory_pool_t                   gpu_pool_{};
    hsa_amd_memory_pool_t                   kernarg_pool_{};
    decltype(hsa_amd_memory_pool_allocate)* allocate_fn{};
    decltype(hsa_amd_agents_allow_access)*  allow_access_fn{};
    decltype(hsa_amd_memory_pool_free)*     free_fn{};
    decltype(hsa_memory_copy)*              api_copy_fn{};
    decltype(hsa_amd_memory_fill)*          fill_fn{};

    // Different implementation may choose their settings for alloc
    virtual hsa_status_t Alloc(void** ptr, size_t size, desc_t flags) = 0;

    static hsa_status_t Alloc(void** ptr, size_t size, desc_t flags, void* data);
    static void         Free(void* ptr, void* data);
    static hsa_status_t Copy(void* dst, const void* src, size_t size, void* data);
};

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

    void clear()
    {
        before_krn_pkt.clear();
        after_krn_pkt.clear();
    }
    bool isEmpty() const { return empty; }

    virtual void populate_before() = 0;
    virtual void populate_after()  = 0;

    hsa_agent_t         GetAgent() const { return pool ? pool->gpu_agent : hsa_agent_t{}; }
    aqlprofile_handle_t GetHandle() const { return handle; }
    aqlprofile_handle_t handle{.handle = 0};
    bool                empty{true};

    common::container::small_vector<hsa_ext_amd_aql_pm4_packet_t, 3> before_krn_pkt = {};
    common::container::small_vector<hsa_ext_amd_aql_pm4_packet_t, 2> after_krn_pkt  = {};

    std::shared_ptr<AQLMemoryPool> pool{};
};

class EmptyAQLPacket : public AQLPacket
{
public:
    EmptyAQLPacket()           = default;
    ~EmptyAQLPacket() override = default;

    void populate_before() override{};
    void populate_after() override{};
};

class CounterAQLPacket : public AQLPacket
{
    friend class rocprofiler::aql::CounterPacketConstruct;
    using memory_pool_free_func_t = decltype(::hsa_amd_memory_pool_free)*;

    struct CounterMemoryPool : public AQLMemoryPool
    {
        CounterMemoryPool(const class AgentCache&  agent,
                          const class AmdExtTable& ext,
                          copy_fn_t                copy_fn)
        : AQLMemoryPool(agent, ext, copy_fn){};
        bool         bIgnoreKernArg{false};
        hsa_status_t Alloc(void** ptr, size_t size, desc_t flags) override;
    };

public:
    CounterAQLPacket(aqlprofile_agent_handle_t                  agent,
                     CounterMemoryPool                          pool,
                     const std::vector<aqlprofile_pmc_event_t>& events);
    ~CounterAQLPacket() override { aqlprofile_pmc_delete_packets(this->handle); };

    void populate_before() override
    {
        if(!empty) before_krn_pkt.push_back(packets.start_packet);
    };
    void populate_after() override
    {
        if(empty) return;
        after_krn_pkt.push_back(packets.read_packet);
        after_krn_pkt.push_back(packets.stop_packet);
    };

    aqlprofile_pmc_aql_packets_t packets{};
};

struct TraceMemoryPool : public AQLMemoryPool
{
    TraceMemoryPool(const class AgentCache& agent, const class AmdExtTable& ext, copy_fn_t copy_fn)
    : AQLMemoryPool(agent, ext, copy_fn){};

    aqlprofile_handle_t handle = {.handle = 0};
    hsa_status_t        Alloc(void** ptr, size_t size, desc_t flags) override;
    ~TraceMemoryPool() override
    {
        if(handle.handle) aqlprofile_att_delete_packets(this->handle);
    };
};

class CodeobjMarkerAQLPacket : public AQLPacket
{
    friend class rocprofiler::aql::ThreadTraceAQLPacketFactory;

public:
    CodeobjMarkerAQLPacket(const TraceMemoryPool& _pool,
                           uint64_t               id,
                           uint64_t               addr,
                           uint64_t               size,
                           bool                   bFromStart,
                           bool                   bIsUnload);

    void populate_before() override { before_krn_pkt.push_back(packet); };
    void populate_after() override{};

    hsa_ext_amd_aql_pm4_packet_t packet;
};

class TraceControlAQLPacket : public AQLPacket
{
    friend class rocprofiler::aql::ThreadTraceAQLPacketFactory;
    using code_object_id_t = uint64_t;

public:
    TraceControlAQLPacket(const TraceMemoryPool&          tracepool,
                          const aqlprofile_att_profile_t& profile);
    ~TraceControlAQLPacket() override = default;

    explicit TraceControlAQLPacket(const TraceControlAQLPacket& other)
    : AQLPacket()
    {
        this->pool           = other.pool;
        this->packets        = other.packets;
        this->loaded_codeobj = other.loaded_codeobj;
        this->handle         = other.handle;
        this->empty          = other.empty;
    }

    hsa_agent_t GetAgent() const { return pool->gpu_agent; }

    void populate_before() override
    {
        before_krn_pkt.push_back(packets.start_packet);
        for(auto& [_, codeobj] : loaded_codeobj)
            before_krn_pkt.push_back(codeobj->packet);
    }
    void populate_after() override { after_krn_pkt.push_back(packets.stop_packet); }

    void add_codeobj(code_object_id_t id, uint64_t addr, uint64_t size)
    {
        loaded_codeobj[id] =
            std::make_shared<CodeobjMarkerAQLPacket>(*tracepool, id, addr, size, true, false);
    }
    bool remove_codeobj(code_object_id_t id) { return loaded_codeobj.erase(id) != 0; }

protected:
    std::shared_ptr<TraceMemoryPool>                                              tracepool{};
    aqlprofile_att_control_aql_packets_t                                          packets;
    std::unordered_map<code_object_id_t, std::shared_ptr<CodeobjMarkerAQLPacket>> loaded_codeobj;
};

struct SPMMemoryPool : public AQLMemoryPool
{
    SPMMemoryPool(const class AgentCache& agent, const class AmdExtTable& ext, copy_fn_t copy_fn)
    : AQLMemoryPool(agent, ext, copy_fn){};

    explicit SPMMemoryPool() = default;
    hsa_status_t Alloc(void** ptr, size_t size, desc_t flags) override;
};

class SPMPacket : public AQLPacket
{
public:
    SPMPacket(std::shared_ptr<SPMMemoryPool>  _pool,
              const aqlprofile_spm_profile_t& profile,
              rocprofiler_agent_id_t          agent_id);
    ~SPMPacket() override;

    void kfd_start(rocprofiler_spm_data_callback_t fn, rocprofiler_user_data_t userdata);
    void kfd_stop();
    bool Valid() const { return is_valid; }

    const rocprofiler_agent_id_t agent_id;
    rocprofiler_user_data_t      user_data{};
    aqlprofile_spm_buffer_desc_t aql_desc{};
    // build by packet_construt
    rocprofiler_spm_descriptor_t desc{};
    std::vector<char>            container_desc_data{};

    void populate_before() override { before_krn_pkt.push_back(packets.start_packet); };
    void populate_after() override;

private:
    static void aql_data_callback(aqlprofile_spm_buffer_handle_t, void*, size_t, int, void*);

    aqlprofile_spm_aql_packets_t    packets{};
    rocprofiler_spm_data_callback_t data_fn{};

    std::atomic<bool> running{false};
    bool              is_valid{false};

    const SPM::Dlsym sym;
};

}  // namespace hsa
}  // namespace rocprofiler
