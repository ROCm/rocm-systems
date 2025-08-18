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
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
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

    common::container::small_vector<rocprofiler_packet, 3> before_krn_pkt = {};
    common::container::small_vector<rocprofiler_packet, 2> after_krn_pkt  = {};

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

    hsa::rocprofiler_packet packet{};
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

    ~SPMMemoryPool() override
    {
        if(delete_packets_fn && handle.handle) delete_packets_fn(handle);
    };

    explicit SPMMemoryPool() = default;
    hsa_status_t Alloc(void** ptr, size_t size, desc_t flags) override;

    SPM::Dlsym::DeleteFn* delete_packets_fn{nullptr};
    aqlprofile_handle_t   handle{};
};

class SPMPacket : public AQLPacket
{
public:
    SPMPacket(const aqlprofile_spm_profile_t& profile, rocprofiler_agent_id_t agent_id);
    ~SPMPacket() override;

    explicit SPMPacket(const SPMPacket& other)
    : agent_id(other.agent_id)
    , sym(other.sym)
    {
        packets             = other.packets;
        is_valid            = other.is_valid;
        handle              = other.handle;
        empty               = other.empty;
        pool                = other.pool;
        aql_desc            = other.aql_desc;
        spm_desc            = other.spm_desc;
        container_desc_data = other.container_desc_data;
    }

    void kfd_start();
    void kfd_stop();
    bool Valid() const { return is_valid; }

    const rocprofiler_agent_id_t       agent_id;
    rocprofiler_spm_data_callback_t    decode_data_fn{};
    rocprofiler_user_data_t            user_data{};
    aqlprofile_spm_buffer_desc_t       aql_desc{};
    rocprofiler::SPM::spm_descriptor_t spm_desc{};

    std::shared_ptr<std::vector<char>> container_desc_data{};

    void populate_before() override;
    void populate_after() override;

    const SPM::Dlsym sym{};

private:
    aqlprofile_spm_aql_packets_t packets{};
    std::atomic<bool>            running{false};
    bool                         is_valid{false};
};

using ClientID = int64_t;
using inst_pkt_t =
    common::container::small_vector<std::pair<std::unique_ptr<AQLPacket>, ClientID>, 4>;
}  // namespace hsa
}  // namespace rocprofiler
