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

#include "lib/aqlprofile/aqlprofile.hpp"
#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"

#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <atomic>
#include <cstddef>
#include <optional>

namespace rocprofiler
{
namespace spm
{
/**
 * @brief hsa::SPMPacket contains spm_descriptor_t
 * aqlprofile_spm_buffer_desc_t is returned from aqlprofile
 * aqlprofile_spm_buffer_desc_t has the metadata needed to decode the packet
 * Data contains spm_desc_v0_t + event_map + data in aqlprofile_spm_buffer_desc_t
 * size = sizeof(spm_desc_v0_t) + sizeof(spm_counter_instance_t)*num_events + aql_desc.size
 * seg_size = output segment size
 * buffer_num = number of XCCs
 */
typedef struct spm_descriptor_t
{
    void*  data{nullptr};
    size_t size{0};
    size_t seg_size{0};
    size_t buffer_num{0};
} spm_descriptor_t;
struct spm_interface;
}  // namespace spm

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

/**
 * Struct containing AQL packet information. Including start/stop/read
 * packets along with allocated buffers
 */
class AQLPacket
{
public:
    AQLPacket()          = default;
    virtual ~AQLPacket() = default;

    // Keep move constructors (i.e. std::move())
    AQLPacket(AQLPacket&& other) = default;
    AQLPacket& operator=(AQLPacket&& other) = default;

    // Do not allow copying this class
    AQLPacket(const AQLPacket&) = delete;
    AQLPacket& operator=(const AQLPacket&) = delete;

    void clear()
    {
        before_krn_pkt.clear();
        after_krn_pkt.clear();
        before_krn_barrier_pkt.clear();
    }
    bool isEmpty() const { return empty; }

    virtual void populate_before() = 0;
    virtual void populate_after()  = 0;

    aqlprofile_handle_t GetHandle() const { return handle; }
    aqlprofile_handle_t handle = {.handle = 0};
    bool                empty  = {true};

    common::container::small_vector<hsa_ext_amd_aql_pm4_packet_t, 3> before_krn_pkt         = {};
    common::container::small_vector<hsa_ext_amd_aql_pm4_packet_t, 2> after_krn_pkt          = {};
    common::container::small_vector<hsa_barrier_and_packet_t, 2>     before_krn_barrier_pkt = {};
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

    struct CounterMemoryPool
    {
        using desc_t = aqlprofile_buffer_desc_flags_t;

        hsa_agent_t                             gpu_agent       = {.handle = 0};
        hsa_amd_memory_pool_t                   cpu_pool_       = {.handle = 0};
        hsa_amd_memory_pool_t                   kernarg_pool_   = {.handle = 0};
        decltype(hsa_amd_memory_pool_allocate)* allocate_fn     = nullptr;
        decltype(hsa_amd_agents_allow_access)*  allow_access_fn = nullptr;
        decltype(hsa_amd_memory_pool_free)*     free_fn         = nullptr;
        decltype(hsa_amd_memory_fill)*          fill_fn         = nullptr;
        decltype(hsa_memory_copy)*              api_copy_fn     = nullptr;
        bool                                    bIgnoreKernArg  = false;

        static void         Free(void* ptr, void* data);
        static hsa_status_t Alloc(void** ptr, size_t size, desc_t flags, void* data);
        static hsa_status_t Copy(void* dst, const void* src, size_t size, void* data);
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

protected:
    CounterMemoryPool pool{};
};

struct TraceMemoryPool
{
    using desc_t = aqlprofile_buffer_desc_flags_t;

    hsa_agent_t                             gpu_agent;
    hsa_amd_memory_pool_t                   cpu_pool_;
    hsa_amd_memory_pool_t                   gpu_pool_;
    decltype(hsa_amd_memory_pool_allocate)* allocate_fn;
    decltype(hsa_amd_agents_allow_access)*  allow_access_fn;
    decltype(hsa_amd_memory_pool_free)*     free_fn;
    decltype(hsa_memory_copy)*              api_copy_fn;

    aqlprofile_handle_t handle;
    ~TraceMemoryPool() { aqlprofile_att_delete_packets(this->handle); };

    static hsa_status_t Alloc(void** ptr, size_t size, desc_t flags, void* data);
    static void         Free(void* ptr, void* data);
    static hsa_status_t Copy(void* dst, const void* src, size_t size, void* data);
};

class CodeobjMarkerAQLPacket : public AQLPacket
{
    friend class rocprofiler::aql::ThreadTraceAQLPacketFactory;

public:
    CodeobjMarkerAQLPacket(const TraceMemoryPool& tracepool,
                           uint64_t               id,
                           uint64_t               addr,
                           uint64_t               size,
                           bool                   bFromStart,
                           bool                   bIsUnload);
    ~CodeobjMarkerAQLPacket() override = default;

    void populate_before() override { before_krn_pkt.push_back(packet); };
    void populate_after() override{};

    aqlprofile_handle_t GetHandle() const { return tracepool.handle; }
    hsa_agent_t         GetAgent() const { return tracepool.gpu_agent; }

    hsa_ext_amd_aql_pm4_packet_t packet;

protected:
    TraceMemoryPool tracepool;
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
        this->tracepool      = other.tracepool;
        this->packets        = other.packets;
        this->loaded_codeobj = other.loaded_codeobj;
    }

    aqlprofile_handle_t GetHandle() const { return tracepool->handle; }
    hsa_agent_t         GetAgent() const { return tracepool->gpu_agent; }

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
    std::shared_ptr<TraceMemoryPool>     tracepool;
    aqlprofile_att_control_aql_packets_t packets;

    std::unordered_map<code_object_id_t, std::shared_ptr<CodeobjMarkerAQLPacket>> loaded_codeobj;
};

struct sqtt_buffer_status_t
{
    void*                        data{};
    uint64_t                     size{};
    hsa_ext_amd_aql_pm4_packet_t packet{};
    bool                         gpu_full{};
};

// Virtual members for mocking in tests
class SQTTBufferingPackets
{
public:
    SQTTBufferingPackets(aqlprofile_handle_t handle, int shader_engine_id);
    virtual ~SQTTBufferingPackets() = default;

    hsa_ext_amd_aql_pm4_packet_t                query_status{};
    virtual std::optional<sqtt_buffer_status_t> query_buffer_status();

    void reset_current_buffer() { current_buffer = 0; };

    const aqlprofile_handle_t handle;
    const int                 shader_engine_id;
    uint64_t                  header{0};

private:
    size_t                                    current_buffer{0};
    std::vector<hsa_ext_amd_aql_pm4_packet_t> buffer_swap{};
};
struct SPMMemoryPool
{
    using desc_t                                            = aqlprofile_buffer_desc_flags_t;
    using copy_fn_t                                         = decltype(hsa_memory_copy);
    hsa_agent_t                             gpu_agent       = {.handle = 0};
    hsa_amd_memory_pool_t                   cpu_pool_       = {.handle = 0};
    hsa_amd_memory_pool_t                   gpu_pool_       = {.handle = 0};
    hsa_amd_memory_pool_t                   kernarg_pool_   = {.handle = 0};
    decltype(hsa_amd_memory_pool_allocate)* allocate_fn     = nullptr;
    decltype(hsa_amd_agents_allow_access)*  allow_access_fn = nullptr;
    decltype(hsa_amd_memory_pool_free)*     free_fn         = nullptr;
    decltype(hsa_memory_copy)*              api_copy_fn     = nullptr;
    decltype(hsa_amd_memory_fill)*          fill_fn         = nullptr;

    SPMMemoryPool(const class AgentCache& agent, const class AmdExtTable& ext, copy_fn_t copy_fn);
    ~SPMMemoryPool()                    = default;
    SPMMemoryPool(const SPMMemoryPool&) = delete;
    SPMMemoryPool& operator=(const SPMMemoryPool&) = delete;
    explicit SPMMemoryPool()                       = default;
    static hsa_status_t Alloc(void**                         ptr,
                              size_t                         size,
                              aqlprofile_buffer_desc_flags_t flags,
                              void*                          data);
    static void         Free(void* ptr, void* data);
    static hsa_status_t Copy(void* dst, const void* src, size_t size, void* data);
};

struct SPMCallbackContext
{
    std::optional<rocprofiler_buffer_id_t>           buffer;
    rocprofiler_user_data_t                          user_data{};
    void*                                            record_callback_args{};
    rocprofiler_spm_dispatch_counting_record_cb_t    record_cb{};
    rocprofiler_spm_dispatch_counting_service_data_t dispatch_data{};
};

struct SPMProfileData
{
    aqlprofile_spm_buffer_desc_t            aql_desc{};
    rocprofiler::spm::spm_descriptor_t      spm_desc{};
    std::shared_ptr<std::vector<char>>      container_desc_data{};
    aqlprofile_spm_aql_packets_t            packets{};
    std::vector<aqlprofile_pmc_event_t>     aql_events{};
    std::vector<aqlprofile_spm_parameter_t> aql_params{};
};

class SPMPacket : public AQLPacket
{
public:
    SPMPacket(aqlprofile_agent_handle_t                 aql_agent,
              std::shared_ptr<SPMMemoryPool>            _pool,
              std::vector<aqlprofile_pmc_event_t>&&     events,
              std::vector<aqlprofile_spm_parameter_t>&& params);

    ~SPMPacket() override;
    SPMPacket& operator=(const SPMPacket&) = delete;
    SPMPacket(const SPMPacket&)            = delete;

    bool        kfd_start();
    bool        kfd_stop();
    hsa_agent_t GetAgent() const { return pool ? pool->gpu_agent : hsa_agent_t{}; }
    void        populate_before() override;
    void        populate_after() override;
    bool        valid() const { return is_valid; }

    void clear()
    {
        AQLPacket::clear();
        cb = {};
    }

    SPMCallbackContext             cb{};
    SPMProfileData                 profile{};
    std::shared_ptr<SPMMemoryPool> pool{};
    const spm::spm_interface*      sym = nullptr;

private:
    common::Synchronized<bool> running{false};
    bool                       is_valid{false};
};

}  // namespace hsa
}  // namespace rocprofiler
