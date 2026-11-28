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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/aqlprofile/core/memorymanager.hpp"
#include "lib/aqlprofile/core/pm4_factory.h"
#include "lib/common/scope_destructor.hpp"
#include "gtest/gtest.h"
#include <cstring>
#include <set>

// Dummy alloc/dealloc functions for testing
hsa_status_t
dummy_alloc(void** ptr, size_t size, aqlprofile_buffer_desc_flags_t, void*)
{
    *ptr = malloc(size);
    return *ptr ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}
void
dummy_dealloc(void* ptr, void*)
{
    free(ptr);
}
hsa_status_t
dummy_alloc_fail(void**, size_t, aqlprofile_buffer_desc_flags_t, void*)
{
    return HSA_STATUS_ERROR;
}
hsa_status_t
dummy_copy(void* dst, const void* src, size_t size, void*)
{
    memcpy(dst, src, size);
    return HSA_STATUS_SUCCESS;
}

// Helper subclass to expose protected AllocMemory for testing
class TestCounterMemoryManager : public CounterMemoryManager
{
public:
    using CounterMemoryManager::AllocMemory;
    using CounterMemoryManager::CounterMemoryManager;
};

TEST(CounterMemoryManagerTest, AllocMemory_Success)
{
    hsa_agent_t                    agent = {.handle = 1};
    TestCounterMemoryManager       mgr(agent, dummy_alloc, dummy_dealloc, nullptr);
    aqlprofile_buffer_desc_flags_t flags{};
    auto                           mem = mgr.AllocMemory(64, flags);
    ASSERT_NE(mem.get(), nullptr);
}

TEST(CounterMemoryManagerTest, AllocMemory_FailureThrows)
{
    hsa_agent_t                    agent = {.handle = 1};
    TestCounterMemoryManager       mgr(agent, dummy_alloc_fail, dummy_dealloc, nullptr);
    aqlprofile_buffer_desc_flags_t flags{};
    EXPECT_THROW(mgr.AllocMemory(64, flags), hsa_status_t);
}

TEST(CounterMemoryManagerTest, CmdBufAndOutputBuf)
{
    hsa_agent_t          agent = {.handle = 1};
    CounterMemoryManager mgr(agent, dummy_alloc, dummy_dealloc, nullptr);
    mgr.CreateCmdBuf(128);
    ASSERT_NE(mgr.GetCmdBuf(), nullptr);
    mgr.CreateOutputBuf(256);
    ASSERT_NE(mgr.GetOutputBuf(), nullptr);
    ASSERT_EQ(mgr.GetOutputBufSize(), 256u);
}

TEST(CounterMemoryManagerTest, RegisterAndGetManager)
{
    hsa_agent_t agent = {.handle = 1};
    auto   mgr = std::make_shared<CounterMemoryManager>(agent, dummy_alloc, dummy_dealloc, nullptr);
    size_t handle = mgr->GetHandler();
    CounterMemoryManager::RegisterManager(mgr);
    auto found = CounterMemoryManager::GetManager(handle);
    ASSERT_EQ(found.get(), mgr.get());
    CounterMemoryManager::DeleteManager(handle);
    ASSERT_EQ(CounterMemoryManager::GetManager(handle), nullptr);
}

TEST(CounterMemoryManagerTest, CopyEvents)
{
    hsa_agent_t          agent = {.handle = 1};
    CounterMemoryManager mgr(agent, dummy_alloc, dummy_dealloc, nullptr);

    aqlprofile_pmc_event_t events[2]{};
    events[0].event_id  = 1;
    events[0].flags.raw = 0;
    events[1].event_id  = 2;
    events[1].flags.raw = 1;

    mgr.CopyEvents(events, 2);
    auto& ev = mgr.GetEvents();
    ASSERT_GE(ev.size(), 2u);
}

TEST(TraceMemoryManagerTest, TraceControlBufAndATTParams)
{
    aqlprofile_agent_handle_t agent = {.handle = 1};
    TraceMemoryManager        mgr(agent, dummy_alloc, dummy_dealloc, dummy_copy, nullptr);
    mgr.CreateTraceControlBuf(64);
    auto buf = mgr.GetTraceControlBuf<uint8_t>();
    ASSERT_NE(buf, nullptr);

    hsa_ven_amd_aqlprofile_parameter_t params[2]{};
    params[0].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET;
    params[0].value          = 42;
    params[1].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION;
    params[1].value          = 0xF;

    mgr.CopyATTParams(params, 2);
    ASSERT_EQ(mgr.GetSimdMask(), 0xF);
    const auto& att_params = mgr.GetATTParams();
    ASSERT_EQ(att_params.size(), 2u);
}

TEST(TraceMemoryManagerTest, MultiSEDrainResetsAllSwapCountersOnCallbackError)
{
    auto info                 = aqlprofile_agent_info_v1_t{};
    info.agent_gfxip          = "gfx900";
    info.cu_num               = 64;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    auto agent                = aql_profile::RegisterAgent(&info);
    auto mgr                  = std::make_shared<TraceMemoryManager>(
        agent, dummy_alloc, dummy_dealloc, dummy_copy, nullptr);
    mgr->CreateTraceControlBuf(info.se_num * sizeof(pm4_builder::TraceControl));
    auto* control = mgr->GetTraceControlBuf<pm4_builder::TraceControl>();
    for(size_t se = 0; se < info.se_num; ++se)
    {
        control[se]                      = {};
        control[se].wptr                 = se + 1;
        mgr->config.target_cu_per_se[se] = se % 2 == 0 ? 0 : -1;
        mgr->config.buffer_data[se]      = {mgr->AddSqttOutputBuf(256), mgr->AddSqttOutputBuf(256)};
    }
    mgr->config.capacity_per_se = 256;
    mgr->AddExtraCmdBuf(64);
    mgr->buffer_swaps = {1, 0, 2, 0};
    mgr->read_offsets.assign(info.se_num, 0);
    MemoryManager::RegisterManager(mgr);
    auto cleanup = rocprofiler::common::scope_destructor{
        [&] { MemoryManager::DeleteManager(mgr->GetHandler()); }};

    struct state_t
    {
        TraceMemoryManager*   mgr;
        std::vector<uint32_t> engines{};
        hsa_status_t          callback_status;
    };
    auto callback = [](uint32_t se, void* data, uint64_t size, void* userdata) {
        auto& state = *static_cast<state_t*>(userdata);
        state.engines.push_back(se);
        EXPECT_EQ(size, (se + 1) * 32u);
        EXPECT_EQ(data, state.mgr->config.buffer_data.at(se).at(se == 0 ? 0 : 1));
        return state.callback_status;
    };
    for(auto status : {HSA_STATUS_SUCCESS, HSA_STATUS_ERROR})
    {
        mgr->buffer_swaps = {1, 0, 2, 0};
        for(size_t se = 0; se < info.se_num; ++se)
            control[se].cntr =
                mgr->buffer_swaps[se] * mgr->config.capacity_per_se / 32 + control[se].wptr;
        auto state = state_t{mgr.get(), {}, status};
        EXPECT_EQ(aqlprofile_att_iterate_data({mgr->GetHandler()}, callback, &state), status);
        EXPECT_EQ(state.engines,
                  status == HSA_STATUS_SUCCESS ? std::vector<uint32_t>({0, 2})
                                               : std::vector<uint32_t>({0}));
        EXPECT_EQ(mgr->buffer_swaps, std::vector<size_t>(info.se_num, 0));
    }
}

TEST(TraceMemoryManagerTest, Gfx11FamilyReservesDisabledSEBuffers)
{
    for(const auto* gfxip : {"gfx1100", "gfx1151", "gfx1170"})
    {
        SCOPED_TRACE(gfxip);
        auto info                 = aqlprofile_agent_info_v1_t{};
        info.agent_gfxip          = gfxip;
        info.cu_num               = 64;
        info.se_num               = 4;
        info.xcc_num              = 1;
        info.shader_arrays_per_se = 2;
        auto agent                = aql_profile::RegisterAgent(&info);
        auto parameter            = aqlprofile_att_parameter_t{};
        parameter.parameter_name  = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE;
        parameter.value           = 1 << 20;
        auto profile              = aqlprofile_att_profile_t{agent, &parameter, 1};
        auto handle               = aqlprofile_handle_t{};
        auto packets              = aqlprofile_att_control_aql_packets_t{};
        ASSERT_EQ(aqlprofile_att_create_packets(
                      &handle, &packets, profile, dummy_alloc, dummy_dealloc, dummy_copy, nullptr),
                  HSA_STATUS_SUCCESS);
        auto cleanup =
            rocprofiler::common::scope_destructor{[&] { aqlprofile_att_delete_packets(handle); }};
        auto mgr =
            std::dynamic_pointer_cast<TraceMemoryManager>(MemoryManager::GetManager(handle.handle));
        ASSERT_NE(mgr, nullptr);
        const auto alignment =
            aql_profile::Pm4Factory::Create(agent)->GetSqttBuilder()->BufferAlignment();
        EXPECT_EQ(mgr->config.capacity_per_disabled_se, uint64_t{1} << alignment);
        EXPECT_LT(mgr->config.capacity_per_se, parameter.value);
    }
}

namespace
{
constexpr uint32_t att_agent_se_num = 4;
constexpr uint32_t pm4_wait_reg_mem = 0x3C;
constexpr uint32_t pm4_copy_data    = 0x40;

aqlprofile_agent_handle_t
register_att_agent(const char* gfxip)
{
    auto info                 = aqlprofile_agent_info_v1_t{};
    info.agent_gfxip          = gfxip;
    info.cu_num               = 64;
    info.se_num               = att_agent_se_num;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    return aql_profile::RegisterAgent(&info);
}

aqlprofile_att_parameter_t
att_parameter(uint32_t name, uint32_t value)
{
    auto parameter           = aqlprofile_att_parameter_t{};
    parameter.parameter_name = static_cast<hsa_ven_amd_aqlprofile_parameter_name_t>(name);
    parameter.value          = value;
    return parameter;
}

struct att_capture
{
    std::vector<void*>                allocations{};
    std::vector<std::vector<uint8_t>> copies{};
};

hsa_status_t
capture_alloc(void** ptr, size_t size, aqlprofile_buffer_desc_flags_t flags, void* userdata)
{
    const auto status = dummy_alloc(ptr, size, flags, userdata);
    if(status == HSA_STATUS_SUCCESS)
        static_cast<att_capture*>(userdata)->allocations.push_back(*ptr);
    return status;
}

hsa_status_t
capture_copy(void* dst, const void* src, size_t size, void* userdata)
{
    const auto* bytes = static_cast<const uint8_t*>(src);
    static_cast<att_capture*>(userdata)->copies.emplace_back(bytes, bytes + size);
    return dummy_copy(dst, src, size, userdata);
}

hsa_status_t
switchable_alloc(void** ptr, size_t size, aqlprofile_buffer_desc_flags_t flags, void* userdata)
{
    if(*static_cast<bool*>(userdata)) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    return dummy_alloc(ptr, size, flags, userdata);
}

// Type-3 PM4 packets as (opcode, body)
std::vector<std::pair<uint32_t, std::vector<uint32_t>>>
pm4_packets(const std::vector<uint8_t>& bytes)
{
    std::vector<uint32_t> dwords(bytes.size() / sizeof(uint32_t));
    std::memcpy(dwords.data(), bytes.data(), dwords.size() * sizeof(uint32_t));

    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> packets{};
    for(size_t i = 0; i < dwords.size();)
    {
        const size_t count = ((dwords[i] >> 16) & 0x3FFF) + 1;
        if(dwords[i] >> 30 != 3 || i + 1 + count > dwords.size())
        {
            ADD_FAILURE() << "malformed PM4 packet at dword " << i;
            break;
        }
        packets.emplace_back(
            (dwords[i] >> 8) & 0xFF,
            std::vector<uint32_t>(dwords.begin() + i + 1, dwords.begin() + i + 1 + count));
        i += 1 + count;
    }
    return packets;
}
}  // namespace

TEST(TraceMemoryManagerTest, StopWaitPollsTheStatusRegister)
{
    // STOP waits for the trace to go idle, then copies the status register into
    // TraceControl::status. The wait must poll that same register.
    for(const auto* gfxip : {"gfx942", "gfx1201"})
    {
        SCOPED_TRACE(gfxip);
        auto capture = att_capture{};
        auto parameter =
            att_parameter(HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE, 1 << 20);
        auto profile = aqlprofile_att_profile_t{register_att_agent(gfxip), &parameter, 1};
        auto handle  = aqlprofile_handle_t{};
        auto packets = aqlprofile_att_control_aql_packets_t{};
        ASSERT_EQ(
            aqlprofile_att_create_packets(
                &handle, &packets, profile, capture_alloc, dummy_dealloc, capture_copy, &capture),
            HSA_STATUS_SUCCESS);
        auto cleanup =
            rocprofiler::common::scope_destructor{[&] { aqlprofile_att_delete_packets(handle); }};
        ASSERT_EQ(capture.copies.size(), 2u);

        const auto control     = reinterpret_cast<uint64_t>(capture.allocations.front());
        const auto control_end = control + att_agent_se_num * sizeof(pm4_builder::TraceControl);
        std::set<uint32_t> status_registers{};
        std::set<uint32_t> polled_registers{};
        for(const auto& [opcode, body] : pm4_packets(capture.copies.back()))
        {
            if(opcode == pm4_wait_reg_mem && (body.at(0) & 0x30) == 0)
                polled_registers.insert(body.at(1));
            if(opcode == pm4_copy_data && (body.at(0) & 0xF) == 0)
            {
                const uint64_t destination = body.at(3) | (uint64_t{body.at(4)} << 32);
                if(destination >= control && destination < control_end &&
                   (destination - control) % sizeof(pm4_builder::TraceControl) == 0)
                    status_registers.insert(body.at(1));
            }
        }
        ASSERT_EQ(status_registers.size(), 1u);
        EXPECT_EQ(polled_registers, status_registers);
    }
}

TEST(TraceMemoryManagerTest, UpdateBufferStatusRejectsUnknownShaderEngine)
{
    auto parameter = att_parameter(HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE, 1 << 20);
    auto profile   = aqlprofile_att_profile_t{register_att_agent("gfx942"), &parameter, 1};
    auto handle    = aqlprofile_handle_t{};
    auto packets   = aqlprofile_att_control_aql_packets_t{};
    ASSERT_EQ(aqlprofile_att_create_packets(
                  &handle, &packets, profile, dummy_alloc, dummy_dealloc, dummy_copy, nullptr),
              HSA_STATUS_SUCCESS);
    auto cleanup =
        rocprofiler::common::scope_destructor{[&] { aqlprofile_att_delete_packets(handle); }};

    for(int shader_engine : {-1, static_cast<int>(att_agent_se_num)})
    {
        auto status = aqlprofile_att_buffer_status_t{};
        EXPECT_EQ(aqlprofile_att_update_buffer_status(&status, handle, shader_engine, 0),
                  HSA_STATUS_ERROR_INVALID_ARGUMENT)
            << shader_engine;
        EXPECT_EQ(status._size, 0u) << shader_engine;
    }
    EXPECT_EQ(aqlprofile_att_update_buffer_status(nullptr, handle, 0, 0),
              HSA_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST(TraceMemoryManagerTest, GetBufferPacketsReturnsAllocationFailure)
{
    aqlprofile_att_parameter_t parameters[] = {
        att_parameter(HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE, 1 << 20),
        att_parameter(AQLPROFILE_ATT_PARAMETER_NAME_NUM_BUFFERS, 2)};
    auto fail_allocations = false;
    auto profile          = aqlprofile_att_profile_t{register_att_agent("gfx942"), parameters, 2};
    auto handle           = aqlprofile_handle_t{};
    auto packets          = aqlprofile_att_control_aql_packets_t{};
    ASSERT_EQ(aqlprofile_att_create_packets(&handle,
                                            &packets,
                                            profile,
                                            switchable_alloc,
                                            dummy_dealloc,
                                            dummy_copy,
                                            &fail_allocations),
              HSA_STATUS_SUCCESS);
    auto cleanup =
        rocprofiler::common::scope_destructor{[&] { aqlprofile_att_delete_packets(handle); }};

    fail_allocations  = true;
    auto header       = uint64_t{0};
    auto query        = hsa_ext_amd_aql_pm4_packet_t{};
    auto swap_packets = std::vector<hsa_ext_amd_aql_pm4_packet_t>(2);
    auto swaps     = std::vector<hsa_ext_amd_aql_pm4_packet_t*>{&swap_packets[0], &swap_packets[1]};
    auto num_swaps = uint64_t{swaps.size()};
    EXPECT_EQ(
        aqlprofile_att_get_buffer_packets(&header, &query, swaps.data(), &num_swaps, handle, 0, 0),
        HSA_STATUS_ERROR_INVALID_ALLOCATION);
}

namespace
{
struct alloc_counts_t
{
    size_t allocs   = 0;
    size_t deallocs = 0;
};

hsa_status_t
counting_alloc(void** ptr, size_t size, aqlprofile_buffer_desc_flags_t flags, void* data)
{
    ++static_cast<alloc_counts_t*>(data)->allocs;
    return dummy_alloc(ptr, size, flags, nullptr);
}

void
counting_dealloc(void* ptr, void* data)
{
    ++static_cast<alloc_counts_t*>(data)->deallocs;
    dummy_dealloc(ptr, nullptr);
}
}  // namespace

TEST(TraceMemoryManagerTest, HostBufferIsReusedAndOnlyGrows)
{
    aqlprofile_agent_handle_t agent = {.handle = 1};
    alloc_counts_t            counts{};
    {
        TraceMemoryManager mgr(agent, counting_alloc, counting_dealloc, dummy_copy, &counts);

        void* first = mgr.GetHostBuffer(4096);
        ASSERT_NE(first, nullptr);
        EXPECT_EQ(mgr.GetHostBuffer(4096), first);
        EXPECT_EQ(mgr.GetHostBuffer(1024), first);
        EXPECT_EQ(counts.allocs, 1u);
        EXPECT_EQ(counts.deallocs, 0u);

        void* grown = mgr.GetHostBuffer(3u << 20);
        ASSERT_NE(grown, nullptr);
        std::memset(grown, 0, 3u << 20);
        EXPECT_EQ(counts.allocs, 2u);
        EXPECT_EQ(counts.deallocs, 1u);
        EXPECT_EQ(mgr.GetHostBuffer(1024), grown);
        EXPECT_EQ(counts.allocs, 2u);
    }
    EXPECT_EQ(counts.deallocs, counts.allocs);
}

TEST(CodeobjMemoryManagerTest, CmdBufferAlloc)
{
    aqlprofile_agent_handle_t agent = {.handle = 1};
    CodeobjMemoryManager      mgr(agent, dummy_alloc, dummy_dealloc, 128, nullptr);
    ASSERT_NE(mgr.cmd_buffer.get(), nullptr);
}
