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

TEST(CodeobjMemoryManagerTest, CmdBufferAlloc)
{
    aqlprofile_agent_handle_t agent = {.handle = 1};
    CodeobjMemoryManager      mgr(agent, dummy_alloc, dummy_dealloc, 128, nullptr);
    ASSERT_NE(mgr.cmd_buffer.get(), nullptr);
}
