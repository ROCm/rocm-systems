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

#include "core/memorymanager.hpp"
#include "gtest/gtest.h"
#include <cstring>

// Dummy alloc/dealloc functions for testing
hsa_status_t dummy_alloc(void** ptr, size_t size, aqlprofile_buffer_desc_flags_t, void*) {
    *ptr = malloc(size);
    return *ptr ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}
void dummy_dealloc(void* ptr, void*) {
    free(ptr);
}
hsa_status_t dummy_alloc_fail(void**, size_t, aqlprofile_buffer_desc_flags_t, void*) {
    return HSA_STATUS_ERROR;
}
hsa_status_t dummy_copy(void* dst, const void* src, size_t size, void*) {
    memcpy(dst, src, size);
    return HSA_STATUS_SUCCESS;
}

// Helper subclass to expose protected AllocMemory for testing
class TestCounterMemoryManager : public CounterMemoryManager {
public:
    using CounterMemoryManager::CounterMemoryManager;
    using CounterMemoryManager::AllocMemory;
};

TEST(CounterMemoryManagerTest, AllocMemory_Success) {
    hsa_agent_t agent = {.handle = 1};
    TestCounterMemoryManager mgr(agent, dummy_alloc, dummy_dealloc, nullptr);
    aqlprofile_buffer_desc_flags_t flags{};
    auto mem = mgr.AllocMemory(64, flags);
    ASSERT_NE(mem.get(), nullptr);
}

TEST(CounterMemoryManagerTest, AllocMemory_FailureThrows) {
    hsa_agent_t agent = {.handle = 1};
    TestCounterMemoryManager mgr(agent, dummy_alloc_fail, dummy_dealloc, nullptr);
    aqlprofile_buffer_desc_flags_t flags{};
    EXPECT_THROW(mgr.AllocMemory(64, flags), hsa_status_t);
}

TEST(CounterMemoryManagerTest, CmdBufAndOutputBuf) {
    hsa_agent_t agent = {.handle = 1};
    CounterMemoryManager mgr(agent, dummy_alloc, dummy_dealloc, nullptr);
    mgr.CreateCmdBuf(128);
    ASSERT_NE(mgr.GetCmdBuf(), nullptr);
    mgr.CreateOutputBuf(256);
    ASSERT_NE(mgr.GetOutputBuf(), nullptr);
    ASSERT_EQ(mgr.GetOutputBufSize(), 256u);
}

TEST(CounterMemoryManagerTest, RegisterAndGetManager) {
    hsa_agent_t agent = {.handle = 1};
    auto mgr = std::make_shared<CounterMemoryManager>(agent, dummy_alloc, dummy_dealloc, nullptr);
    size_t handle = mgr->GetHandler();
    CounterMemoryManager::RegisterManager(mgr);
    auto found = CounterMemoryManager::GetManager(handle);
    ASSERT_EQ(found.get(), mgr.get());
    CounterMemoryManager::DeleteManager(handle);
    ASSERT_EQ(CounterMemoryManager::GetManager(handle), nullptr);
}

TEST(CounterMemoryManagerTest, CopyEvents) {
    hsa_agent_t agent = {.handle = 1};
    CounterMemoryManager mgr(agent, dummy_alloc, dummy_dealloc, nullptr);

    aqlprofile_pmc_event_t events[2]{};
    events[0].event_id = 1;
    events[0].flags.raw = 0;
    events[1].event_id = 2;
    events[1].flags.raw = 1;

    mgr.CopyEvents(events, 2);
    auto& ev = mgr.GetEvents();
    ASSERT_GE(ev.size(), 2u);
}

TEST(TraceMemoryManagerTest, TraceControlBufAndATTParams) {
    hsa_agent_t agent = {.handle = 1};
    TraceMemoryManager mgr(agent, dummy_alloc, dummy_dealloc, dummy_copy, nullptr);
    mgr.CreateTraceControlBuf(64);
    auto buf = mgr.GetTraceControlBuf<uint8_t>();
    ASSERT_NE(buf, nullptr);

    hsa_ven_amd_aqlprofile_parameter_t params[2]{};
    params[0].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET;
    params[0].value = 42;
    params[1].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION;
    params[1].value = 0xF;

    mgr.CopyATTParams(params, 2);
    ASSERT_EQ(mgr.GetSimdMask(), 0xF);
    const auto& att_params = mgr.GetATTParams();
    ASSERT_EQ(att_params.size(), 2u);
}

TEST(CodeobjMemoryManagerTest, CmdBufferAlloc) {
    hsa_agent_t agent = {.handle = 1};
    CodeobjMemoryManager mgr(agent, dummy_alloc, dummy_dealloc, 128, nullptr);
    ASSERT_NE(mgr.cmd_buffer.get(), nullptr);
}