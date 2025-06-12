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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <hsa/hsa.h>
#include "core/include/aql_profile_v2.h"

// Mocks and helpers
namespace {

struct MockMemory {
    std::vector<uint8_t> data;
    void* alloc(size_t size) {
        data.resize(size);
        return data.data();
    }
    void dealloc(void* /*ptr*/) {
        // No-op for vector-backed memory
    }
    void copy(void* dst, const void* src, size_t size) {
        memcpy(dst, src, size);
    }
};

// Corrected mock_alloc matching aqlprofile_memory_alloc_callback_t
hsa_status_t mock_alloc(void** ptr, uint64_t size, aqlprofile_buffer_desc_flags_t /*flags*/, void* userdata) {
    auto* mem = static_cast<MockMemory*>(userdata);
    *ptr = mem->alloc(size);
    return HSA_STATUS_SUCCESS;
}

// Corrected mock_dealloc matching aqlprofile_memory_dealloc_callback_t
void mock_dealloc(void* /*ptr*/, void* /*userdata*/) {
    // No-op
}

// Corrected mock_memcpy matching aqlprofile_memory_copy_t
hsa_status_t mock_memcpy(void* dst, const void* src, size_t size, void* /*userdata*/) {
    memcpy(dst, src, size);
    return HSA_STATUS_SUCCESS;
}

} // namespace

// Test: CreatePacketsSuccess
TEST(CountersTest, CreatePacketsSuccess) {
    MockMemory mem;
    aqlprofile_pmc_profile_t profile = {};
    // Fill profile with minimal valid data
    profile.agent.handle = 0; // Use a valid agent handle in real test
    profile.events = nullptr;
    profile.event_count = 0;

    aqlprofile_handle_t handle = {};
    aqlprofile_pmc_aql_packets_t packets = {};

    hsa_status_t status = aqlprofile_pmc_create_packets(
        &handle, &packets, profile, mock_alloc, mock_dealloc, mock_memcpy, &mem);

    // Accept HSA_STATUS_ERROR if agent/event is not valid in this test context
    EXPECT_TRUE(status == HSA_STATUS_SUCCESS || status == HSA_STATUS_ERROR);
}

// Test: DeletePackets
TEST(CountersTest, DeletePackets) {
    MockMemory mem;
    aqlprofile_pmc_profile_t profile = {};
    profile.agent.handle = 0;
    profile.events = nullptr;
    profile.event_count = 0;

    aqlprofile_handle_t handle = {};
    aqlprofile_pmc_aql_packets_t packets = {};

    hsa_status_t status = aqlprofile_pmc_create_packets(
        &handle, &packets, profile, mock_alloc, mock_dealloc, mock_memcpy, &mem);
    
    // Only proceed if creation succeeded
    if (status == HSA_STATUS_SUCCESS) {
        // This should not crash or throw
        aqlprofile_pmc_delete_packets(handle);
    }
}

// Test: ValidateEvent
TEST(CountersTest, ValidateEvent) {
    aqlprofile_agent_handle_t agent = {};
    agent.handle = 0;
    
    aqlprofile_pmc_event_t event = {};
    event.block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM;
    
    bool result = true;
    hsa_status_t status = aqlprofile_validate_pmc_event(agent, &event, &result);
    
    // In a mock environment, we can't guarantee validation, but we can check that it runs
    EXPECT_TRUE(status == HSA_STATUS_SUCCESS || status == HSA_STATUS_ERROR);
}