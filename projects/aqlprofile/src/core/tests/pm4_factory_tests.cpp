//Copyright © Advanced Micro Devices, Inc., or its affiliates.
//SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "core/pm4_factory.h"

using namespace aql_profile;

namespace {

// Helper to create a valid agent info struct
aqlprofile_agent_info_v1_t makeTestAgentInfo(const char* gfxip = "gfx900") {
    aqlprofile_agent_info_v1_t info{};
    info.agent_gfxip = strdup(gfxip);
    info.cu_num = 64;
    info.se_num = 4;
    info.xcc_num = 1;
    info.shader_arrays_per_se = 2;
    info.domain = 0;
    info.location_id = 0x1234;
    return info;
}

} // namespace

// Test: Register agent and retrieve info (happy path)
TEST(Pm4FactoryTest, RegisterAgentAndGetAgentInfo) {
    auto agentInfo = makeTestAgentInfo();
    aqlprofile_agent_handle_t handle = RegisterAgent(&agentInfo);
    const AgentInfo* info = GetAgentInfo(handle);
    ASSERT_NE(info, nullptr) << "AgentInfo should not be null";
    EXPECT_EQ(info->cu_num, 64u);
    EXPECT_EQ(info->se_num, 4u);
    EXPECT_EQ(info->xcc_num, 1u);
    EXPECT_EQ(info->shader_arrays_per_se, 2u);
}

// Test: Register agent with v2 struct including cu_bitmap
TEST(Pm4FactoryTest, RegisterAgentV2WithCuBitmap) {
    aqlprofile_agent_info_v2_t info_v2{};
    info_v2.agent_gfxip = "gfx942";
    info_v2.cu_num = 304;
    info_v2.se_num = 32;
    info_v2.xcc_num = 8;
    info_v2.shader_arrays_per_se = 1;
    info_v2.domain = 0;
    info_v2.location_id = 0x5678;
    // Set asymmetric cu_bitmap: SE0 has 0x3FFFF (18 CUs), SE1 has 0xFFFF (16 CUs)
    info_v2.cu_bitmap[0][0] = 0x3FFFF;
    info_v2.cu_bitmap[1][0] = 0xFFFF;

    aqlprofile_agent_handle_t handle = RegisterAgent(&info_v2);
    const AgentInfo* info = GetAgentInfo(handle);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->cu_num, 304u);
    EXPECT_EQ(info->se_num, 32u);
    EXPECT_EQ(info->xcc_num, 8u);
    EXPECT_EQ(info->cu_bitmap[0][0], 0x3FFFFu);
    EXPECT_EQ(info->cu_bitmap[1][0], 0xFFFFu);
    EXPECT_EQ(info->cu_bitmap[2][0], 0u);
}

// Test: v2 registration via RegisterAgent directly
TEST(Pm4FactoryTest, RegisterAgentInfoV2) {
    aqlprofile_agent_info_v2_t info_v2{};
    info_v2.agent_gfxip = "gfx900";
    info_v2.cu_num = 64;
    info_v2.se_num = 4;
    info_v2.xcc_num = 1;
    info_v2.shader_arrays_per_se = 2;
    info_v2.domain = 1;
    info_v2.location_id = 0xABCD;
    info_v2.cu_bitmap[0][0] = 0xFFFF;
    info_v2.cu_bitmap[0][1] = 0xFFFF;
    info_v2.cu_bitmap[1][0] = 0xFFFF;
    info_v2.cu_bitmap[1][1] = 0xFFFF;

    aqlprofile_agent_handle_t handle = RegisterAgent(&info_v2);

    const AgentInfo* info = GetAgentInfo(handle);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->cu_num, 64u);
    EXPECT_EQ(info->cu_bitmap[0][0], 0xFFFFu);
    EXPECT_EQ(info->cu_bitmap[1][1], 0xFFFFu);
    EXPECT_EQ(info->bdf_id, 0xABCDu);
}

// Test: GetAgentInfo returns nullptr for invalid handle
TEST(Pm4FactoryTest, GetAgentInfoInvalidHandleReturnsNull) {
    aqlprofile_agent_handle_t invalidHandle{};
    invalidHandle.handle = 99999; // unlikely to exist
    const AgentInfo* info = GetAgentInfo(invalidHandle);
    EXPECT_EQ(info, nullptr);
}
