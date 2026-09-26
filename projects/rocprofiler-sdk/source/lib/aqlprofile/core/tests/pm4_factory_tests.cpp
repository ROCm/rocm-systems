// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "lib/aqlprofile/core/pm4_factory.h"

using namespace aql_profile;

namespace
{
// Helper to create a valid agent info struct
aqlprofile_agent_info_v1_t
makeTestAgentInfo(const char* gfxip = "gfx900")
{
    aqlprofile_agent_info_v1_t info{};
    info.agent_gfxip          = strdup(gfxip);
    info.cu_num               = 64;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    info.domain               = 0;
    info.location_id          = 0x1234;
    return info;
}

}  // namespace

// Test: Register agent and retrieve info (happy path)
TEST(Pm4FactoryTest, RegisterAgentAndGetAgentInfo)
{
    auto                      agentInfo = makeTestAgentInfo();
    aqlprofile_agent_handle_t handle    = RegisterAgent(&agentInfo);
    const AgentInfo*          info      = GetAgentInfo(handle);
    ASSERT_NE(info, nullptr) << "AgentInfo should not be null";
    EXPECT_EQ(info->cu_num, 64u);
    EXPECT_EQ(info->se_num, 4u);
    EXPECT_EQ(info->xcc_num, 1u);
    EXPECT_EQ(info->shader_arrays_per_se, 2u);
}

// Test: GetAgentInfo returns nullptr for invalid handle
TEST(Pm4FactoryTest, GetAgentInfoInvalidHandleReturnsNull)
{
    aqlprofile_agent_handle_t invalidHandle{};
    invalidHandle.handle  = 99999;  // unlikely to exist
    const AgentInfo* info = GetAgentInfo(invalidHandle);
    EXPECT_EQ(info, nullptr);
}

// Test: gfx117X resolves to the dedicated factory id rather than falling back
// to the generic GFX11 id.
TEST(Pm4FactoryTest, GetGpuIdGfx117x)
{
    EXPECT_EQ(Pm4Factory::GetGpuId("gfx1170"), GFX117X_GPU_ID);
    EXPECT_EQ(Pm4Factory::GetGpuId("gfx1171"), GFX117X_GPU_ID);
    EXPECT_EQ(Pm4Factory::GetGpuId("gfx1172"), GFX117X_GPU_ID);
}

// Test: the gfxip table is matched by ordered prefix, so adding "gfx117" must
// not shadow the neighbouring entries matched by shorter or longer prefixes.
TEST(Pm4FactoryTest, GetGpuIdPrefixOrderingIsPreserved)
{
    EXPECT_EQ(Pm4Factory::GetGpuId("gfx1100"), GFX11_GPU_ID);
    EXPECT_EQ(Pm4Factory::GetGpuId("gfx1151"), GFX115X_GPU_ID);
    EXPECT_EQ(Pm4Factory::GetGpuId("gfx1200"), GFX12_GPU_ID);
    EXPECT_EQ(Pm4Factory::GetGpuId("gfx1250"), MI450_GPU_ID);
    EXPECT_EQ(Pm4Factory::GetGpuId("gfx942"), MI300_GPU_ID);
    EXPECT_EQ(Pm4Factory::GetGpuId("gfx950"), MI350_GPU_ID);
    EXPECT_EQ(Pm4Factory::GetGpuId("gfxJunk"), INVAL_GPU_ID);
}

// Test: the enumerator sits between GFX11 and GFX12, which the relational
// comparisons used for PM4 dispatch and data collection rely on.
TEST(Pm4FactoryTest, Gfx117xGpuIdOrdering)
{
    EXPECT_GT(GFX117X_GPU_ID, GFX11_GPU_ID);
    EXPECT_LT(GFX117X_GPU_ID, GFX12_GPU_ID);
    EXPECT_GT(GFX117X_GPU_ID, MI350_GPU_ID);
}
