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

// Test: GetAgentInfo returns nullptr for invalid handle
TEST(Pm4FactoryTest, GetAgentInfoInvalidHandleReturnsNull) {
    aqlprofile_agent_handle_t invalidHandle{};
    invalidHandle.handle = 99999; // unlikely to exist
    const AgentInfo* info = GetAgentInfo(invalidHandle);
    EXPECT_EQ(info, nullptr);
}
