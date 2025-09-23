#include <gtest/gtest.h>
#include "util/hsa_rsrc_factory.h"

// Test fixture for HsaRsrcFactory
class HsaRsrcFactoryTest : public ::testing::Test {
protected:
    void TearDown() override {
        HsaRsrcFactory::Destroy();
    }
};

// Test: Factory instance creation and destruction (happy path)
TEST_F(HsaRsrcFactoryTest, FactoryCreationAndDestruction) {
    HsaRsrcFactory* factory = HsaRsrcFactory::Create();
    ASSERT_NE(factory, nullptr);
    HsaRsrcFactory::Destroy();
}

// Test: Singleton pattern is enforced
TEST_F(HsaRsrcFactoryTest, SingletonBehavior) {
    HsaRsrcFactory* factory1 = HsaRsrcFactory::Create();
    HsaRsrcFactory* factory2 = HsaRsrcFactory::Create();
    EXPECT_EQ(factory1, factory2);
    HsaRsrcFactory::Destroy();
}

// Test: At least one CPU agent is detected (edge: system dependent)
TEST_F(HsaRsrcFactoryTest, CpuAgentCountNonZero) {
    HsaRsrcFactory* factory = HsaRsrcFactory::Create();
    EXPECT_GT(factory->GetCountOfCpuAgents(), 0u);
}

// Test: GPU agent count is valid (edge: may be zero if no GPU present)
TEST_F(HsaRsrcFactoryTest, GpuAgentCountValid) {
    HsaRsrcFactory* factory = HsaRsrcFactory::Create();
    EXPECT_GE(factory->GetCountOfGpuAgents(), 0u);
}

// Test: GetCpuAgentInfo returns valid info for first CPU agent (happy path)
TEST_F(HsaRsrcFactoryTest, GetCpuAgentInfoReturnsValid) {
    HsaRsrcFactory* factory = HsaRsrcFactory::Create();
    const AgentInfo* info = nullptr;
    bool ok = factory->GetCpuAgentInfo(0, &info);
    EXPECT_TRUE(ok);
    EXPECT_NE(info, nullptr);
}

// Test: GetCpuAgentInfo returns false for out-of-range index (edge case)
TEST_F(HsaRsrcFactoryTest, GetCpuAgentInfoOutOfRange) {
    HsaRsrcFactory* factory = HsaRsrcFactory::Create();
    const AgentInfo* info = nullptr;
    size_t count = factory->GetCountOfCpuAgents();
    bool ok = factory->GetCpuAgentInfo(count, &info);
    EXPECT_FALSE(ok);
    EXPECT_EQ(info, nullptr);
}

// Test: GetGpuAgentInfo returns false for out-of-range index (edge case)
TEST_F(HsaRsrcFactoryTest, GetGpuAgentInfoOutOfRange) {
    HsaRsrcFactory* factory = HsaRsrcFactory::Create();
    const AgentInfo* info = nullptr;
    size_t count = factory->GetCountOfGpuAgents();
    bool ok = factory->GetGpuAgentInfo(count, &info);
    EXPECT_FALSE(ok);
    EXPECT_EQ(info, nullptr);
}
