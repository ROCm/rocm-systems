//Copyright © Advanced Micro Devices, Inc., or its affiliates.
//SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include "core/architecture_registry.hpp"
#include "core/hardware_architecture.hpp"
#include "pm4/cmd_builder.h"
#include "def/gpu_block_info.h"

namespace aql_profile {
namespace {

// Mock architecture for testing
class MockArchitecture : public HardwareArchitecture {
 public:
  explicit MockArchitecture(const std::string& gfxip, const std::string& name)
      : config_(), schema_() {
    config_.gfxip = gfxip;
    config_.name = name;
  }

  const HardwareConfig& GetConfig() const override { return config_; }
  const RegisterSchema& GetRegisterSchema() const override { return schema_; }
  const GpuBlockInfo* GetBlockInfo(uint32_t block_id) const override { return nullptr; }
  uint32_t FindBlockByName(const char* name) const override { return UINT32_MAX; }
  uint32_t GetBlockCount() const override { return 0; }
  pm4_builder::CmdBuilder* CreateCmdBuilder() const override { return nullptr; }

 private:
  HardwareConfig config_;
  RegisterSchema schema_;
};

class ArchitectureRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear registry before each test
    ArchitectureRegistry::Instance().Clear();
  }

  void TearDown() override {
    // Clean up after each test
    ArchitectureRegistry::Instance().Clear();
  }
};

TEST_F(ArchitectureRegistryTest, RegisterAndLookup) {
  auto& registry = ArchitectureRegistry::Instance();

  auto arch = std::make_unique<MockArchitecture>("gfx90a", "MI200");
  registry.Register("gfx90a", std::move(arch));

  const auto* found = registry.Lookup("gfx90a");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->GetConfig().gfxip, "gfx90a");
  EXPECT_EQ(found->GetConfig().name, "MI200");
}

TEST_F(ArchitectureRegistryTest, LookupNotFound) {
  auto& registry = ArchitectureRegistry::Instance();

  const auto* found = registry.Lookup("gfx90a");
  EXPECT_EQ(found, nullptr);
}

TEST_F(ArchitectureRegistryTest, PrefixMatching) {
  auto& registry = ArchitectureRegistry::Instance();

  auto arch = std::make_unique<MockArchitecture>("gfx90a", "MI200");
  registry.Register("gfx90a", std::move(arch));

  // Should match with full gfxip string that has the prefix
  const auto* found = registry.Lookup("gfx90a:sramecc+:xnack-");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->GetConfig().gfxip, "gfx90a");
}

TEST_F(ArchitectureRegistryTest, LongestPrefixFirst) {
  auto& registry = ArchitectureRegistry::Instance();

  // Register both gfx90a and gfx90
  registry.Register("gfx90a", std::make_unique<MockArchitecture>("gfx90a", "MI200"));
  registry.Register("gfx90", std::make_unique<MockArchitecture>("gfx90", "Vega20"));

  // Looking up "gfx90a" should match the longer prefix first
  const auto* found = registry.Lookup("gfx90a");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->GetConfig().name, "MI200");

  // Looking up "gfx906" should match "gfx90"
  found = registry.Lookup("gfx906");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->GetConfig().name, "Vega20");
}

TEST_F(ArchitectureRegistryTest, MultipleArchitectures) {
  auto& registry = ArchitectureRegistry::Instance();

  registry.Register("gfx908", std::make_unique<MockArchitecture>("gfx908", "MI100"));
  registry.Register("gfx90a", std::make_unique<MockArchitecture>("gfx90a", "MI200"));
  registry.Register("gfx940", std::make_unique<MockArchitecture>("gfx940", "MI300"));
  registry.Register("gfx1100", std::make_unique<MockArchitecture>("gfx1100", "Navi31"));

  EXPECT_EQ(registry.Lookup("gfx908")->GetConfig().name, "MI100");
  EXPECT_EQ(registry.Lookup("gfx90a")->GetConfig().name, "MI200");
  EXPECT_EQ(registry.Lookup("gfx940")->GetConfig().name, "MI300");
  EXPECT_EQ(registry.Lookup("gfx1100")->GetConfig().name, "Navi31");
}

TEST_F(ArchitectureRegistryTest, GetExactMatch) {
  auto& registry = ArchitectureRegistry::Instance();

  registry.Register("gfx90a", std::make_unique<MockArchitecture>("gfx90a", "MI200"));

  const auto* found = registry.GetExact("gfx90a");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->GetConfig().name, "MI200");
}

TEST_F(ArchitectureRegistryTest, GetExactNoMatch) {
  auto& registry = ArchitectureRegistry::Instance();

  registry.Register("gfx90a", std::make_unique<MockArchitecture>("gfx90a", "MI200"));

  // Exact match should not find with suffix
  const auto* found = registry.GetExact("gfx90a:sramecc+");
  EXPECT_EQ(found, nullptr);
}

TEST_F(ArchitectureRegistryTest, IsRegistered) {
  auto& registry = ArchitectureRegistry::Instance();

  registry.Register("gfx90a", std::make_unique<MockArchitecture>("gfx90a", "MI200"));

  EXPECT_TRUE(registry.IsRegistered("gfx90a"));
  EXPECT_FALSE(registry.IsRegistered("gfx940"));
}

TEST_F(ArchitectureRegistryTest, GetRegisteredPrefixes) {
  auto& registry = ArchitectureRegistry::Instance();

  registry.Register("gfx908", std::make_unique<MockArchitecture>("gfx908", "MI100"));
  registry.Register("gfx90a", std::make_unique<MockArchitecture>("gfx90a", "MI200"));
  registry.Register("gfx940", std::make_unique<MockArchitecture>("gfx940", "MI300"));

  auto prefixes = registry.GetRegisteredPrefixes();
  EXPECT_EQ(prefixes.size(), 3u);
  EXPECT_THAT(prefixes, ::testing::UnorderedElementsAre("gfx908", "gfx90a", "gfx940"));
}

TEST_F(ArchitectureRegistryTest, Clear) {
  auto& registry = ArchitectureRegistry::Instance();

  registry.Register("gfx90a", std::make_unique<MockArchitecture>("gfx90a", "MI200"));
  EXPECT_TRUE(registry.IsRegistered("gfx90a"));

  registry.Clear();
  EXPECT_FALSE(registry.IsRegistered("gfx90a"));
  EXPECT_EQ(registry.GetRegisteredPrefixes().size(), 0u);
}

TEST_F(ArchitectureRegistryTest, RegisterNullThrows) {
  auto& registry = ArchitectureRegistry::Instance();

  EXPECT_THROW(registry.Register("gfx90a", nullptr), std::invalid_argument);
}

TEST_F(ArchitectureRegistryTest, ThreadSafety) {
  auto& registry = ArchitectureRegistry::Instance();

  // Register an architecture
  registry.Register("gfx90a", std::make_unique<MockArchitecture>("gfx90a", "MI200"));

  // Multiple threads should be able to lookup concurrently
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};

  for (int i = 0; i < 10; ++i) {
    threads.emplace_back([&registry, &success_count]() {
      for (int j = 0; j < 100; ++j) {
        const auto* arch = registry.Lookup("gfx90a");
        if (arch != nullptr && arch->GetConfig().name == "MI200") {
          success_count++;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(success_count.load(), 1000);
}

}  // namespace
}  // namespace aql_profile
