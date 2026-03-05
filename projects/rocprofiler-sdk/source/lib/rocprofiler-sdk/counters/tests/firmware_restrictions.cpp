// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../firmware_restrictions.hpp"

#include "lib/rocprofiler-sdk/agent.hpp"

#include <gtest/gtest.h>
#include <string>

namespace rocprofiler
{
namespace counters
{
namespace
{
TEST(FirmwareRestrictions, ValidYamlParsing)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters:
    - name: TEST_COUNTER
      description: "Test counter for unit tests"
      properties: []
      definitions:
        - architectures: ["gfx908"]
          block: SQ
          event: 1
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: CP
      min_version: 199
      reason: "CP firmware below version 199 has critical bugs"
      affected_architectures:
        - "gfx908"
        - "gfx90a"
    - firmware_type: SDMA
      min_version: 150
      reason: "SDMA firmware below 150 causes instability"
      affected_architectures:
        - "gfx1030"
)";

    auto result = parse_firmware_restrictions(yaml_content);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2);

    // Check first restriction
    EXPECT_EQ((*result)[0].firmware_type, "CP");
    EXPECT_EQ((*result)[0].min_version, 199);
    EXPECT_EQ((*result)[0].reason, "CP firmware below version 199 has critical bugs");
    ASSERT_EQ((*result)[0].affected_architectures.size(), 2);
    EXPECT_EQ((*result)[0].affected_architectures[0], "gfx908");
    EXPECT_EQ((*result)[0].affected_architectures[1], "gfx90a");

    // Check second restriction
    EXPECT_EQ((*result)[1].firmware_type, "SDMA");
    EXPECT_EQ((*result)[1].min_version, 150);
    EXPECT_EQ((*result)[1].reason, "SDMA firmware below 150 causes instability");
    ASSERT_EQ((*result)[1].affected_architectures.size(), 1);
    EXPECT_EQ((*result)[1].affected_architectures[0], "gfx1030");
}

TEST(FirmwareRestrictions, EmptyRestrictionsList)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions: []
)";

    auto result = parse_firmware_restrictions(yaml_content);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 0);
}

TEST(FirmwareRestrictions, MissingOptionalReason)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: CP
      min_version: 199
      affected_architectures:
        - "gfx908"
)";

    auto result = parse_firmware_restrictions(yaml_content);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ((*result)[0].firmware_type, "CP");
    EXPECT_EQ((*result)[0].min_version, 199);
    EXPECT_EQ((*result)[0].reason, "");  // Should be empty when not provided
    ASSERT_EQ((*result)[0].affected_architectures.size(), 1);
}

TEST(FirmwareRestrictions, EmptyArchitecturesListAppliesToAll)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: MEC
      min_version: 99999
      reason: "Empty list should apply to all architectures"
      affected_architectures: []
)";

    auto result = parse_firmware_restrictions(yaml_content);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ((*result)[0].firmware_type, "MEC");
    EXPECT_EQ((*result)[0].min_version, 99999);
    EXPECT_EQ((*result)[0].affected_architectures.size(), 0);

    // Empty affected_architectures should apply to all agents
    // so check_agent_firmware_restrictions should return false
    // (since no agent has firmware version >= 99999)
    bool check_result = check_agent_firmware_restrictions(yaml_content);
    EXPECT_FALSE(check_result) << "Empty affected_architectures should apply to all architectures";
}

TEST(FirmwareRestrictions, MissingTopLevelKey)
{
    std::string yaml_content = R"(
fw-restriction-schema-version: 1
firmware_restrictions:
  - firmware_type: CP
    min_version: 199
)";

    auto result = parse_firmware_restrictions(yaml_content);

    EXPECT_FALSE(result.has_value());
}

TEST(FirmwareRestrictions, MissingSchemaVersion)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  firmware_restrictions:
    - firmware_type: CP
      min_version: 199
)";

    auto result = parse_firmware_restrictions(yaml_content);

    EXPECT_FALSE(result.has_value());
}

TEST(FirmwareRestrictions, MissingRequiredFirmwareType)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - min_version: 199
      affected_architectures:
        - "gfx908"
)";

    auto result = parse_firmware_restrictions(yaml_content);

    EXPECT_FALSE(result.has_value());
}

TEST(FirmwareRestrictions, MissingRequiredMinVersion)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: CP
      affected_architectures:
        - "gfx908"
)";

    auto result = parse_firmware_restrictions(yaml_content);

    EXPECT_FALSE(result.has_value());
}

TEST(FirmwareRestrictions, InvalidYamlSyntax)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: CP
      min_version: 199
      affected_architectures
        - "gfx908"  # Missing colon after affected_architectures
)";

    auto result = parse_firmware_restrictions(yaml_content);

    EXPECT_FALSE(result.has_value());
}

TEST(FirmwareRestrictions, MultipleRestrictions)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: CP
      min_version: 199
      reason: "Bug fixes"
      affected_architectures:
        - "gfx908"
        - "gfx90a"
        - "gfx940"
    - firmware_type: SDMA
      min_version: 150
      reason: "Stability"
      affected_architectures:
        - "gfx1030"
        - "gfx1100"
    - firmware_type: MEC
      min_version: 210
      reason: "Scheduling"
      affected_architectures:
        - "gfx906"
)";

    auto result = parse_firmware_restrictions(yaml_content);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3);

    // Verify all three restrictions are parsed correctly
    EXPECT_EQ((*result)[0].firmware_type, "CP");
    EXPECT_EQ((*result)[0].affected_architectures.size(), 3);

    EXPECT_EQ((*result)[1].firmware_type, "SDMA");
    EXPECT_EQ((*result)[1].affected_architectures.size(), 2);

    EXPECT_EQ((*result)[2].firmware_type, "MEC");
    EXPECT_EQ((*result)[2].affected_architectures.size(), 1);
}

TEST(FirmwareRestrictions, LargeVersionNumbers)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: TEST
      min_version: 4294967295
      affected_architectures:
        - "test_arch"
)";

    auto result = parse_firmware_restrictions(yaml_content);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ((*result)[0].min_version, 4294967295U);  // Max uint32_t
}

TEST(FirmwareRestrictions, NoRestrictionsField)
{
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
)";

    auto result = parse_firmware_restrictions(yaml_content);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 0);  // Should return empty vector when no restrictions field
}

TEST(FirmwareRestrictions, CheckAgentFirmwareRestrictions)
{
    // Test with empty restrictions (should pass)
    std::string yaml_empty = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions: []
)";

    bool result_empty = check_agent_firmware_restrictions(yaml_empty);
    EXPECT_TRUE(result_empty);

    // Test with high version requirements (may fail depending on actual agent firmware)
    std::string yaml_high_requirements = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: MEC
      min_version: 99999
      reason: "Test high version requirement"
      affected_architectures:
        - "gfx940"
        - "gfx941"
        - "gfx942"
)";

    // This call should use std::call_once and may show warnings if agents don't meet requirements
    check_agent_firmware_restrictions(yaml_high_requirements);
    // Don't assert the result since it depends on actual hardware firmware versions
    // But the function should not crash
}

TEST(FirmwareRestrictions, CheckAgentFirmwareRestrictionsCallOnce)
{
    // Test that multiple calls to check_agent_firmware_restrictions work correctly
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: MEC
      min_version: 1
      reason: "Minimal test"
      affected_architectures:
        - "gfx940"
)";

    // Call multiple times - should return consistent results
    bool result1 = check_agent_firmware_restrictions(yaml_content);
    bool result2 = check_agent_firmware_restrictions(yaml_content);
    bool result3 = check_agent_firmware_restrictions(yaml_content);

    // All calls should return the same result
    EXPECT_EQ(result1, result2);
    EXPECT_EQ(result2, result3);
}

TEST(FirmwareRestrictions, CheckInstalledFirmwareRestrictionsCallOnce)
{
    // Test that std::call_once ensures the installed file check only executes once
    // Call multiple times - should use cached result after first call
    bool result1 = check_installed_firmware_restrictions();
    bool result2 = check_installed_firmware_restrictions();
    bool result3 = check_installed_firmware_restrictions();

    // All calls should return the same result
    EXPECT_EQ(result1, result2);
    EXPECT_EQ(result2, result3);

    // Note: Don't assert specific return value since it depends on:
    // 1. Whether config.yaml file exists
    // 2. Whether actual agents meet the requirements in the file
    // The important thing is that std::call_once works correctly
}

TEST(FirmwareRestrictions, CheckAgentFirmwareRestrictionsWithRealAgents)
{
    // Get real agents from the device to create realistic test
    auto agents = rocprofiler::agent::get_agents();

    // Skip test if no agents found
    if(agents.empty())
    {
        GTEST_SKIP() << "No agents found on device, skipping real agent test";
        return;
    }

    // Find GPU agents and their architectures
    std::vector<std::string> real_architectures;
    for(const auto* agent : agents)
    {
        if(agent && agent->type == ROCPROFILER_AGENT_TYPE_GPU && agent->name)
        {
            real_architectures.push_back(agent->name);
        }
    }

    // Skip test if no GPU agents found
    if(real_architectures.empty())
    {
        GTEST_SKIP() << "No GPU agents found on device, skipping real agent test";
        return;
    }

    // Create YAML with high firmware requirements for actual architectures
    std::string yaml_content = R"(
rocprofiler-sdk:
  counters-schema-version: 1
  counters: []
  fw-restriction-schema-version: 1
  firmware_restrictions:
    - firmware_type: MEC
      min_version: 99999
      reason: "Test with impossibly high requirement to force failure"
      affected_architectures:)";

    // Add real architectures to the YAML
    for(const auto& arch : real_architectures)
    {
        yaml_content += "\n        - \"" + arch + "\"";
    }

    // Test the function - should return false since no real firmware meets 99999
    bool result = check_agent_firmware_restrictions(yaml_content);
    EXPECT_FALSE(result)
        << "Expected firmware check to fail with impossibly high requirements for real agents";
}

}  // namespace
}  // namespace counters
}  // namespace rocprofiler
