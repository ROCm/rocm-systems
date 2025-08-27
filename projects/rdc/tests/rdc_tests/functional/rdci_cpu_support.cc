/*
Copyright (c) 2025 - Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @brief Test CPU support infrastructure in RDC
 *
 * These tests validate the CPU support framework without requiring AMD CPU hardware.
 * They focus on infrastructure validation, field definitions, partition string parsing,
 * and entity management rather than actual CPU metric collection.
 */

#include <gtest/gtest.h>
#include <sys/types.h>

#include <string>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcEntityCodec.h"

class RdcCpuSupportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    rdc_status_t result = rdc_init(0x0);
    ASSERT_EQ(result, RDC_ST_OK);

    result = rdc_start_embedded(RDC_OPERATION_MODE_AUTO, &rdc_handle_);
    ASSERT_EQ(result, RDC_ST_OK);
    ASSERT_NE(rdc_handle_, nullptr);
  }

  void TearDown() override {
    if (rdc_handle_ != nullptr) {
      rdc_stop_embedded(rdc_handle_);
      rdc_handle_ = nullptr;
    }
    rdc_shutdown();
  }

  rdc_handle_t rdc_handle_ = nullptr;
};

// Test CPU partition string parsing
TEST_F(RdcCpuSupportTest, CpuPartitionStringParsing) {
  // generate map to test this feature
  std::map<std::pair<uint32_t, uint32_t>, std::string> cpu_partition_string_map = {
      {{0, 0}, "c0.0"}, {{1, 0}, "c1.0"}, {{0, 1}, "c0.1"}, {{2, 0}, "c2.0"}, {{15, 7}, "c15.7"}};

  for (const auto& [indices, partition_str] : cpu_partition_string_map) {
    uint32_t device_index = 255;
    uint32_t instance_index = 255;

    bool parsed = rdc_parse_partition_string(partition_str.c_str(), &device_index, &instance_index);
    EXPECT_TRUE(parsed) << "Should successfully parse CPU partition string: " << partition_str;

    EXPECT_EQ(device_index, indices.first)
        << "Device index should match for partition string: " << partition_str;
    EXPECT_EQ(instance_index, indices.second)
        << "Instance index should match for partition string: " << partition_str;
  }
}

// Test CPU field group creation
TEST_F(RdcCpuSupportTest, CpuFieldGroupCreation) {
  // Test creating field groups with CPU fields
  std::vector<rdc_field_t> cpu_fields = {RDC_FI_CPU_SKT_COUNT, RDC_FI_CPU_MODEL};

  rdc_field_grp_t field_group_id = 0;
  const char* group_name = "cpu_test_group";

  // Create field group with CPU fields
  rdc_status_t result =
      rdc_group_field_create(rdc_handle_, static_cast<uint32_t>(cpu_fields.size()), &cpu_fields[0],
                             group_name, &field_group_id);

  // Should succeed in creating CPU field group
  EXPECT_EQ(result, RDC_ST_OK) << "Should be able to create CPU field group";

  if (result == RDC_ST_OK) {
    // Verify we can get field group info
    rdc_field_group_info_t group_info;
    rdc_status_t info_result = rdc_group_field_get_info(rdc_handle_, field_group_id, &group_info);

    EXPECT_EQ(info_result, RDC_ST_OK) << "Should be able to get field group info";

    if (info_result == RDC_ST_OK) {
      EXPECT_EQ(group_info.count, cpu_fields.size()) << "Field group should have correct count";
      EXPECT_STREQ(group_info.group_name, group_name) << "Field group should have correct name";
    }

    // Clean up
    rdc_group_field_destroy(rdc_handle_, field_group_id);
  }
}

// Test mixed GPU/CPU field groups
TEST_F(RdcCpuSupportTest, MixedGpuCpuFieldGroups) {
  // Test creating field groups with both GPU and CPU fields
  std::vector<rdc_field_t> mixed_fields = {
      RDC_FI_GPU_UTIL,           // GPU field
      RDC_FI_CPU_CCLK_LIMIT,     // CPU field
      RDC_FI_GPU_TEMP,           // GPU field
      RDC_FI_CPU_FCLK_FREQUENCY  // CPU field
  };

  rdc_field_grp_t field_group_id = 0;
  const char* group_name = "mixed_test_group";

  // Create field group with mixed GPU/CPU fields
  rdc_status_t result =
      rdc_group_field_create(rdc_handle_, static_cast<uint32_t>(mixed_fields.size()),
                             &mixed_fields[0], group_name, &field_group_id);

  // Should succeed in creating mixed field group
  EXPECT_EQ(result, RDC_ST_OK) << "Should be able to create mixed GPU/CPU field group";

  if (result == RDC_ST_OK) {
    // Clean up
    rdc_group_field_destroy(rdc_handle_, field_group_id);
  }
}
