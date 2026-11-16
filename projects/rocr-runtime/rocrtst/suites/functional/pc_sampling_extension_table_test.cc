/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>
#include <string>

#include "common/platform_filter.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ven_amd_pc_sampling.h"

TEST(rocrtstFunc, PcSamplingExtensionTable) {
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_init());

  // Check availability without using the table query under test.
  const hsa_status_t probe = hsa_ven_amd_pcs_iterate_configuration(
      hsa_agent_t{0},
      [](const hsa_ven_amd_pcs_configuration_t*, void*) { return HSA_STATUS_SUCCESS; }, nullptr);
  if (probe == HSA_STATUS_ERROR_NOT_INITIALIZED) {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string name = std::string(info->test_case_name()) + "." + info->name();
    const std::string reason = "PC sampling extension disabled or not built";
    std::cout << "[ SKIPPED ] " << reason << '\n';
    rocrtst::SkippedTestTracker::getInstance().recordSkip(name, reason);
  } else {
    EXPECT_EQ(HSA_STATUS_ERROR_INVALID_AGENT, probe);
    hsa_ven_amd_pc_sampling_1_00_pfn_t table{};
    const hsa_status_t status = hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_PC_SAMPLING,
                                                                     1, sizeof(table), &table);
    EXPECT_EQ(HSA_STATUS_SUCCESS, status);
    if (status == HSA_STATUS_SUCCESS) {
      EXPECT_EQ(&hsa_ven_amd_pcs_iterate_configuration,
                table.hsa_ven_amd_pcs_iterate_configuration);
    }
  }

  EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_shut_down());
}
