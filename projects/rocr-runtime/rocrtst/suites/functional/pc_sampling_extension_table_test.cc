/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/functional/pc_sampling_extension_table_test.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ven_amd_pc_sampling.h"

PcSamplingExtensionTableTest::PcSamplingExtensionTableTest() : TestBase() {
  set_title("PC Sampling Extension Table Test");
  set_description("Checks the public PC sampling table query and configuration function pointer.");
}

void PcSamplingExtensionTableTest::Run(void) {
  if (::testing::Test::HasFatalFailure()) return;
  TestBase::Run();

  // Check availability without using the table query under test.
  const hsa_status_t probe = hsa_ven_amd_pcs_iterate_configuration(
      hsa_agent_t{0},
      [](const hsa_ven_amd_pcs_configuration_t*, void*) { return HSA_STATUS_SUCCESS; }, nullptr);
  if (probe == HSA_STATUS_ERROR_NOT_INITIALIZED) {
    rocrtst::SkipCurrentTest("PC sampling extension disabled or not built");
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
}
