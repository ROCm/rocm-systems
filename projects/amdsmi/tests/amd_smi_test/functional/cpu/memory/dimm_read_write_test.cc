// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::AmdsmiStatusIsExpected;
using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_cpu_dimm_sb_reg / amdsmi_set_cpu_dimm_sb_reg.
TEST_F(CpuFunctionalReadWrite, DimmSbReg_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_cpu_dimm_sb_reg");
  const uint32_t kDimmAddr = 0;
  const uint32_t kLid = 0;
  const uint32_t kRegOffset = 0;
  const uint32_t kRegSpace = 0;  // volatile space, so a write survives no reboot
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t initial = 0;
    if (amdsmi_get_cpu_dimm_sb_reg(cpus()[i], kDimmAddr, kLid, kRegOffset, kRegSpace, &initial) !=
        AMDSMI_STATUS_SUCCESS)
      continue;
    // The setter only accepts 8-bit data, so a wider register cannot be put back.
    if (initial > AMDSMI_MAX_SPD_WRITE_DATA) continue;

    uint32_t target = (initial == 0) ? 1u : (initial - 1u);
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_dimm_sb_reg",
                       "cpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err =
        amdsmi_set_cpu_dimm_sb_reg(cpus()[i], kDimmAddr, kLid, kRegOffset, kRegSpace, target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint32_t readback = 0;
      if (amdsmi_get_cpu_dimm_sb_reg(cpus()[i], kDimmAddr, kLid, kRegOffset, kRegSpace,
                                     &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "cpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr =
          amdsmi_set_cpu_dimm_sb_reg(cpus()[i], kDimmAddr, kLid, kRegOffset, kRegSpace, initial);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore dimm sb reg";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_dimm_sb_reg(cpus()[i], kDimmAddr, kLid, kRegOffset, kRegSpace,
                                     &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "cpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}
