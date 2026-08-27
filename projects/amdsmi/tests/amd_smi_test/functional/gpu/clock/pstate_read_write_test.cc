/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_soc_pstate / amdsmi_set_soc_pstate.
TEST_F(GpuFunctionalReadWrite, SetSocPstate_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_soc_pstate", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_soc_pstate(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadWrite, SocPstate_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_soc_pstate");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dpm_policy_t policy;
    memset(&policy, 0, sizeof(policy));
    if (amdsmi_get_soc_pstate(gpus()[i], &policy) != AMDSMI_STATUS_SUCCESS) continue;
    if (policy.num_supported == 0 || policy.current >= policy.num_supported) continue;

    uint32_t initial_id = policy.policies[policy.current].policy_id;
    uint32_t target_id = initial_id;
    bool found = false;
    for (uint32_t j = 0; j < policy.num_supported; ++j) {
      if (policy.policies[j].policy_id != initial_id) {
        target_id = policy.policies[j].policy_id;
        found = true;
        break;
      }
    }
    if (!found) continue;

    DISPLAY_AMDSMI_API("amdsmi_set_soc_pstate",
                       "gpu=" + std::to_string(i) + " set=" + std::to_string(target_id), kVerbose);
    amdsmi_status_t err = amdsmi_set_soc_pstate(gpus()[i], target_id);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::test::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      amdsmi_dpm_policy_t readback;
      memset(&readback, 0, sizeof(readback));
      if (amdsmi_get_soc_pstate(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS &&
          readback.current < readback.num_supported) {
        EXPECT_EQ(readback.policies[readback.current].policy_id, target_id)
            << "gpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_soc_pstate(gpus()[i], initial_id);
      DISPLAY_AMDSMI_API("amdsmi_set_soc_pstate",
                         "gpu=" + std::to_string(i) + " restore=" + std::to_string(initial_id),
                         kVerbose);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "gpu=" << i << " failed to restore soc pstate";
    }
  }
  col.ExpectNoFailures();
}

// amdsmi_get_xgmi_plpd / amdsmi_set_xgmi_plpd.
TEST_F(GpuFunctionalReadWrite, SetXgmiPlpd_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_xgmi_plpd", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_xgmi_plpd(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadWrite, XgmiPlpd_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_xgmi_plpd");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dpm_policy_t policy;
    memset(&policy, 0, sizeof(policy));
    if (amdsmi_get_xgmi_plpd(gpus()[i], &policy) != AMDSMI_STATUS_SUCCESS) continue;
    if (policy.num_supported == 0 || policy.current >= policy.num_supported) continue;

    uint32_t initial_id = policy.policies[policy.current].policy_id;
    uint32_t target_id = initial_id;
    bool found = false;
    for (uint32_t j = 0; j < policy.num_supported; ++j) {
      if (policy.policies[j].policy_id != initial_id) {
        target_id = policy.policies[j].policy_id;
        found = true;
        break;
      }
    }
    if (!found) continue;

    DISPLAY_AMDSMI_API("amdsmi_set_xgmi_plpd",
                       "gpu=" + std::to_string(i) + " set=" + std::to_string(target_id), kVerbose);
    amdsmi_status_t err = amdsmi_set_xgmi_plpd(gpus()[i], target_id);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::test::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      amdsmi_dpm_policy_t readback;
      memset(&readback, 0, sizeof(readback));
      if (amdsmi_get_xgmi_plpd(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS &&
          readback.current < readback.num_supported) {
        EXPECT_EQ(readback.policies[readback.current].policy_id, target_id)
            << "gpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_xgmi_plpd(gpus()[i], initial_id);
      DISPLAY_AMDSMI_API("amdsmi_set_xgmi_plpd",
                         "gpu=" + std::to_string(i) + " restore=" + std::to_string(initial_id),
                         kVerbose);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "gpu=" << i << " failed to restore xgmi plpd";
    }
  }
  col.ExpectNoFailures();
}
