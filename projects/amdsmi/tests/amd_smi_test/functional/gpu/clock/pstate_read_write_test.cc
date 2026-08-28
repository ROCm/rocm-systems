// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_soc_pstate / amdsmi_set_soc_pstate.
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
