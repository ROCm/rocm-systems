// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_WSL_WSL_FIXTURE_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_WSL_WSL_FIXTURE_H_

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cstdint>

#include "hsakmt/hsakmt.h"
#include "hsakmt/rocdxg_smi.h"

// Shared by every WSL functional test: GoogleTest requires one fixture class per suite.
// Opens KFD and enumerates GPU nodes at suite startup.
// Live tests call RequireGpu() to skip when no WSL GPU is present.
class WslFunctionalReadOnly : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    struct stat st{};
    if (stat("/dev/dxg", &st) != 0) return;  // not WSL2
    wsl_present_ = true;

    HSAKMT_STATUS r = hsaKmtOpenKFD();
    if (r != HSAKMT_STATUS_SUCCESS && r != HSAKMT_STATUS_KERNEL_ALREADY_OPENED) return;
    kfd_opened_ = true;

    // Enumerate GPU nodes using the same path as WSLGPUBackend::TryPopulate.
    // rocdxg_smi_get_device_count requires the full topology snapshot which
    // hsaKmtAcquireSystemProperties sets up.
    HsaSystemProperties sys_props = {};
    r = hsaKmtAcquireSystemProperties(&sys_props);
    if (r != HSAKMT_STATUS_SUCCESS) return;

    for (uint32_t i = 0; i < sys_props.NumNodes; ++i) {
      HsaNodeProperties node_props = {};
      if (hsaKmtGetNodeProperties(i, &node_props) != HSAKMT_STATUS_SUCCESS) continue;
      if (node_props.NumFComputeCores > 0 && !gpu_ok_) {
        node_id_ = i;
        gpu_ok_ = true;
      }
    }
    // Do NOT call hsaKmtReleaseSystemProperties here — it clears the internal
    // wdevices_ snapshot that rocdxg_smi_* functions need for the process lifetime.
    // It is called in TearDownTestSuite instead.
  }

  static void TearDownTestSuite() {
    if (kfd_opened_) {
      hsaKmtReleaseSystemProperties();
      hsaKmtCloseKFD();
    }
  }

  // Call at the start of live tests that require an actual WSL GPU.
  // Uses GTEST_SKIP_ directly so the macro's `return` exits the test body.
#define RequireGpu()                                                                          \
  do {                                                                                        \
    if (!wsl_present_) GTEST_SKIP() << "No /dev/dxg — not running under WSL2";                \
    if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available (no node with NumFComputeCores > 0)"; \
  } while (0)

  inline static bool wsl_present_ = false;
  inline static bool kfd_opened_ = false;
  inline static bool gpu_ok_ = false;
  inline static uint32_t node_id_ = 0;
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_WSL_WSL_FIXTURE_H_
