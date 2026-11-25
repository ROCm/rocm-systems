/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>
#include <cstdlib>

/**
 * @addtogroup hipGetDeviceCount hipGetDeviceCount
 * @{
 * @ingroup DeviceTest
 * `hipGetDeviceCount(int* count)` -
 * Return number of compute-capable devices.
 */

/**
 * Test Description
 * ------------------------
 *  - Passes invalid pointer as output parameter for device count - `nullptr`
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceCount.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGetDeviceCount_NegTst) {
  REQUIRE_FALSE(hipGetDeviceCount(nullptr) == hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Validates correct functionality when the device visibility
 *    environment variables are set. Uses unit/device/hipDeviceCount_exe.cc
 *    to set visibility.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceCount.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGetDeviceCount_HideDevices) {
  int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  for (int i = deviceCount; i >= 1; i--) {
    std::string visibleStr;
    for (int j = 0; j < i; j++) {  // Generate a string which has first i devices
      visibleStr += std::to_string(j);
      if (j != (i - 1)) {
        visibleStr += ",";
      }
    }

    hip::SpawnProc proc("getDeviceCount", true);
    INFO("Output from process : " << proc.getOutput());
    REQUIRE(proc.run(visibleStr) == i);
  }
}

#if __linux__
/**
 * Test Description
 * ------------------------
 *  - Runs unit/device/hipCpuAffinity_exe.cc, which calls hipGetDeviceCount and checks that the
 *    CPU mask of the calling thread is unchanged. Launchers enumerate devices and then fork
 *    workers that inherit this mask, so narrowing it to the GPU's NUMA node breaks per-rank
 *    CPU binding (AMD_CPU_AFFINITY is off by default).
 *  - Can only fail on hosts with more than one NUMA node, where the runtime is able to read the
 *    CPU mask, and where the cpuset holds every online CPU (skipped otherwise).
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceCount.cc
 * Test requirements
 * ------------------------
 *  - Linux
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hipGetDeviceCount_KeepsCpuAffinity) {
  // Exit codes of hipCpuAffinity_exe.cc
  constexpr int kMaskKept = 0;
  constexpr int kMaskRestricted = 2;

  const char* affinity = std::getenv("AMD_CPU_AFFINITY");
  if (affinity != nullptr && std::string(affinity) != "0") {
    HIP_SKIP_TEST("AMD_CPU_AFFINITY is enabled");
  }
  hip::SpawnProc proc("hipCpuAffinity", true);
  const int result = proc.run();
  INFO("Output from process : " << proc.getOutput());
  if (result == kMaskRestricted) {
    HIP_SKIP_TEST("cpuset doesn't include every online CPU");
  }
  REQUIRE(result == kMaskKept);
}
#endif

/**
 * End doxygen group DeviceTest.
 * @}
 */
