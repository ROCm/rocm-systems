/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include "../printf/printf_common.h"  // NOLINT

#include <string>

/**
 * @addtogroup hipModuleLaunchKernel hipModuleLaunchKernel
 * @{
 * @ingroup ModuleTest
 */

/**
 * Test Description
 * ------------------------
 * - Loads a device-only code object whose amdhsa.printf metadata lists the same format string
 *   twice (same hash), launches its kernel and checks that both buffered printf lines are
 *   printed. The runtime used to reject the repeated hash as a collision and drop the output.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModulePrintf.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipModuleLaunchKernel_Printf_DuplicateFormatString) {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPcieAtomicUnsupported);
  }
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoad(&module, "printf_duplicate_module.code"));
  HIP_CHECK(hipModuleGetFunction(&function, module, "printf_duplicate_format_string"));

  int* count_d = nullptr;
  int count[2] = {-1, -1};
  HIP_CHECK(hipMalloc(&count_d, sizeof(count)));
  void* args[] = {&count_d};

  CaptureStream captured(stdout);
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  auto captured_data = captured.getCapturedData();
  std::string device_output = captured.gulp(captured_data);
  HIP_CHECK(hipMemcpy(count, count_d, sizeof(count), hipMemcpyDeviceToHost));

  REQUIRE(count[0] == 0);
  REQUIRE(count[1] == 0);
  REQUIRE(device_output == "Duplicate format string: 1\nDuplicate format string: 2\n");

  HIP_CHECK(hipFree(count_d));
  HIP_CHECK(hipModuleUnload(module));
}

/**
 * End doxygen group ModuleTest.
 * @}
 */
