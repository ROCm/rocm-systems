/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

/**
 * @addtogroup printf printf
 * @{
 * @ingroup PrintfTest
 * `int printf()` -
 * Method to print the content on output device.
 */

namespace {
constexpr int kGreetLines = 4;

std::string expectedOutput() {
  std::string reference;
  for (int i = 0; i < kGreetLines; ++i) {
    reference += "hostcall serviced\n";
  }
  return reference;
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Dispatches a printf kernel from a process holding a large static TLS block, which a
 *      listener asking for a stack smaller than that footprint cannot be created in.
 *
 * Test source
 * ------------------------
 *    - unit/printf/hostcallStackSize.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Printf_HostcallLargeStaticTls_Positive) {
  CHECK_PCIE_ATOMIC_SUPPORT

  hip::SpawnProc proc("hostcallStackSize_exe", true);
  REQUIRE(proc.run() == 0);
  REQUIRE(proc.getOutput() == expectedOutput());
}

/**
 * Test Description
 * ------------------------
 *    - Same dispatch with CQ_THREAD_STACK_SIZE undersized, to confirm the listener no longer
 *      takes its stack size from that flag. The value stays above PTHREAD_STACK_MIN so that a
 *      runtime still honouring it fails in pthread_create(), not in setstacksize().
 *
 * Test source
 * ------------------------
 *    - unit/printf/hostcallStackSize.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Printf_HostcallUndersizedStackRequest_Positive) {
  CHECK_PCIE_ATOMIC_SUPPORT

  hip::SpawnProc proc("hostcallStackSize_exe", true);
  proc.setEnv("CQ_THREAD_STACK_SIZE", "32768");
  REQUIRE(proc.run() == 0);
  REQUIRE(proc.getOutput() == expectedOutput());
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
