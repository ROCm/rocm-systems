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

bool hostcallSupported() {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  return pcieAtomic != 0;
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Dispatches a printf kernel from a process holding a large static TLS block, which forces
 *      the hostcall listener's stack request to be smaller than the process' static TLS.
 *      glibc refuses such a request with EINVAL, which used to abort the process.
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
  if (!hostcallSupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPcieAtomicUnsupported);
  }

  hip::SpawnProc proc("hostcallStackSize_exe", true);
  REQUIRE(proc.run() == 0);
  REQUIRE(proc.getOutput() == expectedOutput());
}

/**
 * Test Description
 * ------------------------
 *    - Same dispatch, but the listener's stack request is shrunk below the static TLS of even a
 *      minimal process. This keeps the coverage independent of how much static TLS the runtime's
 *      own dependencies contribute. The request stays above PTHREAD_STACK_MIN so that it is
 *      pthread_create() that rejects it, rather than pthread_attr_setstacksize().
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
  if (!hostcallSupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPcieAtomicUnsupported);
  }

  hip::SpawnProc proc("hostcallStackSize_exe", true);
  proc.setEnv("CQ_THREAD_STACK_SIZE", "32768");
  REQUIRE(proc.run() == 0);
  REQUIRE(proc.getOutput() == expectedOutput());
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
