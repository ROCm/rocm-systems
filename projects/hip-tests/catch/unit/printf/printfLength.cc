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

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `printf(format, ...)` to check all format specifier length sub-specifiers.
 *
 * Test source
 * ------------------------
 *    - unit/printf/printfLength.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Printf_length_Sanity_Positive) {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
#if HT_NVIDIA
  std::string reference(R"here(-42 -42
-42 -42
-42 -42
42 52
42 52
42 52
2a 2A
2a 2A
2a 2A
123.456000
x
)here");
#else
  std::string reference(R"here(-42 -42
-42 -42
-42 -42
42 52
42 52
42 52
2a 2A
2a 2A
2a 2A
123.456000
x
123.456000
-42 -42
-42 -42
-42 -42
0 0
42 52
42 52
42 52
0 0
)here");
#endif

  hip::SpawnProc proc("printfLength_exe", true);
  REQUIRE(0 == proc.run());
  REQUIRE(proc.getOutput() == reference);
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
