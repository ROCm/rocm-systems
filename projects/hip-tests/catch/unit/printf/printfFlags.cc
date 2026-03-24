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
 *    - Sanity test for `printf(format, ...)` to check all format specifier flags.
 *
 * Test source
 * ------------------------
 *    - unit/printf/printfFlags.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Printf_flags_Sanity_Positive) {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  std::string reference(R"here(00000042
-0000042
00000042
0123.456
+0000042
-42
+0000042
xyzzy
-42
 42
00000042
        00000042
052
0x2a
0X2A
42.000000
4.200000e+01
4.200000E+01
42.0000
42.0000
0x1.5p+5
0X1.5P+5
)here");

  hip::SpawnProc proc("printfFlags_exe", true);
  REQUIRE(proc.run() == 0);
  REQUIRE(proc.getOutput() == reference);
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
