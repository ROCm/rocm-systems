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
 * - Test case to verify the printf return value from other process for the compiler option
 * -mprintf-kind=buffered
 * - Fetch the printf content from a process. Compare it with reference string.
 * Test source
 * ------------------------
 * - catch/unit/printf/printfFlagsNonHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */

HIP_TEST_CASE(Unit_Buffered_Printf_Flags) {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  std::string reference =
      std::string(R"here(00000042
-0000042
00000042
0123.456
+0000042
-42
+0000042
)here") +
      std::string("xyzzy") + std::string(3, ' ') + "\n" +
      std::string("-42\n") +
      std::string("00000042") + std::string(8, ' ') + "\n" +
      std::string("        00000042\n");

  hip::SpawnProc proc("printfFlagsNonHost_exe", true);
  REQUIRE(proc.run() == 0);
  REQUIRE(proc.getOutput() == reference);
}


/**
 * End doxygen group PrintfTest.
 * @}
 */
