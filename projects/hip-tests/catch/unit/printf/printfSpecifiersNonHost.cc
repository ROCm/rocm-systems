/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>


/**
 * @addtogroup printf
 * @{
 * @ingroup PrintfTest
 * `int printf()` -
 * Method to print the content on output device.
 */
/**
 * Test Description
 * ------------------------
 * - Test case to verify the different format specifiers. Test case should compile with the compiler
 * option -mprintf-kind=buffered
 * - Fetch the printf content from a process which will verify format specifier. Compare it with
 * reference string. Test source
 * ------------------------
 * - catch/unit/printf/printfSpecifiersNonHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */

TEST_CASE(Unit_Buffered_Printf_Specifier) {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
#ifdef __HIP_PLATFORM_NVIDIA__
  std::string reference(R"here(xyzzy
%
hello % world
%s
%s0xf01dab1eca55e77e
%cxyzzy
sep
-42
42
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
x
(null)
(nil)
3.14159000    hello 0xf01dab1eca55e77e
)here");
#elif !defined(_WIN32)
  std::string reference(R"here(xyzzy
%
hello % world
%s
%s0xf01dab1eca55e77e
%cxyzzy
sep
-42
42
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
x

(nil)
3.14159000    hello 0xf01dab1eca55e77e
)here");
#else
  std::string reference(R"here(xyzzy
%
hello % world
%s
%sF01DAB1ECA55E77E
%cxyzzy
sep
-42
42
123.456000
-123.456000
-1.234560e+02
1.234560E+02
123.456
-123.456
x

0000000000000000
3.14159000    hello F01DAB1ECA55E77E
)here");
#endif

  hip::SpawnProc proc("printfSpecifiersNonHost_exe", true);
  REQUIRE(0 == proc.run());
  REQUIRE(proc.getOutput() == reference);
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
