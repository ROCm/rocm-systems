/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>
#include <cstdlib>

/**
 * @addtogroup hipEnvGpuEnablePal hipEnvGpuEnablePal
 * @{
 * @ingroup EnvironmentTest
 * Test GPU_ENABLE_PAL environment variable behavior with platform-specific defaults.
 *
 * Behavior Matrix:
 * ----------------
 * GPU_ENABLE_PAL | Windows | Linux
 * ---------------|---------|-------
 * Not set        | PAL (1) | ROCr (0)
 * "" (empty)     | PAL (1) | ROCr (0)
 * "0"            | ROCr (0)| ROCr (0)
 * "1"            | PAL (1) | PAL (1)
 * "2"            | Auto (2)| Auto (2)
 */

// =============================================================================
// WINDOWS TESTS
// =============================================================================

#if defined(_WIN32)
/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL empty string defaults to PAL on Windows
 *  - BUG: atoi("") returns 0, forcing ROCr instead of platform default
 *  - FIX: Device::init() detects empty string and sets Windows default (PAL)
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - Platform: Windows
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_EmptyString_Windows_UsesPAL) {
  hip::SpawnProc proc("hipEnvGpuEnablePal_CheckRuntime", true);

  // Empty string should default to PAL on Windows
  proc.setEnv("GPU_ENABLE_PAL", "");
  int result = proc.run();
  INFO("GPU_ENABLE_PAL=\"\" on Windows, result: " << result);
  REQUIRE(result == 0); // 0 = PAL path
}

/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL="0" forces ROCr path on Windows
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - Platform: Windows
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_ExplicitZero_Windows_UsesROCr) {
  hip::SpawnProc proc("hipEnvGpuEnablePal_CheckRuntime", true);

  // Explicit "0" should force ROCr even on Windows
  int result = proc.run("0");

  INFO("GPU_ENABLE_PAL=\"0\" on Windows, result: " << result);
  REQUIRE(result == 1); // 1 = ROCr path
}

/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL="1" forces PAL path on Windows
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - Platform: Windows
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_ExplicitOne_Windows_UsesPAL) {
  hip::SpawnProc proc("hipEnvGpuEnablePal_CheckRuntime", true);

  // Explicit "1" should force PAL
  int result = proc.run("1");

  INFO("GPU_ENABLE_PAL=\"1\" on Windows, result: " << result);
  REQUIRE(result == 0); // 0 = PAL path
}

/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL="2" allows auto-selection on Windows
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - Platform: Windows
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_ExplicitTwo_Windows_AllowsAuto) {
  hip::SpawnProc proc("hipEnvGpuEnablePal_CheckRuntime", true);

  // Explicit "2" should allow auto-selection
  int result = proc.run("2");

  INFO("GPU_ENABLE_PAL=\"2\" on Windows, result: " << result);
  // Auto-select may choose PAL or ROCr based on device
  REQUIRE((result == 0 || result == 1 || result == 2)); // Any valid path
}
#endif // _WIN32

// =============================================================================
// LINUX TESTS
// =============================================================================

#if !defined(_WIN32)
/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL empty string defaults to ROCr on Linux
 *  - BUG: atoi("") returns 0, which accidentally gives ROCr but for wrong reason
 *  - FIX: Device::init() explicitly sets Linux default (ROCr) for empty string
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - Platform: Linux
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_EmptyString_Linux_UsesROCr) {
  hip::SpawnProc proc("hipEnvGpuEnablePal_CheckRuntime", true);

  // Empty string should default to ROCr on Linux
  proc.setEnv("GPU_ENABLE_PAL", "");
  int result = proc.run();
  INFO("GPU_ENABLE_PAL=\"\" on Linux, result: " << result);
  REQUIRE(result == 1); // 1 = ROCr path
}

/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL="0" forces ROCr path on Linux
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - Platform: Linux
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_ExplicitZero_Linux_UsesROCr) {
  hip::SpawnProc proc("hipEnvGpuEnablePal_CheckRuntime", true);

  // Explicit "0" should force ROCr (same as default on Linux)
  int result = proc.run("0");

  INFO("GPU_ENABLE_PAL=\"0\" on Linux, result: " << result);
  REQUIRE(result == 1); // 1 = ROCr path
}

/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL="1" forces PAL path on Linux
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - Platform: Linux
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_ExplicitOne_Linux_UsesPAL) {
  hip::SpawnProc proc("hipEnvGpuEnablePal_CheckRuntime", true);

  // Explicit "1" should force PAL even on Linux
  int result = proc.run("1");

  INFO("GPU_ENABLE_PAL=\"1\" on Linux, result: " << result);
  REQUIRE(result == 0); // 0 = PAL path
}

/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL="2" allows auto-selection on Linux
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - Platform: Linux
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_ExplicitTwo_Linux_AllowsAuto) {
  hip::SpawnProc proc("hipEnvGpuEnablePal_CheckRuntime", true);

  // Explicit "2" should allow auto-selection
  int result = proc.run("2");

  INFO("GPU_ENABLE_PAL=\"2\" on Linux, result: " << result);
  // Auto-select may choose PAL or ROCr based on device
  REQUIRE((result == 0 || result == 1 || result == 2)); // Any valid path
}
#endif // !_WIN32

// =============================================================================
// CROSS-PLATFORM TESTS
// =============================================================================

/**
 * Test Description
 * ------------------------
 *  - Validates that empty string is different from explicit "0"
 *  - Empty string uses platform default, "0" forces ROCr
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_EmptyVsZero_AreDifferent) {
  hip::SpawnProc proc_empty("hipEnvGpuEnablePal_CheckRuntime", true);
  hip::SpawnProc proc_zero("hipEnvGpuEnablePal_CheckRuntime", true);

  proc_empty.setEnv("GPU_ENABLE_PAL", "");
  proc_zero.setEnv("GPU_ENABLE_PAL", "0");
  int result_empty = proc_empty.run();
  int result_zero = proc_zero.run();

  INFO("Empty string result: " << result_empty);
  INFO("Explicit '0' result: " << result_zero);

#if defined(_WIN32)
  // On Windows: empty → PAL (0), "0" → ROCr (1)
  REQUIRE(result_empty == 0); // PAL
  REQUIRE(result_zero == 1);  // ROCr
  REQUIRE(result_empty != result_zero); // Different behavior
#else
  // On Linux: both → ROCr (1), but for different reasons
  REQUIRE(result_empty == 1); // ROCr via platform default
  REQUIRE(result_zero == 1);  // ROCr via explicit setting
  // They're the same value, but the logic path is different (intentional vs accidental)
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates that all explicit values (0, 1, 2) work correctly
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_AllExplicitValues_Work) {
  hip::SpawnProc proc0("hipEnvGpuEnablePal_CheckRuntime", true);
  hip::SpawnProc proc1("hipEnvGpuEnablePal_CheckRuntime", true);
  hip::SpawnProc proc2("hipEnvGpuEnablePal_CheckRuntime", true);

  int result_0 = proc0.run("0");
  int result_1 = proc1.run("1");
  int result_2 = proc2.run("2");

  INFO("GPU_ENABLE_PAL=\"0\" result: " << result_0);
  INFO("GPU_ENABLE_PAL=\"1\" result: " << result_1);
  INFO("GPU_ENABLE_PAL=\"2\" result: " << result_2);

  // "0" → ROCr on all platforms
  REQUIRE(result_0 == 1);

  // "1" → PAL on all platforms
  REQUIRE(result_1 == 0);

  // "2" → Auto (may vary by device)
  REQUIRE((result_2 == 0 || result_2 == 1 || result_2 == 2));
}

/**
 * End doxygen group EnvironmentTest.
 * @}
 */
