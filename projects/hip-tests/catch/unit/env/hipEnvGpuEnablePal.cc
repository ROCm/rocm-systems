/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>
#include <cstdlib>

// These tests are AMD-only (PAL vs ROCr backends)
#if HT_AMD

/**
 * @addtogroup hipEnvGpuEnablePal hipEnvGpuEnablePal
 * @{
 * @ingroup EnvironmentTest
 * Test GPU_ENABLE_PAL environment variable behavior with platform-specific defaults.
 * AMD-only: Tests ROCr vs PAL backend selection.
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

// Calls hipGetDeviceCount. Parent cases run this by name.
TEST_CASE("Unit_hipEnvGpuEnablePal_Probe", "[.]") {
  int deviceCount = 0;
  REQUIRE(hipGetDeviceCount(&deviceCount) == hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL unset (default) uses platform default
 *  - Windows: PAL (default)
 *  - Linux: ROCr (default)
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_Default_UsesPlatformDefault) {
  hip::SpawnProc proc(getSelfExePath(), true, /*captureStderr=*/true);
  proc.setEnv("AMD_LOG_LEVEL", "7");
  proc.clearEnv("GPU_ENABLE_PAL");

  int result = proc.run("Unit_hipEnvGpuEnablePal_Probe");
  std::string output = proc.getOutput();

  REQUIRE(result == 0);  // HIP should initialize successfully

  // Debug: print what we received
  INFO("Received output length: " << output.length());
  INFO("Output content: " << output);

  // Verify actual runtime path from logs
  bool palInitialized = (output.find("] PAL backend initialized") != std::string::npos);
  bool rocrInitialized = (output.find("] ROCr backend initialized") != std::string::npos);

#if defined(_WIN32)
  // Windows: default should use PAL
  REQUIRE(palInitialized == true);
  REQUIRE(rocrInitialized == false);
#else
  // Linux: default should use ROCr
  REQUIRE(rocrInitialized == true);
  REQUIRE(palInitialized == false);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates that GPU_ENABLE_PAL empty string uses platform default
 *  - Windows: PAL (default)
 *  - Linux: ROCr (default)
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_EmptyString_UsesPlatformDefault) {
  hip::SpawnProc proc(getSelfExePath(), true, /*captureStderr=*/true);
  proc.setEnv("AMD_LOG_LEVEL", "7");
  proc.setEnv("GPU_ENABLE_PAL", "");

  int result = proc.run("Unit_hipEnvGpuEnablePal_Probe");
  std::string output = proc.getOutput();

  REQUIRE(result == 0);  // HIP should initialize successfully
  // Debug: print what we received
  INFO("Received output length: " << output.length());
  INFO("Output content: " << output);
  // Verify actual runtime path from logs
  // Look for the actual device.cpp log line pattern
  bool palInitialized = (output.find("] PAL backend initialized") != std::string::npos);
  bool rocrInitialized = (output.find("] ROCr backend initialized") != std::string::npos);

#if defined(_WIN32)
  // Windows: empty string should use PAL (platform default)
  REQUIRE(palInitialized == true);
  REQUIRE(rocrInitialized == false);
#else
  // Linux: empty string should use ROCr (platform default)
  REQUIRE(rocrInitialized == true);
  REQUIRE(palInitialized == false);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates GPU_ENABLE_PAL explicit values (0, 1, 2) work correctly
 *  - "0" forces ROCr on all platforms
 *  - "1" forces PAL on all platforms
 *  - "2" allows auto-selection
 * Test source
 * ------------------------
 *  - unit/env/hipEnvGpuEnablePal.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipEnvGpuEnablePal_ExplicitValues_WorkCorrectly) {
  // Test GPU_ENABLE_PAL="0" -> ROCr
  // AIRUNTIME-2370. fails rocr init and conitinues to init PAL
  // Once the issue is fixed this can be enabled
  {
    hip::SpawnProc proc0(getSelfExePath(), true, /*captureStderr=*/true);
    proc0.setEnv("AMD_LOG_LEVEL", "7");
    proc0.setEnv("GPU_ENABLE_PAL", "0");
    int result0 = proc0.run("Unit_hipEnvGpuEnablePal_Probe");
    std::string output0 = proc0.getOutput();
    // Debug: print what we received
    INFO("Received output length: " << output0.length());
    INFO("Output content: " << output0);
    REQUIRE(result0 == 0);  // HIP should initialize successfully

    bool rocrInit0 = (output0.find("] ROCr backend initialized") != std::string::npos);
    bool palInit0 = (output0.find("] PAL backend initialized") != std::string::npos);
    REQUIRE(rocrInit0 == true);
    REQUIRE(palInit0 == false); 
  }

  // Test GPU_ENABLE_PAL="1" -> PAL
  {
    hip::SpawnProc proc1(getSelfExePath(), true, /*captureStderr=*/true);
    proc1.setEnv("AMD_LOG_LEVEL", "7");
    proc1.setEnv("GPU_ENABLE_PAL", "1");
    int result1 = proc1.run("Unit_hipEnvGpuEnablePal_Probe");
    std::string output1 = proc1.getOutput();
    // Debug: print what we received
    INFO("Received output length: " << output1.length());
    INFO("Output content: " << output1);
    bool palInit1 = (output1.find("] PAL backend initialized") != std::string::npos);
    bool rocrInit1 = (output1.find("] ROCr backend initialized") != std::string::npos);
    #if defined(_WIN32)
      REQUIRE(result1 == 0);  // HIP should initialize successfully
      REQUIRE(palInit1 == true);
      REQUIRE(rocrInit1 == false);
    #else
      REQUIRE(result1 != 0);  // HIP init fails
      // Linux: PAL not supported . So init fails
      REQUIRE(palInit1 == false);
      REQUIRE(rocrInit1 == false);
    #endif
  }

  // Test GPU_ENABLE_PAL="2" -> Auto-select
  {
    hip::SpawnProc proc2(getSelfExePath(), true, /*captureStderr=*/true);
    proc2.setEnv("AMD_LOG_LEVEL", "7");
    proc2.setEnv("GPU_ENABLE_PAL", "2");
    int result2 = proc2.run("Unit_hipEnvGpuEnablePal_Probe");
    std::string output2 = proc2.getOutput();
    // Debug: print what we received
    INFO("Received output length: " << output2.length());
    INFO("Output content: " << output2);
    REQUIRE(result2 == 0);  // HIP should initialize successfully

    // Auto-select initializes both backends
    bool palInit2 = (output2.find("] PAL backend initialized") != std::string::npos);
    bool rocrInit2 = (output2.find("] ROCr backend initialized") != std::string::npos);
    #if defined(_WIN32)
      REQUIRE(palInit2 == true);
      REQUIRE(rocrInit2 == true);
    #else
      // Linux: PAL not supported . rocr should init correctly
      REQUIRE(palInit2 == false);
      REQUIRE(rocrInit2 == true);
    #endif
  }
}

/**
 * End doxygen group EnvironmentTest.
 * @}
 */

#endif  // HT_AMD
