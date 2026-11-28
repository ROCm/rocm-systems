/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <filesystem>

/**
 * Test Description
 * ------------------------
 *  - Runs unit/dynamicLoading/hipDispatchTableReentry_exe.cc with
 *    unit/dynamicLoading/dispatchTableReentryTool.cc loaded as profiler library through
 *    rocprofiler-register (ROCPROFILER_REGISTER_LIBRARY). The tool calls HIP while the runtime
 *    registers its dispatch tables, as profiler and MPI tool libraries do when they initialize.
 *    That call must neither hang nor abort on a dispatch table that is still being set up.
 *    Skipped when rocprofiler-register doesn't load the tool.
 * Test source
 * ------------------------
 *  - unit/dynamicLoading/hipDispatchTableReentry.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hipDispatchTable_ToolCallsHipDuringRegistration) {
  // Exit codes of hipDispatchTableReentry_exe.cc
  constexpr int kPassed = 0;
  constexpr int kToolNotLoaded = 2;

  const auto tool = std::filesystem::path(TestContext::get().currentPath()) /
                    "libdispatchTableReentryTool.so";
  hip::SpawnProc proc("hipDispatchTableReentry_exe", true, true);
  proc.setEnv("ROCPROFILER_REGISTER_LIBRARY", tool.string());
  const int result = proc.run();
  INFO("Output from process : " << proc.getOutput());
  if (result == kToolNotLoaded) {
    HIP_SKIP_TEST("rocprofiler-register did not load the tool library");
  }
  REQUIRE(result == kPassed);
}
