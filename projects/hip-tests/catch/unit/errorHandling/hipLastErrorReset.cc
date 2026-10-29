/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <cstdint>
#include <string>

/**
 * Regression for the cross-test last-error reset listener (hip_test_listener.cc).
 *
 * The two leaf cases only exercise the listener when run together in one process
 * in declaration order. CTest (catch_discover_tests) runs each case in its own
 * process, so the harness case re-invokes this binary with both of them to drive
 * the actual check -- in normal and in strict mode.
 */

// Leaves a sticky last error on purpose and does not consume it; the listener
// must clear it before the next case runs.
HIP_TEST_CASE(Unit_hipLastErrorReset_LeaveError) {
  void* ptr = nullptr;
  hipError_t err = hipMalloc(&ptr, SIZE_MAX);
  REQUIRE(err != hipSuccess);
}

// Passes only if the listener reset the previous case's leaked error.
HIP_TEST_CASE(Unit_hipLastErrorReset_SeesCleanState) {
  REQUIRE(hipGetLastError() == hipSuccess);
}

// Runs the pair above together, in declaration order, in a child process.
HIP_TEST_CASE(Unit_hipLastErrorReset_Harness) {
  const std::string args =
      "Unit_hipLastErrorReset_LeaveError,Unit_hipLastErrorReset_SeesCleanState --order decl";

  // Normal run: both cases pass (exit 0) and the leak is reported on stderr.
  // Clear strict mode explicitly so an inherited HIP_TEST_STRICT_LAST_ERROR
  // from the environment does not turn this into a failing run.
  {
    hip::SpawnProc proc(getSelfExePath(), true, true);
    proc.setEnv("HIP_TEST_STRICT_LAST_ERROR", "");
    int exitCode = proc.run(args);
    const std::string out = proc.getOutput();
    INFO("child output:\n" << out);
    REQUIRE(exitCode == 0);
    REQUIRE(out.find("[HIP last-error]") != std::string::npos);
    REQUIRE(out.find("Unit_hipLastErrorReset_LeaveError") != std::string::npos);
  }

  // Strict mode: the run fails (non-zero exit) and names the culprit.
  {
    hip::SpawnProc proc(getSelfExePath(), true, true);
    proc.setEnv("HIP_TEST_STRICT_LAST_ERROR", "1");
    int exitCode = proc.run(args);
    const std::string out = proc.getOutput();
    INFO("child output:\n" << out);
    REQUIRE(exitCode != 0);
    REQUIRE(out.find("Unit_hipLastErrorReset_LeaveError") != std::string::npos);
  }
}
