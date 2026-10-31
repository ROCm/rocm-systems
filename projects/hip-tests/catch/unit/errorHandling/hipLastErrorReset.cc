/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <cstdint>
#include <string>

#if HT_AMD
// Not declared in the public HIP headers; reads/clears the separate AMD command
// last-error state (hip_error.cpp), which the listener also resets.
extern "C" hipError_t hipExtGetLastError(void);
#endif

/**
 * Regression for the cross-test last-error reset listener (hip_test_listener.cc).
 *
 * CTest runs each case in its own process, so the harness re-invokes this binary
 * with the two leaf cases together, in declaration order, to drive the check.
 *
 * The leaf cases use raw TEST_CASE with only the hidden tag "[.]" (not
 * HIP_TEST_CASE): LeaveError leaks on purpose, so it must stay out of every
 * normal suite run (including [level_N] selections and ENABLE_YAML_TAGS=OFF),
 * yet a hidden test still runs when the harness selects it by exact name.
 */

// Leaves a sticky last error on purpose and does not consume it; the listener
// must clear it before the next case runs.
TEST_CASE("Unit_hipLastErrorReset_LeaveError", "[.]") {
  void* ptr = nullptr;
  hipError_t err = hipMalloc(&ptr, SIZE_MAX);
  REQUIRE(err != hipSuccess);
}

// Passes only if the listener reset the previous case's leaked error(s).
TEST_CASE("Unit_hipLastErrorReset_SeesCleanState", "[.]") {
  REQUIRE(hipGetLastError() == hipSuccess);
#if HT_AMD
  REQUIRE(hipExtGetLastError() == hipSuccess);
#endif
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
