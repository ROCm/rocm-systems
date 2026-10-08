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
// The child reproduces the defect nine times in ten on a loaded core, so a few runs make a
// regression practically impossible to miss while costing a second or so each when it is absent.
constexpr int kRuns = 5;

// The child arms a watchdog before it exits and leaves with this code if teardown never finishes.
constexpr int kChildTimedOut = 66;
}  // namespace

/**
 * Test Description
 * ------------------------
 *    - Starts and stops the hostcall listener in a child process with the listener thread still
 *      on its way to its entry point, which is the state the runtime used to mishandle. It could
 *      be left either believing a listener was running when none was, spinning forever in a static
 *      destructor, or believing none was running while one still was, finalizing underneath a live
 *      thread. Both show up only as the process exits, so the child is a separate executable that
 *      is required to exit promptly and cleanly.
 *
 * Test source
 * ------------------------
 *    - unit/printf/hostcallShutdown.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Printf_HostcallShutdown_Positive) {
  CHECK_PCIE_ATOMIC_SUPPORT

  for (int run = 0; run < kRuns; ++run) {
    hip::SpawnProc proc("hostcallShutdown_exe");
    const int result = proc.run();
    INFO("run " << run << " exited with " << result
                << (result == kChildTimedOut ? " (watchdog fired: hung while exiting)" : ""));
    REQUIRE(result == 0);
  }
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
