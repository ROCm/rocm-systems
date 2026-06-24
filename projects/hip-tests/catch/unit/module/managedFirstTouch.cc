/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup StatCO StatCO
 * @{
 * @ingroup ModuleTest
 * Validates thread-safety of StatCO::InitManagedVarDevicePtr, which lazily
 * initialises __managed__ global variable device pointers on the first kernel
 * launch and is protected by a per-device atomic flag with acquire/release
 * double-checked fast path.
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>

/**
 * Test Description
 * ------------------------
 *    - Spawn a fresh child process (managedFirstTouch_exe) that releases
 *      N worker threads simultaneously through a pthread_barrier_t onto the
 *      very first kernel launch of that process.  All threads touch the same
 *      __managed__ global variable through ihipModuleLaunchKernel, exercising
 *      the concurrent first-touch path of StatCO::InitManagedVarDevicePtr.
 *      The child process validates that the managed variable is correctly
 *      initialised and holds the expected value after the burst completes.
 *      The test must run in a child process because the per-device
 *      initialisation flag is process-global one-shot state: any earlier
 *      kernel launch in the Catch2 runner would set the flag before this test
 *      case executes, permanently closing the race window.
 * Test source
 * ------------------------
 *    - catch/unit/module/managedFirstTouch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_StatCO_ManagedVarConcurrentFirstTouch_InChildProcess) {
  hip::SpawnProc proc("managedFirstTouch_exe", true);
  REQUIRE(proc.run() == 0);
}
