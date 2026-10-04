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
 * initializes __managed__ global variable device pointers.
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>

/**
 * Test Description
 * ------------------------
 *    - A fresh child process releases worker threads simultaneously onto the
 *      first kernel launch so they contend on the same initialization generation.
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
