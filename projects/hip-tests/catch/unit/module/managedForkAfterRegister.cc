/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/**
 * @addtogroup StatCO StatCO
 * @{
 * @ingroup ModuleTest
 * Covers a program that forks after registering a __managed__ variable and before its
 * first HIP API call.
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>

/**
 * Test Description
 * ------------------------
 *    - A program that declares a __managed__ variable must still be able to fork before
 *      its first HIP API call and have the child use the GPU.
 *
 *    - This tests the deferred initialization feature in CLR.
 *
 * Test source
 * ------------------------
 *    - catch/unit/module/managedForkAfterRegister.cc
 * Test requirements
 * ------------------------
 *    - Linux
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_StatCO_ManagedVarForkAfterRegister_InChildProcess) {
  CHECK_MANAGED_MEMORY_SUPPORT

  hip::SpawnProc proc("managedForkAfterRegister_exe", true);
  proc.setEnv("HIP_ENABLE_DEFERRED_LOADING", "1");
  const int exitCode = proc.run();
  INFO(proc.getOutput());
  REQUIRE(exitCode == 0);
}
