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

//TODO Review this

/**
 * Test Description
 * ------------------------
 *    - Deferred managed-variable allocation exists so that declaring a __managed__
 *      variable does not attach the process to the KFD during static initialization.
 *      A child forked before the first HIP API call can then still use the GPU, where
 *      libhsakmt fails every hsaKmt call in a child that inherited an already-open
 *      connection.
 *
 *    - The first sub-case is that guarantee: no KFD connection exists before main, and a
 *      forked child carries the first managed-symbol touch of the process through to a
 *      completed kernel. The second inverts the deferral and requires the connection to
 *      be open, so a regression that attaches at registration time cannot leave the first
 *      sub-case passing vacuously.
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

  SECTION("deferred allocation leaves a forked child able to use the GPU") {
    hip::SpawnProc proc("managedForkAfterRegister_exe", true);
    proc.setEnv("HIP_ENABLE_DEFERRED_LOADING", "1");
    const int exitCode = proc.run();
    INFO(proc.getOutput());
    REQUIRE(exitCode == 0);
  }

  SECTION("eager allocation attaches to the KFD before main") {
    hip::SpawnProc proc("managedForkAfterRegister_exe", true);
    proc.setEnv("HIP_ENABLE_DEFERRED_LOADING", "0");
    const int exitCode = proc.run();
    INFO(proc.getOutput());
    REQUIRE(exitCode == 0);
  }
}
