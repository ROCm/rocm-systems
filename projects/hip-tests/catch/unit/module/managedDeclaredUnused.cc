/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/**
 * @addtogroup StatCO StatCO
 * @{
 * @ingroup ModuleTest
 * Covers a program that declares __managed__ variables without ever using them.
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>

// TODO Review this
/**
 * Test Description
 * ------------------------
 *    - A child process declares __managed__ variables and never reads or writes them.
 *      One sub-case enters HIP through two trivial APIs, the other calls no HIP API at
 *      all; the neighbouring managed tests all touch their variables, so neither state
 *      is reachable from them.
 *
 *    - Deliberately not gated on hipDeviceAttributeManagedMemory. HIP reports that
 *      attribute from hmmSupported_, so gating would skip exactly the configuration
 *      this test exists for: a declared managed variable must not break a program on a
 *      system that reports managed memory as unsupported, because the declaration is
 *      legal there and degrades to pinned host memory.
 * Test source
 * ------------------------
 *    - catch/unit/module/managedDeclaredUnused.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_StatCO_ManagedVarDeclaredUnused_InChildProcess) {
  SECTION("trivial HIP API call promotes the unused variables") {
    hip::SpawnProc proc("managedDeclaredUnused_exe", true);
    REQUIRE(proc.run("trivial-api") == 0);
  }

  SECTION("no HIP API call leaves them unpromoted until teardown") {
    hip::SpawnProc proc("managedDeclaredUnused_exe", true);
    REQUIRE(proc.run("no-api") == 0);
  }
}
