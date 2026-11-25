/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/**
 * @addtogroup StatCO StatCO
 * @{
 * @ingroup ModuleTest
 * Covers the lifetime of a __managed__ variable's storage across hipDeviceReset().
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>

namespace {

// The helper needs exactly one visible device. With more than one, the variable's storage is
// shared across devices and a reset leaves it alone regardless, so the test would pass whether
// or not the runtime is correct.
void runWithOneVisibleDevice([[maybe_unused]] const char* deferredLoading) {
  hip::SpawnProc proc("managedVarAfterDeviceReset_exe", true);
#if HT_NVIDIA
  proc.setEnv("CUDA_VISIBLE_DEVICES", "0");
#else
  proc.setEnv("HIP_VISIBLE_DEVICES", "0");
  proc.setEnv("HIP_ENABLE_DEFERRED_LOADING", deferredLoading);
#endif
  const int exitCode = proc.run();
  INFO(proc.getOutput());
  REQUIRE(exitCode == 0);
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *    - A module-scope __managed__ variable must stay usable, and keep its contents, after
 *      hipDeviceReset(). Its storage is part of the program image and there is no API to
 *      re-register one, so the runtime has to keep it across a reset.
 *
 *    - Covers deferred loading, where the variable's storage is a host range registered
 *      with the runtime.
 *
 * Test source
 * ------------------------
 *    - catch/unit/module/managedVarAfterDeviceReset.cc
 * Test requirements
 * ------------------------
 *    - Linux
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_StatCO_ManagedVarAfterDeviceReset_DeferredLoading) {
  CHECK_MANAGED_MEMORY_SUPPORT

  runWithOneVisibleDevice("1");
}

/**
 * Test Description
 * ------------------------
 *    - The same guarantee with deferred loading turned off, where the variable's storage is
 *      allocated by the runtime at registration instead. This is a separate allocation path
 *      and has to be kept across a reset as well.
 *
 * Test source
 * ------------------------
 *    - catch/unit/module/managedVarAfterDeviceReset.cc
 * Test requirements
 * ------------------------
 *    - Linux
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_StatCO_ManagedVarAfterDeviceReset_EagerLoading) {
  CHECK_MANAGED_MEMORY_SUPPORT

  runWithOneVisibleDevice("0");
}
