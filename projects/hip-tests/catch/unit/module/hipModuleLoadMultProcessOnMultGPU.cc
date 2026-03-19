/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>
/**
 * @addtogroup hipModuleLoad hipModuleLoadData hipModuleLoadDataEx
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleLoad(hipModule_t* module, const char* fname)` -
 * Loads code object from file into a module
 * `hipError_t 	hipModuleLoadData (hipModule_t *module, const void *image)` -
 * Builds module from code object which resides in host memory. Image is pointer to that location.
 * `hipError_t 	hipModuleLoadDataEx (hipModule_t *module, const void *image,
 *        unsigned int numOptions, hipJitOption *options, void **optionValues)` -
 * Builds module from code object which resides in host memory. Image is pointer to that
 * location. Options are not used.
 */

/**
 * Test Description
 * ------------------------
 * - Test case to load and execute a code object file for multiprocess and multiGPU.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleLoadMultProcessOnMultGPU.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE(Unit_hipModuleLoad_MultProcess_MultGPU) {
  int deviceCount{0};
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount != 0);
  // Spawn 1 Process for each device
  for (int deviceNo = 0; deviceNo < deviceCount; deviceNo++) {
    // set the device id for the current process
    HIP_CHECK(hipSetDevice(deviceNo));
    hip::SpawnProc proc("testhipModuleLoadUnloadFunc_exe", true);
    REQUIRE(proc.run("1") == true);
    REQUIRE(proc.run("2") == true);
    REQUIRE(proc.run("3") == true);
  }
}
