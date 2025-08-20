/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
TEST_CASE("Unit_hipModuleLoad_MultProcess_MultGPU") {
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
