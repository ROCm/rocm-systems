/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <dlfcn.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipGetProcAddress hipGetProcAddress_spt
 * @{
 * @ingroup DynamicLoading
 * `hipError_t hipGetProcAddress (const char * symbol, void ** pfn, int
                                  hipVersion, uint64_t flags,
                                  hipDriverProcAddressQueryResult *
 symbolStatus)` -
 * Gets the pointer of requested HIP driver function.
*/

/**
 * Test Description
 * ------------------------
 * - This will verify behavior of hipGetProcAddress
 *   with compiler option -fgpu-default-stream as per-thread
 * Test source
 * ------------------------
 *  - unit/dynamicLoading/hipGetProcAddressCompilerOption.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.12
 */
#if HT_AMD
HIP_TEST_CASE(Unit_hipGetProcAddress_With_spt_CompilerOption) {
  void *funcPtr_default = nullptr;
  void *funcPtr_legacy = nullptr;
  void *funcPtr_spt = nullptr;

  hipDriverProcAddressQueryResult status;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  void *handle = nullptr;
  handle = dlopen("libamdhip64.so", RTLD_LAZY);

  if (handle == nullptr) {
    // Try to find in the user defined rocm path
    char *rocmPath = nullptr;
    rocmPath = std::getenv("ROCM_PATH");

    if (rocmPath) {
      std::string libPathFile(rocmPath);
      libPathFile += "/lib/libamdhip64.so";
      handle = dlopen(libPathFile.c_str(), RTLD_LAZY);
    }
    // Try to find in the /opt/rocm/lib path
    else {
      handle = dlopen("/opt/rocm/lib/libamdhip64.so", RTLD_LAZY);
    }
  }
  REQUIRE(handle != nullptr);

  void *sym_hipMemcpy = dlsym(handle, "hipMemcpy");
  void *sym_hipMemcpy_spt = dlsym(handle, "hipMemcpy_spt");

  HIP_CHECK(hipGetProcAddress("hipMemcpy", &funcPtr_default, currentHipVersion,
                              HIP_GET_PROC_ADDRESS_DEFAULT, &status));

  HIP_CHECK(hipGetProcAddress("hipMemcpy", &funcPtr_legacy, currentHipVersion,
                              HIP_GET_PROC_ADDRESS_LEGACY_STREAM, &status));

  HIP_CHECK(hipGetProcAddress("hipMemcpy", &funcPtr_spt, currentHipVersion,
                              HIP_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
                              &status));

  REQUIRE(funcPtr_default == sym_hipMemcpy_spt);
  REQUIRE(funcPtr_legacy == sym_hipMemcpy);
  REQUIRE(funcPtr_spt == sym_hipMemcpy_spt);

  REQUIRE(dlclose(handle) == 0);
}
#endif

/**
 * End doxygen group DeviceTest.
 * @}
 */
