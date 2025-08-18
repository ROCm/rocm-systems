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
#include <hip_test_helper.hh>
#include <utils.hh>

 /**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - amd dbg APIS from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular APIs
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddressAmdDbgApi.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_amd_dbgapi") {
  void* amd_dbgapi_get_build_name_ptr = nullptr;
  void* amd_dbgapi_get_git_hash_ptr = nullptr;
  void* amd_dbgapi_get_build_id_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress(
            "amd_dbgapi_get_build_name",
            &amd_dbgapi_get_build_name_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "amd_dbgapi_get_git_hash",
            &amd_dbgapi_get_git_hash_ptr,
            currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress(
            "amd_dbgapi_get_build_id",
            &amd_dbgapi_get_build_id_ptr,
            currentHipVersion, 0, nullptr));

  const char* (*dyn_amd_dbgapi_get_build_name_ptr)() =
    reinterpret_cast<const char* (*)()>
    (amd_dbgapi_get_build_name_ptr);
  const char* (*dyn_amd_dbgapi_get_git_hash_ptr)() =
    reinterpret_cast<const char* (*)()>
    (amd_dbgapi_get_git_hash_ptr);
  size_t (*dyn_amd_dbgapi_get_build_id_ptr)() =
    reinterpret_cast<size_t (*)()>
    (amd_dbgapi_get_build_id_ptr);

  const char * buildNameFromOrg = amd_dbgapi_get_build_name();
  const char * buildNameFromPtr = dyn_amd_dbgapi_get_build_name_ptr();
  REQUIRE(strcmp(buildNameFromOrg, buildNameFromPtr) == 0);

  const char * gitHashFromOrg = amd_dbgapi_get_git_hash();
  const char * gitHashFromPtr = dyn_amd_dbgapi_get_git_hash_ptr();
  REQUIRE(strcmp(gitHashFromOrg, gitHashFromPtr) == 0);

  size_t buildIdFromOrg = amd_dbgapi_get_build_id();
  size_t buildIdFromPtr = dyn_amd_dbgapi_get_build_id_ptr();
  REQUIRE(buildIdFromOrg == buildIdFromPtr);
}
