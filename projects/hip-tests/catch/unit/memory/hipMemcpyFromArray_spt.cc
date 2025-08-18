/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipMemcpyFromArray_spt hipMemcpyFromArray_spt
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpyFromArray_spt(void* dst, hipArray_const_t srcArray,
 size_t wOffset, size_t hOffset, size_t count, hipMemcpyKind kind)` -
 * Copies data between host and device.
 */
/**
 * Test Description
 * ------------------------
 * - Basic test to verify the functionality of hipMemcpyFromArray_spt.
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpyFromArray_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpyFromArray_spt_Basic_Postive") {
  size_t width = 64;
  size_t height = 1;
  const int N = width * height;
  int value = 10;
  int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
  REQUIRE(hostMem != nullptr);
  for (int i = 0; i < N; i++) {
    hostMem[i] = value;
  }
  hipArray_t array = nullptr;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  unsigned int flags = hipArrayDefault;
  HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
  REQUIRE(array != nullptr);
  HIP_CHECK(hipMemcpyToArray(array, 0, 0, hostMem, N * sizeof(int), hipMemcpyHostToDevice));
  int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
  REQUIRE(hostMemory != nullptr);
  HIP_CHECK(hipMemcpyFromArray_spt(hostMemory, array, 0, 0, N * sizeof(int), hipMemcpyDefault));
  for (int i = 0; i < N; i++) {
    if (hostMemory[i] != value) {
      REQUIRE(false);
    }
  }
  free(hostMem);
  free(hostMemory);
  HIP_CHECK(hipFreeArray(array));
}
/**
 * Test Description
 * ------------------------
 * - Negative tests of hipMemcpyFromArray_spt.
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpyFromArray_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpyFromArray_spt_NegativeTests") {
  size_t width = 64;
  size_t height = 1;
  const int N = width * height;
  int value = 10;
  int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
  REQUIRE(hostMem != nullptr);
  for (int i = 0; i < N; i++) {
    hostMem[i] = value;
  }
  hipArray_t array = nullptr;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  unsigned int flags = hipArrayDefault;
  HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
  REQUIRE(array != nullptr);
  HIP_CHECK(hipMemcpyToArray(array, 0, 0, hostMem, N * sizeof(int), hipMemcpyHostToDevice));
  int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
  REQUIRE(hostMemory != nullptr);
  SECTION("Destination Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyFromArray_spt(nullptr, array, 0, 0, N * sizeof(int), hipMemcpyDefault),
                    hipErrorInvalidValue);
  }
  SECTION("Source Array as nullptr") {
    HIP_CHECK_ERROR(
        hipMemcpyFromArray_spt(hostMemory, nullptr, 0, 0, N * sizeof(int), hipMemcpyDefault),
        hipErrorInvalidValue);
  }
  SECTION("Invalid Size") {
    HIP_CHECK_ERROR(hipMemcpyFromArray_spt(hostMemory, array, 0, 0, -3, hipMemcpyDefault),
                    hipErrorInvalidValue);
  }
  free(hostMem);
  free(hostMemory);
  HIP_CHECK(hipFreeArray(array));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
