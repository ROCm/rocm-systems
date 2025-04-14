/*Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * @addtogroup hipMemcpyFromSymbol_spt hipMemcpyFromSymbol_spt
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol,
                               size_t sizeBytes, size_t offset __dparm(0),
                               hipMemcpyKind kind
 __dparm(hipMemcpyDeviceToHost))` -
 * Copies data from the given symbol on the device.
 */
__device__ int devSymbol[10];
__constant__ int constSymbol[10];
/**
 * Test Description
 * ------------------------
 * - Basic test to check the negative cases of hipMemcpyFromSymbol_spt.
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpyFromSymbol_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpyFromSymbol_spt_Negative") {
  SECTION("Invalid Src Ptr") {
    HIP_CHECK_ERROR(hipMemcpyFromSymbol_spt(nullptr, HIP_SYMBOL(devSymbol),
                                            sizeof(int), 0,
                                            hipMemcpyDeviceToHost),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid Dst Ptr") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyFromSymbol_spt(&result, nullptr, sizeof(int), 0,
                                            hipMemcpyDeviceToHost),
                    hipErrorInvalidSymbol);
  }
  SECTION("Invalid Size") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyFromSymbol_spt(&result, HIP_SYMBOL(devSymbol),
                                            sizeof(int) * 100, 0,
                                            hipMemcpyDeviceToHost),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid Offset") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyFromSymbol_spt(&result, HIP_SYMBOL(devSymbol),
                                            sizeof(int), 300,
                                            hipMemcpyDeviceToHost),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid Direction") {
    int result{0};
    HIP_CHECK_ERROR(hipMemcpyFromSymbol_spt(&result, HIP_SYMBOL(devSymbol),
                                            sizeof(int), 0,
                                            hipMemcpyHostToDevice),
                    hipErrorInvalidMemcpyDirection);
  }
}
/**
 * Test Description
 * ------------------------
 * -  Test Verifies hipMemcpyFromSymbol_spt for simple use case
 * For single value From Symbol
 * For Array Values From Symbol
 * For Array Values with offset From Symbol
 * Test source
 * ------------------------
 * - catch\unit\memory\hipMemcpyFromSymbol_spt.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpyFromSymbol_spt_Sync") {
  SECTION("Singular Value") {
    int set{42};
    int result{0};
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), &set, sizeof(int)));
    HIP_CHECK(
        hipMemcpyFromSymbol_spt(&result, HIP_SYMBOL(devSymbol), sizeof(int)));
    REQUIRE(result == set);
  }
  SECTION("Array Values") {
    constexpr size_t size{10};
    int set[size] = {4, 2, 4, 2, 4, 2, 4, 2, 4, 2};
    int result[size] = {0};
    HIP_CHECK(
        hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), set, sizeof(int) * size));
    HIP_CHECK(hipMemcpyFromSymbol_spt(&result, HIP_SYMBOL(devSymbol),
                                      sizeof(int) * size));
    for (size_t i = 0; i < size; i++) {
      REQUIRE(result[i] == set[i]);
    }
  }
  SECTION("Offset'ed Values") {
    constexpr size_t size{10};
    constexpr size_t offset = 5 * sizeof(int);
    int set[size] = {9, 9, 9, 9, 9, 2, 4, 2, 4, 2};
    int result[size] = {0};
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), set, offset));
    HIP_CHECK(
        hipMemcpyToSymbol(HIP_SYMBOL(devSymbol), set + 5, offset, offset));
    HIP_CHECK(hipMemcpyFromSymbol_spt(result, HIP_SYMBOL(devSymbol),
                                      sizeof(int) * size));
    for (size_t i = 0; i < size; i++) {
      REQUIRE(result[i] == set[i]);
    }
  }
}
/**
 * End doxygen group MemoryTest.
 * @}
 */

