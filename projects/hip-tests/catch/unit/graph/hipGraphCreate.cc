/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @addtogroup hipGraphCreate hipGraphCreate
 * @{
 * @ingroup GraphTest
 * `hipGraphCreate(hipGraph_t *pGraph, unsigned int flags)` -
 * creates a graph
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameter test for hipGraphCreate:
 *        -# Expected hipErrorInvalidValue when pGraph is null
 *        -# Expected hipErrorInvalidValue when flags is not 0
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphCreate_Negative_Parameters") {
  hipGraph_t graph = nullptr;

  SECTION("pGraph is nullptr") {
    HIP_CHECK_ERROR(hipGraphCreate(nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("flags is not 0") { HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue); }
}

/**
 * Test Description
 * ------------------------
 *    - Basic positive test for hipGraphCreate
 *    - Create an emtpy graph
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphCreate_Positive_Basic") {
  hipGraph_t graph = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  REQUIRE(nullptr != graph);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
