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
 * @addtogroup hipGraphDestroy hipGraphDestroy
 * @{
 * @ingroup GraphTest
 * `hipGraphDestroy(hipGraph_t graph)` -
 * Destroys a graph
 */

/**
 * Test Description
 * ------------------------
 *    - Basic positive test for hipGraphDestroy
 *    - Create an emtpy graph and then destroy it
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphDestroy.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphDestroy_Positive_Basic") {
  hipGraph_t graph = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  REQUIRE(nullptr != graph);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Basic negative parameter test for hipGraphDestroy
 *        -# Expected hipErrorInvalidValue when graph is invalid
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphDestroy.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphDestroy_Negative_Parameters") {
  HIP_CHECK_ERROR(hipGraphDestroy(static_cast<hipGraph_t>(nullptr)), hipErrorInvalidValue);
}

/**
 * End doxygen group GraphTest.
 * @}
 */
