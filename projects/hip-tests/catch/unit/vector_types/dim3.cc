/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * @addtogroup dim3 dim3
 * @{
 * @ingroup VectorTypeTest
 */

__global__ void Dim3VectorKernel(dim3* vector) { *vector = dim3(); }
__global__ void Dim3VectorKernel(dim3* vector, const uint32_t x) { *vector = dim3(x); }
__global__ void Dim3VectorKernel(dim3* vector, const uint32_t x, const uint32_t y) {
  *vector = dim3(x, y);
}
__global__ void Dim3VectorKernel(dim3* vector, const uint32_t x, const uint32_t y,
                                 const uint32_t z) {
  *vector = dim3(x, y, z);
}

/**
 * Test Description
 * ------------------------
 *    - Creates a dim3 with an empty constructor:
 *        -# Expected result: dim3(1, 1, 1)
 *    - Calls dim3 from the device side
 * Test source
 * ------------------------
 *    - unit/vector_types/dim3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_dim3_Empty_Positive_Device") {
  dim3 vector_h{0, 0, 0};
  dim3* vector_d;
  HIP_CHECK(hipMalloc(&vector_d, sizeof(dim3)));
  HIP_CHECK(hipMemcpy(vector_d, &vector_h, sizeof(dim3), hipMemcpyHostToDevice));
  Dim3VectorKernel<<<1, 1, 0, 0>>>(vector_d);
  HIP_CHECK(hipMemcpy(&vector_h, vector_d, sizeof(dim3), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(vector_d));

  REQUIRE(vector_h.x == 1);
  REQUIRE(vector_h.y == 1);
  REQUIRE(vector_h.z == 1);
}

/**
 * Test Description
 * ------------------------
 *    - Creates a dim3 with an constructor with one parameter (X):
 *        -# Expected result: dim3(X, 1, 1)
 *    - Calls dim3 from the device side
 * Test source
 * ------------------------
 *    - unit/vector_types/dim3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_dim3_X_Positive_Device") {
  dim3 vector_h{0, 0, 0};
  dim3* vector_d;
  HIP_CHECK(hipMalloc(&vector_d, sizeof(dim3)));
  HIP_CHECK(hipMemcpy(vector_d, &vector_h, sizeof(dim3), hipMemcpyHostToDevice));
  uint32_t value_x =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  Dim3VectorKernel<<<1, 1, 0, 0>>>(vector_d, value_x);
  HIP_CHECK(hipMemcpy(&vector_h, vector_d, sizeof(dim3), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(vector_d));

  REQUIRE(vector_h.x == value_x);
  REQUIRE(vector_h.y == 1);
  REQUIRE(vector_h.z == 1);
}

/**
 * Test Description
 * ------------------------
 *    - Creates a dim3 with an constructor with two parameters (X, Y):
 *        -# Expected result: dim3(X, Y, 1)
 *    - Calls dim3 from the device side
 * Test source
 * ------------------------
 *    - unit/vector_types/dim3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_dim3_XY_Positive_Device") {
  dim3 vector_h{0, 0, 0};
  dim3* vector_d;
  HIP_CHECK(hipMalloc(&vector_d, sizeof(dim3)));
  HIP_CHECK(hipMemcpy(vector_d, &vector_h, sizeof(dim3), hipMemcpyHostToDevice));
  uint32_t value_x =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  uint32_t value_y =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  Dim3VectorKernel<<<1, 1, 0, 0>>>(vector_d, value_x, value_y);
  HIP_CHECK(hipMemcpy(&vector_h, vector_d, sizeof(dim3), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(vector_d));

  REQUIRE(vector_h.x == value_x);
  REQUIRE(vector_h.y == value_y);
  REQUIRE(vector_h.z == 1);
}

/**
 * Test Description
 * ------------------------
 *    - Creates a dim3 with an constructor with three parameters (X, Y, Z):
 *        -# Expected result: dim3(X, Y, Z)
 *    - Calls dim3 from the device side
 * Test source
 * ------------------------
 *    - unit/vector_types/dim3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_dim3_XYZ_Positive_Device") {
  dim3 vector_h{0, 0, 0};
  dim3* vector_d;
  HIP_CHECK(hipMalloc(&vector_d, sizeof(dim3)));
  HIP_CHECK(hipMemcpy(vector_d, &vector_h, sizeof(dim3), hipMemcpyHostToDevice));
  uint32_t value_x =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  uint32_t value_y =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());

  uint32_t value_z =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  Dim3VectorKernel<<<1, 1, 0, 0>>>(vector_d, value_x, value_y, value_z);
  HIP_CHECK(hipMemcpy(&vector_h, vector_d, sizeof(dim3), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(vector_d));

  REQUIRE(vector_h.x == value_x);
  REQUIRE(vector_h.y == value_y);
  REQUIRE(vector_h.z == value_z);
}

/**
 * Test Description
 * ------------------------
 *    - Creates a dim3 with an empty constructor:
 *        -# Expected result: dim3(1, 1, 1)
 *    - Calls dim3 from the host side
 * Test source
 * ------------------------
 *    - unit/vector_types/dim3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_dim3_Empty_Positive_Host") {
  dim3 vector = dim3();
  REQUIRE(vector.x == 1);
  REQUIRE(vector.y == 1);
  REQUIRE(vector.z == 1);
}

/**
 * Test Description
 * ------------------------
 *    - Creates a dim3 with an constructor with one parameter (X):
 *        -# Expected result: dim3(X, 1, 1)
 *    - Calls dim3 from the host side
 * Test source
 * ------------------------
 *    - unit/vector_types/dim3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_dim3_X_Positive_Host") {
  uint32_t value_x =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  dim3 vector = dim3(value_x);
  REQUIRE(vector.x == value_x);
  REQUIRE(vector.y == 1);
  REQUIRE(vector.z == 1);
}

/**
 * Test Description
 * ------------------------
 *    - Creates a dim3 with an constructor with two parameters (X, Y):
 *        -# Expected result: dim3(X, Y, 1)
 *    - Calls dim3 from the host side
 * Test source
 * ------------------------
 *    - unit/vector_types/dim3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_dim3_XY_Positive_Host") {
  uint32_t value_x =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  uint32_t value_y =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  dim3 vector = dim3(value_x, value_y);
  REQUIRE(vector.x == value_x);
  REQUIRE(vector.y == value_y);
  REQUIRE(vector.z == 1);
}

/**
 * Test Description
 * ------------------------
 *    - Creates a dim3 with an constructor with three parameters (X, Y, Z):
 *        -# Expected result: dim3(X, Y, Z)
 *    - Calls dim3 from the host side
 * Test source
 * ------------------------
 *    - unit/vector_types/dim3.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_dim3_XYZ_Positive_Host") {
  uint32_t value_x =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  uint32_t value_y =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  uint32_t value_z =
      GENERATE(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max() / 2,
               std::numeric_limits<uint32_t>::max());
  dim3 vector = dim3(value_x, value_y, value_z);
  REQUIRE(vector.x == value_x);
  REQUIRE(vector.y == value_y);
  REQUIRE(vector.z == value_z);
}

/**
 * End doxygen group VectorTypeTest.
 * @}
 */
