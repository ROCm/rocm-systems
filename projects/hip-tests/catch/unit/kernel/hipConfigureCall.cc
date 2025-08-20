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
 * Test Description
 * ------------------------
 *  - Basic test checks default behaviour
 * Test source
 * ------------------------
 *  - kernel/hipConfigureCall.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_ConfigureCall") {
  struct dim3 grid_dim{};
  struct dim3 block_dim{};
  size_t shared_memory_size = 1024;

  HIP_CHECK(hipConfigureCall(grid_dim, block_dim, shared_memory_size));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test verifies parameters
 * Test source
 * ------------------------
 *  - kernel/hipConfigureCall.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_ConfigureCall_CheckParams") {
  struct dim3 grid_dim{16, 8, 1};
  struct dim3 test_grid_dim{};
  struct dim3 block_dim{16, 8, 1};
  struct dim3 test_block_dim{};
  size_t shmem_size = 1024;
  size_t test_shmem_size = 0;
  hipStream_t test_stream;

  HIP_CHECK(hipConfigureCall(grid_dim, block_dim, shmem_size));

  HIP_CHECK(
      __hipPopCallConfiguration(&test_grid_dim, &test_block_dim, &test_shmem_size, &test_stream));

  REQUIRE(test_grid_dim.x == grid_dim.x);
  REQUIRE(test_grid_dim.y == grid_dim.y);
  REQUIRE(test_grid_dim.z == grid_dim.z);

  REQUIRE(test_block_dim.x == block_dim.x);
  REQUIRE(test_block_dim.y == block_dim.y);
  REQUIRE(test_block_dim.z == block_dim.z);

  REQUIRE(test_shmem_size == shmem_size);
}
