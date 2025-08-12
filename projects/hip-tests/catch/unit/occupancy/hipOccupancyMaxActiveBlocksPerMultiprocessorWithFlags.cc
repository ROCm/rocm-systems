/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <limits>

/**
 * @addtogroup hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags)` -
 * Returns occupancy for a device function with the specified flags.
 */

// Kernel block size
static constexpr int blockSize = 256;

// Dummy Kernel Function
static __global__ void kern1(int* t) { *t = 1; }

/**
 * Test Description
 * ------------------------
 *  - Test if the function returns expected values when valid arguments are provided.
 * Test source
 * ------------------------
 *  - catch/unit/occupancy/hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_ValidArgs") {
  hipError_t err;

  SECTION("hipOccupancyDefault no shared memory") {
    int numBlocks = 0;
    err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, kern1, blockSize, 0,
                                                                hipOccupancyDefault);

    HIP_CHECK(err);
    REQUIRE(numBlocks > 0);
  }

  SECTION("hipOccupancyDefault with shared memory") {
    int numBlocks = 0;
    err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, kern1, blockSize, 256,
                                                                hipOccupancyDefault);

    HIP_CHECK(err);
    REQUIRE(numBlocks > 0);
  }

  SECTION("hipOccupancyDisableCachingOverride no shared memory") {
    int numBlocks = 0;
    err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, kern1, blockSize, 0,
                                                                hipOccupancyDisableCachingOverride);

    HIP_CHECK(err);
    REQUIRE(numBlocks > 0);
  }

  SECTION("hipOccupancyDisableCachingOverride with shared memory") {
    int numBlocks = 0;
    err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, kern1, blockSize, 256,
                                                                hipOccupancyDisableCachingOverride);

    HIP_CHECK(err);
    REQUIRE(numBlocks > 0);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test if the function returns expected values when invalid arguments are provided.
 * Test source
 * ------------------------
 *  - catch/unit/occupancy/hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_InvalidArgs") {
  hipError_t err;

  SECTION("Zero block size") {
    int numBlocks = 0;
    err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, kern1, 0, 0,
                                                                hipOccupancyDefault);

    HIP_CHECK_ERROR(err, hipErrorInvalidValue);
    REQUIRE(numBlocks == 0);
  }

  SECTION("Null kernel") {
    int numBlocks = 0;
    err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &numBlocks, static_cast<const void*>(nullptr), blockSize, 0, hipOccupancyDefault);

    HIP_CHECK_ERROR(err, hipErrorInvalidDeviceFunction);
    REQUIRE(numBlocks == 0);
  }

  SECTION("Invalid flag") {
    int numBlocks = 0;
    err =
        hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, kern1, blockSize, 0, -1);

    HIP_CHECK_ERROR(err, hipErrorInvalidValue);
    REQUIRE(numBlocks == 0);
  }

  SECTION("Null numBlocks") {
    err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(nullptr, kern1, blockSize, 0,
                                                                hipOccupancyDefault);

    HIP_CHECK_ERROR(err, hipErrorInvalidValue);
  }

  SECTION("Too large block size") {
    int numBlocks = 0;
    err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &numBlocks, kern1, std::numeric_limits<int>::max(), 0, hipOccupancyDefault);

    HIP_CHECK(err);
    REQUIRE(numBlocks == 0);
  }

  SECTION("Too large shared memory size") {
    int numBlocks = 0;
    err = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &numBlocks, kern1, blockSize, std::numeric_limits<int>::max(), hipOccupancyDefault);

    HIP_CHECK(err);
    REQUIRE(numBlocks == 0);
  }
}
