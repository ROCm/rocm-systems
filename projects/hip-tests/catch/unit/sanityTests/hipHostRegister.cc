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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_tests_config.hh>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <vector>

/**
 * @addtogroup HipHostRegisterStressTest HipHostRegisterStressTest
 * @{
 * @ingroup HipHostRegisterStressTest
 */

constexpr size_t MEM_SIZE = 1024 * 1024;   // 1 MB
constexpr int NUM_ITERATIONS = 5;          // iterations with increasing size
constexpr int NBUF_ALLOCATIONS = 100;      // number of buffers (sanity test)
constexpr int NBUF_SIZES = 3;              // 1KB, 1MB, 10MB
constexpr int NBUF_FLAGS = 3;

/**
 * Test Description
 * ------------------------
 *    - Stress test for hipHostRegister/hipHostUnregister with hipHostRegisterPortable.
 *      Allocates host memory of increasing size (1MB, 2MB, ...), registers it,
 *      performs CPU-GPU transfer, then unregisters and frees.
 * Test source
 * ------------------------
 *    - unit/sanityTests/hipHostRegister.cc
 */
HIP_TEST_CASE(Unit_hipHostRegister_RegisterUnregister) {
  HIP_CHECK(hipSetDevice(0));

  for (int count = 1; count <= NUM_ITERATIONS; ++count) {
    size_t size = count * MEM_SIZE;
    void* ptr = calloc(1, size);
    REQUIRE(ptr != nullptr);

    HIP_CHECK(hipHostRegister(ptr, size, hipHostRegisterPortable));

    void* dptr = nullptr;
    HIP_CHECK(hipMalloc(&dptr, size));
    HIP_CHECK(hipMemcpy(dptr, ptr, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ptr, dptr, size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dptr));

    HIP_CHECK(hipHostUnregister(ptr));
    free(ptr);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Stress test for hipHostRegister/hipHostUnregister with multiple buffer
 *      counts and flag values. Allocates many chunks per size, registers all
 *      with each flag, unregisters all, then frees.
 * Test source
 * ------------------------
 *    - unit/sanityTests/hipHostRegister.cc
 */

HIP_TEST_CASE(Unit_hipHostRegister_Nbuf_MultiFlag_RegisterUnregister) {
  static const size_t sizes[NBUF_SIZES] = {1024, 1048576, 10485760};

  static const unsigned int flags[NBUF_FLAGS] = {
      hipHostRegisterDefault, hipHostRegisterPortable, hipHostRegisterMapped};

  std::vector<void*> ptrs(NBUF_ALLOCATIONS);

  for (int j = 0; j < NBUF_SIZES; ++j) {
    size_t size = sizes[j];
    for (int i = 0; i < NBUF_ALLOCATIONS; ++i) {
      ptrs[i] = calloc(1, size);
      REQUIRE(ptrs[i] != nullptr);
    }

    for (int ii = 0; ii < NBUF_FLAGS; ++ii) {
      unsigned int fl = flags[ii];
      for (int i = 0; i < NBUF_ALLOCATIONS; ++i) {
        HIP_CHECK(hipHostRegister(ptrs[i], size, fl));
      }
      for (int ur = 0; ur < NBUF_ALLOCATIONS; ++ur) {
        HIP_CHECK(hipHostUnregister(ptrs[ur]));
      }
    }

    for (int i = 0; i < NBUF_ALLOCATIONS; ++i) {
      free(ptrs[i]);
    }
  }
}

/**
 * End doxygen group HipHostRegisterStressTest.
 * @}
 */
