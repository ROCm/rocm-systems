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
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <array>

/**
 * @addtogroup hipMemsetD16 hipMemsetD16
 * @{
 * @ingroup MemoryTest
 * `hipMemsetD16(hipDeviceptr_t dest, unsigned char value, size_t count)` -
 * Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * short value value.
 */

// Target type the memset will operate on
using test_target_t = uint16_t;
using memset_fn_t = hipError_t (*)(hipDeviceptr_t dest, test_target_t value, size_t count);

// The memset function itself
static constexpr memset_fn_t memset_fn = hipMemsetD16;

// Table with buffer allocation number of elements
static constexpr std::array<size_t, 5> buffer_nelems = {
    4096, 4096 * 8, 4096 * 32, 4096 * 128, 4096 * 256,
};

// Pattern value that buffers will be set to
static constexpr test_target_t pattern = static_cast<test_target_t>(0xDEADBEEF);

using allocator_fn_t = hipError_t (*)(void** ptr, size_t size);
using deallocator_fn_t = hipError_t (*)(void* ptr);

// Helper function to check if buffer has the expected pattern
static bool checkBuffer(const test_target_t* buffer, size_t size, test_target_t pattern) {
  bool result = true;

  test_target_t* host_ptr = new test_target_t[size];
  HIP_CHECK(hipMemcpy(host_ptr, buffer, size * sizeof(test_target_t), hipMemcpyDefault));

  for (size_t i = 0; i < size; i++) {
    if (host_ptr[i] != pattern) {
      CAPTURE(size, i, buffer[i], pattern);
      result = false;

      break;
    }
  }

  delete[] host_ptr;

  return result;
}

// Helper function to allocate and test buffer pattern after memset
static bool testMemset(allocator_fn_t allocator, deallocator_fn_t deallocator) {
  bool result = true;

  for (size_t size : buffer_nelems) {
    void* ptr = nullptr;

    HIP_CHECK(allocator(&ptr, size * sizeof(test_target_t)));

    HIP_CHECK(memset_fn((hipDeviceptr_t)(ptr), pattern, size));

    result = checkBuffer(static_cast<test_target_t*>(ptr), size, pattern);

    HIP_CHECK(deallocator(ptr));

    if (!result) {
      break;
    }
  }

  return result;
}

/**
 * Test Description
 * ------------------------
 *  - Checks that allocated buffers have the expected value
 * after setting it to a known constant.
 * Test source
 * ------------------------
 *  - catch/unit/memory/hipMemsetD16.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemsetD16_ValidBuffer") {
  SECTION("Device Buffer") {
    bool result = testMemset(hipMalloc, hipFree);

    REQUIRE(result == true);
  }

  SECTION("Host Buffer") {
    auto host_malloc_wrapper =
        +[](void** ptr, size_t size) { return hipHostMalloc(ptr, size, hipHostMallocDefault); };

    bool result = testMemset(host_malloc_wrapper, hipHostFree);

    REQUIRE(result == true);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Checks function behaviour when provided invalid arguments.
 * Test source
 * ------------------------
 *  - catch/unit/memory/hipMemsetD16.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemsetD16_InvalidArg") {
  constexpr size_t ptr_test_nelem = 4096;
  void* ptr = nullptr;

  HIP_CHECK(hipMalloc(&ptr, ptr_test_nelem));

  SECTION("nullptr destination") {
    HIP_CHECK_ERROR(memset_fn((hipDeviceptr_t)(nullptr), pattern, ptr_test_nelem),
                    hipErrorInvalidValue);
  }

  SECTION("zero size") { HIP_CHECK(memset_fn((hipDeviceptr_t)(ptr), pattern, 0)); }

  HIP_CHECK(hipFree(ptr));
}

/**
 * Test Description
 * ------------------------
 *  - Checks that the Kernel allocated buffer has the expected value
 * after setting it to a known constant.
 * Test source
 * ------------------------
 *  - catch/unit/memory/hipMemsetD16.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemsetD16_KernelBuffer") {
  constexpr size_t ptr_test_nelem = 4096;
  constexpr unsigned blocksPerCU = 6;
  constexpr unsigned threadsPerBlock = 256;
  test_target_t* src_ptr = nullptr;
  test_target_t* add_by_one_src_ptr = nullptr;
  test_target_t* dest_ptr = nullptr;
  hipStream_t stream = nullptr;
  size_t nbytes = ptr_test_nelem * sizeof(test_target_t);

  HIP_CHECK(hipMalloc(&src_ptr, nbytes));
  HIP_CHECK(hipMalloc(&add_by_one_src_ptr, nbytes));
  HIP_CHECK(hipMalloc(&dest_ptr, nbytes));
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(memset_fn((hipDeviceptr_t)(src_ptr), pattern, ptr_test_nelem));
  HIP_CHECK(memset_fn((hipDeviceptr_t)(add_by_one_src_ptr), 1, ptr_test_nelem));

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, ptr_test_nelem);

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream, src_ptr,
                     add_by_one_src_ptr, dest_ptr, ptr_test_nelem);

  HIP_CHECK(hipStreamSynchronize(stream));

  bool result = checkBuffer(dest_ptr, ptr_test_nelem, pattern + 1);

  HIP_CHECK(hipFree(src_ptr));
  HIP_CHECK(hipFree(add_by_one_src_ptr));
  HIP_CHECK(hipFree(dest_ptr));
  HIP_CHECK(hipStreamDestroy(stream));

  REQUIRE(result == true);
}
