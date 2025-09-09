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
#include "memset_negative_kernels_rtc.hh"

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include <hip/hip_cooperative_groups.h>

/**
 * @addtogroup memset memset
 * @{
 * @ingroup DeviceLanguageTest
 * `memset(void* ptr, int val, size_t size)` -
 * sets device accessible data inside a kernel
 */

template <typename T> using kernel_sig = void (*)(T*, int, const size_t);

template <typename T>
__global__ void memset_at_once_kernel(T* dst, int value, const size_t alloc_size) {
  memset(dst, value, alloc_size);
}

template <typename T> __global__ void memset_one_by_one_kernel(T* dst, int value, const size_t N) {
  const auto tid = cooperative_groups::this_grid().thread_rank();
  const auto stride = cooperative_groups::this_grid().size();

  for (auto i = tid; i < N; i += stride) {
    memset(dst + tid, value, sizeof(T));
  }
}

template <typename T> void MemsetDeviceCommon(kernel_sig<T> memset_kernel) {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto element_count = allocation_size / sizeof(T);

  LinearAllocGuard<T> reference(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> result(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> dst_allocation(LinearAllocs::hipMalloc, allocation_size);

  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr auto expected_value = 42;

  memset(reference.host_ptr(), expected_value, allocation_size);

  if (memset_kernel == &memset_at_once_kernel<T>) {
    memset_at_once_kernel<T><<<1, 1>>>(dst_allocation.ptr(), expected_value, allocation_size);
  } else {
    memset_one_by_one_kernel<T>
        <<<thread_count, block_count>>>(dst_allocation.ptr(), expected_value, element_count);
  }

  HIP_CHECK(
      hipMemcpy(result.host_ptr(), dst_allocation.ptr(), allocation_size, hipMemcpyDeviceToHost));

  ArrayMismatch(reference.host_ptr(), result.host_ptr(), element_count);
}

template <typename T> void MemsetPinnedCommon(kernel_sig<T> memset_kernel) {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto element_count = allocation_size / sizeof(T);

  LinearAllocGuard<T> reference(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> result(LinearAllocs::hipHostMalloc, allocation_size);

  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr auto expected_value = 42;

  memset(reference.host_ptr(), expected_value, allocation_size);

  if (memset_kernel == &memset_at_once_kernel<T>) {
    memset_at_once_kernel<T><<<1, 1>>>(result.host_ptr(), expected_value, allocation_size);
  } else {
    memset_one_by_one_kernel<T>
        <<<thread_count, block_count>>>(result.host_ptr(), expected_value, element_count);
  }

  HIP_CHECK(hipStreamSynchronize(nullptr));

  ArrayMismatch(reference.host_ptr(), result.host_ptr(), element_count);
}

template <typename T> void DeviceMemsetCommon(kernel_sig<T> memset_kernel) {
  SECTION("Set Device memory") { MemsetDeviceCommon<T>(memset_kernel); }

  SECTION("Set Pinned memory") { MemsetPinnedCommon<T>(memset_kernel); }
}

/**
 * Test Description
 * ------------------------
 *  - Verifies basic test cases for setting device/pinned memory inside a kernel using various data
 * types and memory sizes:
 *    -# Set whole memory buffer in one thread
 *    -# Set memory buffer elements one by one in multiple threads/blocks
 * Test source
 * ------------------------
 *  - unit/device_memory/memset.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_memset_Positive", "", char, int, unsigned int, long, unsigned long,
                   long long, unsigned long long, float, double) {
  SECTION("Memset whole buffer in one thread") {
    DeviceMemsetCommon<TestType>(memset_at_once_kernel);
  }
  SECTION("Memset buffer in multiple threads/blocks") {
    DeviceMemsetCommon<TestType>(memset_one_by_one_kernel);
  }
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for memset
 * Test source
 * ------------------------
 *    - unit/device_memory/memset.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_memset_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source = kMemsetParam;

  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "memset_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  int expected_error_count{4};
  std::string error_message{"error:"};

  size_t n_pos = log.find(error_message, 0);
  while (n_pos != std::string::npos) {
    ++error_count;
    n_pos = log.find(error_message, n_pos + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_error_count);
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
