/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy_negative_kernels_rtc.hh"

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include <hip/hip_cooperative_groups.h>

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup DeviceLanguageTest
 * `memcpy(void* dst, const void* src, size_t size)` -
 * copies device accessible data inside a kernel
 */

template <typename T> using kernel_sig = void (*)(T*, T*, const size_t);

template <typename T>
__global__ void memcpy_at_once_kernel(T* dst, T* src, const size_t alloc_size) {
  memcpy(dst, src, alloc_size);
}

template <typename T> __global__ void memcpy_one_by_one_kernel(T* dst, T* src, const size_t N) {
  const auto tid = cooperative_groups::this_grid().thread_rank();
  const auto stride = cooperative_groups::this_grid().size();

  for (auto i = tid; i < N; i += stride) {
    memcpy(dst + tid, src + tid, sizeof(T));
  }
}

template <typename T> void MemcpyDeviceToDeviceCommon(kernel_sig<T> memcpy_kernel) {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto element_count = allocation_size / sizeof(T);

  LinearAllocGuard<T> input(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> result(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> src_allocation(LinearAllocs::hipMalloc, allocation_size);
  LinearAllocGuard<T> dst_allocation(LinearAllocs::hipMalloc, allocation_size);

  /* fill input data */
  for (auto i = 0; i < element_count; i++) {
    input.host_ptr()[i] = static_cast<T>(i);
  }

  /* Copy input data to device memory */
  HIP_CHECK(
      hipMemcpy(src_allocation.ptr(), input.host_ptr(), allocation_size, hipMemcpyHostToDevice));

  /* Launch appropriate kernel*/
  if (memcpy_kernel == &memcpy_at_once_kernel<T>) {
    memcpy_at_once_kernel<T><<<1, 1>>>(dst_allocation.ptr(), src_allocation.ptr(), allocation_size);
  } else {
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    memcpy_one_by_one_kernel<T>
        <<<thread_count, block_count>>>(dst_allocation.ptr(), src_allocation.ptr(), element_count);
  }

  /* Copy filled device memory to result */
  HIP_CHECK(
      hipMemcpy(result.host_ptr(), dst_allocation.ptr(), allocation_size, hipMemcpyDeviceToHost));

  ArrayMismatch(input.host_ptr(), result.host_ptr(), element_count);
}

template <typename T> void MemcpyPinnedToDeviceCommon(kernel_sig<T> memcpy_kernel) {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto element_count = allocation_size / sizeof(T);

  LinearAllocGuard<T> input(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> result(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> dst_allocation(LinearAllocs::hipMalloc, allocation_size);

  /* fill input data */
  for (auto i = 0; i < element_count; i++) {
    input.host_ptr()[i] = static_cast<T>(i);
  }

  /* Launch appropriate kernel*/
  if (memcpy_kernel == &memcpy_at_once_kernel<T>) {
    memcpy_at_once_kernel<T><<<1, 1>>>(dst_allocation.ptr(), input.host_ptr(), allocation_size);
  } else {
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    memcpy_one_by_one_kernel<T>
        <<<thread_count, block_count>>>(dst_allocation.ptr(), input.host_ptr(), element_count);
  }

  /* Copy filled device memory to result */
  HIP_CHECK(
      hipMemcpy(result.host_ptr(), dst_allocation.ptr(), allocation_size, hipMemcpyDeviceToHost));

  ArrayMismatch(input.host_ptr(), result.host_ptr(), element_count);
}

template <typename T> void MemcpyDeviceToPinnedCommon(kernel_sig<T> memcpy_kernel) {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto element_count = allocation_size / sizeof(T);

  LinearAllocGuard<T> input(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> result(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> src_allocation(LinearAllocs::hipMalloc, allocation_size);

  /* fill input data */
  for (auto i = 0; i < element_count; i++) {
    input.host_ptr()[i] = static_cast<T>(i);
  }

  /* Copy input data to device memory */
  HIP_CHECK(
      hipMemcpy(src_allocation.ptr(), input.host_ptr(), allocation_size, hipMemcpyHostToDevice));

  /* Launch appropriate kernel*/
  if (memcpy_kernel == &memcpy_at_once_kernel<T>) {
    memcpy_at_once_kernel<T><<<1, 1>>>(result.host_ptr(), src_allocation.ptr(), allocation_size);
  } else {
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    memcpy_one_by_one_kernel<T>
        <<<thread_count, block_count>>>(result.host_ptr(), src_allocation.ptr(), element_count);
  }

  HIP_CHECK(hipStreamSynchronize(nullptr));

  ArrayMismatch(input.host_ptr(), result.host_ptr(), element_count);
}

template <typename T> void MemcpyPinnedToPinnedCommon(kernel_sig<T> memcpy_kernel) {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto element_count = allocation_size / sizeof(T);

  LinearAllocGuard<T> input(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> result(LinearAllocs::hipHostMalloc, allocation_size);

  /* fill input data */
  for (auto i = 0; i < element_count; i++) {
    input.host_ptr()[i] = static_cast<T>(i);
  }

  /* Launch appropriate kernel*/
  if (memcpy_kernel == &memcpy_at_once_kernel<T>) {
    memcpy_at_once_kernel<T><<<1, 1>>>(result.host_ptr(), input.host_ptr(), allocation_size);
  } else {
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    memcpy_one_by_one_kernel<T>
        <<<thread_count, block_count>>>(result.host_ptr(), input.host_ptr(), element_count);
  }

  HIP_CHECK(hipStreamSynchronize(nullptr));

  ArrayMismatch(input.host_ptr(), result.host_ptr(), element_count);
}

template <typename T> void DeviceMemcpyCommon(kernel_sig<T> memcpy_kernel) {
  SECTION("Device to Device memory") { MemcpyDeviceToDeviceCommon<T>(memcpy_kernel); }

  SECTION("Pinned to Device memory") { MemcpyPinnedToDeviceCommon<T>(memcpy_kernel); }

  SECTION("Device to Pinned memory") { MemcpyDeviceToPinnedCommon<T>(memcpy_kernel); }

  SECTION("Pinned to Pinned memory") { MemcpyPinnedToPinnedCommon<T>(memcpy_kernel); }
}


/**
 * Test Description
 * ------------------------
 *  - Verifies basic test cases for copying device/pinned memory inside a kernel using various data
 * types and memory sizes:
 *    -# Copies whole memory buffer in one thread
 *    -# Copies memory buffer elements one by one in multiple threads/blocks
 * Test source
 * ------------------------
 *  - unit/device_memory/memcpy.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_Device_memcpy_Positive, char, int, unsigned int, long, unsigned long,
                   long long, unsigned long long, float, double) {
  SECTION("Memcpy whole buffer in one thread") {
    DeviceMemcpyCommon<TestType>(memcpy_at_once_kernel);
  }
  SECTION("Memcpy buffer in multiple threads/blocks") {
    DeviceMemcpyCommon<TestType>(memcpy_one_by_one_kernel);
  }
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for memcpy
 * Test source
 * ------------------------
 *    - unit/device_memory/memcpy.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_Device_memcpy_Negative_Parameters_RTC) {
  hiprtcProgram program{};

  const auto program_source = kMemcpyParam;

  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "memcpy_negative.cc", 0, nullptr, nullptr));
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
