/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "atomicXor_negative_kernels_rtc.hh"
#include "bitwise_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicXor atomicXor
 * @{
 * @ingroup AtomicsTest
 * `atomicXor(TestType* address, TestType* val)` -
 * performs atomic bitwise XOR between address and val, returns old value.
 */

// Helper function to run atomicXor tests for same address (single kernel)
template <typename TestType>
static void runAtomicXorSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kXor>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicXor tests for adjacent addresses (single kernel)
template <typename TestType>
static void runAtomicXorAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kXor>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicXor tests for scattered addresses (single kernel)
template <typename TestType>
static void runAtomicXorScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kXor>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run atomicXor tests for same address (multiple kernels)
template <typename TestType>
static void runAtomicXorMultiKernelSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kXor>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicXor tests for adjacent addresses (multiple kernels)
template <typename TestType>
static void runAtomicXorMultiKernelAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kXor>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicXor tests for scattered addresses (multiple kernels)
template <typename TestType>
static void runAtomicXorMultiKernelScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kXor>(
          2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicXor from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicXor_Positive_SameAddress") {
  SECTION("int") { runAtomicXorSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicXorSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicXorSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicXorSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicXor from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicXor_Positive_Adjacent_Addresses") {
  SECTION("int") { runAtomicXorAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicXorAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicXorAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicXorAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicXor from multiple threads on the scattered addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicXor_Positive_Scattered_Addresses") {
  SECTION("int") { runAtomicXorScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicXorScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicXorScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicXorScatteredAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicXor from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicXor_Positive_Multi_Kernel_Same_Address) {
  SECTION("int") { runAtomicXorMultiKernelSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicXorMultiKernelSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicXorMultiKernelSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicXorMultiKernelSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicXor from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicXor_Positive_Multi_Kernel_Adjacent_Addresses) {
  SECTION("int") { runAtomicXorMultiKernelAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicXorMultiKernelAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicXorMultiKernelAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicXorMultiKernelAdjacentAddressesTest<unsigned long long>();
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicXor from multiple threads on the scattered addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicXor_Positive_Multi_Kernel_Scattered_Addresses) {
  SECTION("int") { runAtomicXorMultiKernelScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicXorMultiKernelScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicXorMultiKernelScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicXorMultiKernelScatteredAddressesTest<unsigned long long>();
  }
}

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicXor with invalid parameters.
 *  - Compiles the source with RTC.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicXor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicXor_Negative_Parameters_RTC) {
  hiprtcProgram program{};

  const auto program_source =
      GENERATE(kAtomicXor_int, kAtomicXor_uint, kAtomicXor_ulong, kAtomicXor_ulonglong);
  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "atomicXor_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};
  // Please check the content of negative_kernels_rtc.hh
  int expected_error_count{10};
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
 * End doxygen group AtomicsTest.
 * @}
 */
