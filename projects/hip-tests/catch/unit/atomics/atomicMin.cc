/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "atomicMin_negative_kernels_rtc.hh"
#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicMin atomicMin
 * @{
 * @ingroup AtomicsTest
 * `atomicMin(TestType* address, TestType* val)` -
 * calculates minimum between address and val, returns old value.
 */

// Helper function to run atomicMin tests for same address (single kernel)
template <typename TestType>
static void runAtomicMinSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kMin>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMin tests for adjacent addresses (single kernel)
template <typename TestType>
static void runAtomicMinAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kMin>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMin tests for scattered addresses (single kernel)
template <typename TestType>
static void runAtomicMinScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kMin>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run atomicMin tests for same address (multiple kernels)
template <typename TestType>
static void runAtomicMinMultiKernelSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMin>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMin tests for adjacent addresses (multiple kernels)
template <typename TestType>
static void runAtomicMinMultiKernelAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMin>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMin tests for scattered addresses (multiple kernels)
template <typename TestType>
static void runAtomicMinMultiKernelScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMin>(
          2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicMin_Positive_SameAddress") {
  SECTION("int") { runAtomicMinSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicMinSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMinSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMinSameAddressTest<unsigned long long>(); }
  SECTION("float") { runAtomicMinSameAddressTest<float>(); }
  SECTION("double") { runAtomicMinSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicMin_Positive_Adjacent_Addresses") {
  SECTION("int") { runAtomicMinAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMinAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMinAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMinAdjacentAddressesTest<unsigned long long>(); }
  SECTION("float") { runAtomicMinAdjacentAddressesTest<float>(); }
  SECTION("double") { runAtomicMinAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin from multiple threads on the scaterred addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicMin_Positive_Scattered_Addresses") {
  SECTION("int") { runAtomicMinScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMinScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMinScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMinScatteredAddressesTest<unsigned long long>(); }
  SECTION("float") { runAtomicMinScatteredAddressesTest<float>(); }
  SECTION("double") { runAtomicMinScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMin_Positive_Multi_Kernel_Same_Address) {
  SECTION("int") { runAtomicMinMultiKernelSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicMinMultiKernelSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMinMultiKernelSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMinMultiKernelSameAddressTest<unsigned long long>(); }
  SECTION("float") { runAtomicMinMultiKernelSameAddressTest<float>(); }
  SECTION("double") { runAtomicMinMultiKernelSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMin_Positive_Multi_Kernel_Adjacent_Addresses) {
  SECTION("int") { runAtomicMinMultiKernelAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMinMultiKernelAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMinMultiKernelAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicMinMultiKernelAdjacentAddressesTest<unsigned long long>();
  }
  SECTION("float") { runAtomicMinMultiKernelAdjacentAddressesTest<float>(); }
  SECTION("double") { runAtomicMinMultiKernelAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMin from multiple threads on the scaterred addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMin_Positive_Multi_Kernel_Scattered_Addresses) {
  SECTION("int") { runAtomicMinMultiKernelScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMinMultiKernelScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMinMultiKernelScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicMinMultiKernelScatteredAddressesTest<unsigned long long>();
  }
  SECTION("float") { runAtomicMinMultiKernelScatteredAddressesTest<float>(); }
  SECTION("double") { runAtomicMinMultiKernelScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicMin with invalid parameters.
 *  - Compiles the source with RTC.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMin.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMin_Negative_Parameters_RTC) {
  hiprtcProgram program{};

  const auto program_source = GENERATE(kAtomicMin_int, kAtomicMin_uint, kAtomicMin_ulong,
                                       kAtomicMin_ulonglong, kAtomicMin_float, kAtomicMin_double);
  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "atomicMin_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};
  // Please check the content of negative_kernels_rtc.hh
  int expected_error_count{7};
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
