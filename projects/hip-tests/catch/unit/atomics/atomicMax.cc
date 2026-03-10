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

#include "atomicMax_negative_kernels_rtc.hh"
#include "min_max_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicMax atomicMax
 * @{
 * @ingroup AtomicsTest
 * `atomicMax(TestType* address, TestType* val)` -
 * calculates maximum between address and val, returns old value.
 */

// Helper function to run atomicMax tests for same address (single kernel)
template <typename TestType>
static void runAtomicMaxSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kMax>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMax tests for adjacent addresses (single kernel)
template <typename TestType>
static void runAtomicMaxAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kMax>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMax tests for scattered addresses (single kernel)
template <typename TestType>
static void runAtomicMaxScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kMax>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run atomicMax tests for same address (multiple kernels)
template <typename TestType>
static void runAtomicMaxMultiKernelSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMax>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMax tests for adjacent addresses (multiple kernels)
template <typename TestType>
static void runAtomicMaxMultiKernelAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMax>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicMax tests for scattered addresses (multiple kernels)
template <typename TestType>
static void runAtomicMaxMultiKernelScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kMax>(
          2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicMax_Positive_SameAddress") {
  SECTION("int") { runAtomicMaxSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicMaxSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMaxSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMaxSameAddressTest<unsigned long long>(); }
  SECTION("float") { runAtomicMaxSameAddressTest<float>(); }
  SECTION("double") { runAtomicMaxSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicMax_Positive_Adjacent_Addresses") {
  SECTION("int") { runAtomicMaxAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMaxAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMaxAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMaxAdjacentAddressesTest<unsigned long long>(); }
  SECTION("float") { runAtomicMaxAdjacentAddressesTest<float>(); }
  SECTION("double") { runAtomicMaxAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax from multiple threads on the scaterred addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicMax_Positive_Scattered_Addresses") {
  SECTION("int") { runAtomicMaxScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMaxScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMaxScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMaxScatteredAddressesTest<unsigned long long>(); }
  SECTION("float") { runAtomicMaxScatteredAddressesTest<float>(); }
  SECTION("double") { runAtomicMaxScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMax_Positive_Multi_Kernel_Same_Address) {
  SECTION("int") { runAtomicMaxMultiKernelSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicMaxMultiKernelSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMaxMultiKernelSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicMaxMultiKernelSameAddressTest<unsigned long long>(); }
  SECTION("float") { runAtomicMaxMultiKernelSameAddressTest<float>(); }
  SECTION("double") { runAtomicMaxMultiKernelSameAddressTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMax_Positive_Multi_Kernel_Adjacent_Addresses) {
  SECTION("int") { runAtomicMaxMultiKernelAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMaxMultiKernelAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMaxMultiKernelAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicMaxMultiKernelAdjacentAddressesTest<unsigned long long>();
  }
  SECTION("float") { runAtomicMaxMultiKernelAdjacentAddressesTest<float>(); }
  SECTION("double") { runAtomicMaxMultiKernelAdjacentAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicMax from multiple threads on the scaterred addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMax_Positive_Multi_Kernel_Scattered_Addresses) {
  SECTION("int") { runAtomicMaxMultiKernelScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicMaxMultiKernelScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicMaxMultiKernelScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicMaxMultiKernelScatteredAddressesTest<unsigned long long>();
  }
  SECTION("float") { runAtomicMaxMultiKernelScatteredAddressesTest<float>(); }
  SECTION("double") { runAtomicMaxMultiKernelScatteredAddressesTest<double>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicMax with invalid parameters.
 *  - Compiles the source with RTC.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_atomicMax_Negative_Parameters_RTC) {
  hiprtcProgram program{};

  const auto program_source = GENERATE(kAtomicMax_int, kAtomicMax_uint, kAtomicMax_ulong,
                                       kAtomicMax_ulonglong, kAtomicMax_float, kAtomicMax_double);
  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "atomicMax_negative.cc", 0, nullptr, nullptr));
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
