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

#include "atomicOr_negative_kernels_rtc.hh"
#include "bitwise_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicOr atomicOr
 * @{
 * @ingroup AtomicsTest
 * `atomicOr(TestType* address, TestType* val)` -
 * performs atomic bitwise OR between address and val, returns old value.
 */

// Helper function to run atomicOr tests for same address (single kernel)
template <typename TestType>
static void runAtomicOrSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicOr tests for adjacent addresses (single kernel)
template <typename TestType>
static void runAtomicOrAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicOr tests for scattered addresses (single kernel)
template <typename TestType>
static void runAtomicOrScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run atomicOr tests for same address (multiple kernels)
template <typename TestType>
static void runAtomicOrMultiKernelSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicOr tests for adjacent addresses (multiple kernels)
template <typename TestType>
static void runAtomicOrMultiKernelAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicOr tests for scattered addresses (multiple kernels)
template <typename TestType>
static void runAtomicOrMultiKernelScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(
          2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicOr from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicOr_Positive_SameAddress") {
  SECTION("int") { runAtomicOrSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicOrSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicOrSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicOrSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicOr from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicOr_Positive_Adjacent_Addresses") {
  SECTION("int") { runAtomicOrAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicOrAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicOrAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicOrAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicOr from multiple threads on the scattered addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicOr_Positive_Scattered_Addresses") {
  SECTION("int") { runAtomicOrScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicOrScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicOrScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicOrScatteredAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicOr from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicOr_Positive_Multi_Kernel_Same_Address") {
  SECTION("int") { runAtomicOrMultiKernelSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicOrMultiKernelSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicOrMultiKernelSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicOrMultiKernelSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicOr from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicOr_Positive_Multi_Kernel_Adjacent_Addresses") {
  SECTION("int") { runAtomicOrMultiKernelAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicOrMultiKernelAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicOrMultiKernelAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicOrMultiKernelAdjacentAddressesTest<unsigned long long>();
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicOr from multiple threads on the scattered addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicOr_Positive_Multi_Kernel_Scattered_Addresses") {
  SECTION("int") { runAtomicOrMultiKernelScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicOrMultiKernelScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicOrMultiKernelScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicOrMultiKernelScatteredAddressesTest<unsigned long long>();
  }
}

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicAnd with invalid parameters.
 *  - Compiles the source with RTC.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicOr.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicOr_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source =
      GENERATE(kAtomicOr_int, kAtomicOr_uint, kAtomicOr_ulong, kAtomicOr_ulonglong);
  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "atomicOr_negative.cc", 0, nullptr, nullptr));
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
