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

#include "atomicAnd_negative_kernels_rtc.hh"
#include "bitwise_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup atomicAnd atomicAnd
 * @{
 * @ingroup AtomicsTest
 * `atomicAnd(TestType* address, TestType* val)` -
 * performs atomic bitwise AND between address and val, returns old value.
 */

// Helper function to run atomicAnd tests for same address (single kernel)
template <typename TestType>
static void runAtomicAndSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kAnd>(
          1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicAnd tests for adjacent addresses (single kernel)
template <typename TestType>
static void runAtomicAndAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kAnd>(
          warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicAnd tests for scattered addresses (single kernel)
template <typename TestType>
static void runAtomicAndScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kAnd>(
          warp_size, cache_line_size);
    }
  }
}

// Helper function to run atomicAnd tests for same address (multiple kernels)
template <typename TestType>
static void runAtomicAndMultiKernelSameAddressTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kAnd>(
          2, 1, sizeof(TestType));
    }
  }
}

// Helper function to run atomicAnd tests for adjacent addresses (multiple kernels)
template <typename TestType>
static void runAtomicAndMultiKernelAdjacentAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Adjacent address " << current) {
      Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kAnd>(
          2, warp_size, sizeof(TestType));
    }
  }
}

// Helper function to run atomicAnd tests for scattered addresses (multiple kernels)
template <typename TestType>
static void runAtomicAndMultiKernelScatteredAddressesTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Scattered address " << current) {
      Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kAnd>(
          2, warp_size, cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicAnd from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAnd_Positive_SameAddress") {
  SECTION("int") { runAtomicAndSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicAndSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAndSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAndSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicAnd from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAnd_Positive_Adjacent_Addresses") {
  SECTION("int") { runAtomicAndAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicAndAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAndAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAndAdjacentAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicAnd from multiple threads on the scattered addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAnd_Positive_Scattered_Addresses") {
  SECTION("int") { runAtomicAndScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicAndScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAndScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAndScatteredAddressesTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicAnd from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAnd_Positive_Multi_Kernel_Same_Address") {
  SECTION("int") { runAtomicAndMultiKernelSameAddressTest<int>(); }
  SECTION("unsigned int") { runAtomicAndMultiKernelSameAddressTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAndMultiKernelSameAddressTest<unsigned long>(); }
  SECTION("unsigned long long") { runAtomicAndMultiKernelSameAddressTest<unsigned long long>(); }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicAnd from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAnd_Positive_Multi_Kernel_Adjacent_Addresses") {
  SECTION("int") { runAtomicAndMultiKernelAdjacentAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicAndMultiKernelAdjacentAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAndMultiKernelAdjacentAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicAndMultiKernelAdjacentAddressesTest<unsigned long long>();
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs atomicAnd from multiple threads on the scattered addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAnd_Positive_Multi_Kernel_Scattered_Addresses") {
  SECTION("int") { runAtomicAndMultiKernelScatteredAddressesTest<int>(); }
  SECTION("unsigned int") { runAtomicAndMultiKernelScatteredAddressesTest<unsigned int>(); }
  SECTION("unsigned long") { runAtomicAndMultiKernelScatteredAddressesTest<unsigned long>(); }
  SECTION("unsigned long long") {
    runAtomicAndMultiKernelScatteredAddressesTest<unsigned long long>();
  }
}

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicAnd with invalid parameters.
 *  - Compiles the source with RTC.
 * Test source
 * ------------------------
 *  - unit/atomics/atomicAnd.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAnd_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source =
      GENERATE(kAtomicAnd_int, kAtomicAnd_uint, kAtomicAnd_ulong, kAtomicAnd_ulonglong);
  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "atomicAnd_negative.cc", 0, nullptr, nullptr));
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
