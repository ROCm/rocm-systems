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

#include "atomicExch_common.hh"
#include "atomicExch_negative_kernels_rtc.hh"

/**
 * @addtogroup atomicExch atomicExch
 * @{
 * @ingroup AtomicsTest
 */

// Helper function to run atomicExch tests for same address (compile time)
template <typename TestType> static void runAtomicExchSameAddressCompileTimeTest() {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Positive Same Address" << current) {
      AtomicExchSameAddressTest<TestType, AtomicScopes::device>();
    }
  }
}

// Helper function to run atomicExch tests (single kernel)
template <typename TestType> static void runAtomicExchTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::device>(1, sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::device>(warp_size,
                                                                             sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::device>(warp_size,
                                                                             cache_line_size);
    }
  }
}

// Helper function to run atomicExch tests (multi kernel)
template <typename TestType> static void runAtomicExchMultiKernelTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    DYNAMIC_SECTION("Same address " << current) {
      AtomicExchSingleDeviceMultipleKernelTest<TestType, AtomicScopes::device>(2, 1,
                                                                               sizeof(TestType));
    }

    DYNAMIC_SECTION("Adjacent addresses " << current) {
      AtomicExchSingleDeviceMultipleKernelTest<TestType, AtomicScopes::device>(2, warp_size,
                                                                               sizeof(TestType));
    }

    DYNAMIC_SECTION("Scattered addresses " << current) {
      AtomicExchSingleDeviceMultipleKernelTest<TestType, AtomicScopes::device>(2, warp_size,
                                                                               cache_line_size);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel wherein all threads will perform an atomic exchange in the same(compile
 * time deducible) memory location. Each thread will exchange its own grid wide linear index + 1
 * into the memory location, storing the return value into a separate output array slot
 * corresponding to it. Once complete, the union of output array and exchange memory is validated to
 * contain all values in the range [0, number_of_threads].
 *
 *    - The test is run for:
 *      - All overloads of atomicExch
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated exchange memory
 *      - Exchange memory located in shared memory
 *      - Several grid and block dimension combinations(only one block is used for shared memory)
 * Test source
 * ------------------------
 *    - unit/atomics/atomicExch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicExch_Positive_Same_Address_Compile_Time") {
  SECTION("int") { runAtomicExchSameAddressCompileTimeTest<int>(); }
  SECTION("unsigned int") { runAtomicExchSameAddressCompileTimeTest<unsigned int>(); }
#ifndef HT_NVIDIA
  SECTION("unsigned long") { runAtomicExchSameAddressCompileTimeTest<unsigned long>(); }
#endif
  SECTION("unsigned long long") { runAtomicExchSameAddressCompileTimeTest<unsigned long long>(); }
  SECTION("float") { runAtomicExchSameAddressCompileTimeTest<float>(); }
#ifndef HT_NVIDIA
  SECTION("double") { runAtomicExchSameAddressCompileTimeTest<double>(); }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Executes a single kernel on a single device wherein all threads will perform an atomic
 * exchange into a runtime determined memory location. Each thread will exchange its own grid wide
 * linear index + offset into the memory location, storing the return value into a separate output
 * array slot corresponding to it. Once complete, the union of output array and exchange memory is
 * validated to contain all values in the range [0, number_of_threads +
 * number_of_exchange_memory_slots). Several memory access patterns are tested:
 *      -# All threads exchange to a single memory location
 *      -# Each thread exchanges into an array containing warp_size elements, using tid % warp_size
 *         for indexing
 *      -# Same as the above, but the exchange elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicExch
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated exchange memory
 *      - Exchange memory located in shared memory
 *      - Several grid and block dimension combinations(only one block is used for shared memory)
 * Test source
 * ------------------------
 *    - unit/atomics/atomicExch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicExch_Positive") {
  SECTION("int") { runAtomicExchTest<int>(); }
  SECTION("unsigned int") { runAtomicExchTest<unsigned int>(); }
#ifndef HT_NVIDIA
  SECTION("unsigned long") { runAtomicExchTest<unsigned long>(); }
#endif
  SECTION("unsigned long long") { runAtomicExchTest<unsigned long long>(); }
  SECTION("float") { runAtomicExchTest<float>(); }
#ifndef HT_NVIDIA
  SECTION("double") { runAtomicExchTest<double>(); }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Executes a kernel two times concurrently on a single device wherein all threads will perform
 * an atomic exchange into a runtime determined memory location. Each thread will exchange its own
 * grid wide linear index + offset into the memory location, storing the return value into a
 * separate output array slot corresponding to it. Once complete, the union of output array and
 * exchange memory is validated to contain all values in the range [0, number_of_threads +
 * number_of_exchange_memory_slots). Several memory access patterns are tested:
 *      -# All threads exchange to a single memory location
 *      -# Each thread exchanges into an array containing warp_size elements, using tid % warp_size
 * for indexing
 *      -# Same as the above, but the exchange elements are spread out by L1 cache line size bytes.
 *
 *    - The test is run for:
 *      - All overloads of atomicExch
 *      - hipMalloc, hipMallocManaged, hipHostMalloc and hipHostRegister allocated exchange memory
 *      - Several grid and block dimension combinations
 * Test source
 * ------------------------
 *    - unit/atomics/atomicExch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicExch_Positive_Multi_Kernel") {
  SECTION("int") { runAtomicExchMultiKernelTest<int>(); }
  SECTION("unsigned int") { runAtomicExchMultiKernelTest<unsigned int>(); }
#ifndef HT_NVIDIA
  SECTION("unsigned long") { runAtomicExchMultiKernelTest<unsigned long>(); }
#endif
  SECTION("unsigned long long") { runAtomicExchMultiKernelTest<unsigned long long>(); }
  SECTION("float") { runAtomicExchMultiKernelTest<float>(); }
#ifndef HT_NVIDIA
  SECTION("double") { runAtomicExchMultiKernelTest<double>(); }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass combinations of arguments of invalid types for all overloads of
 * atomicExch
 * Test source
 * ------------------------
 *    - unit/atomics/atomicExch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicExch_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source = GENERATE(kAtomicExchInt, kAtomicExchUnsignedInt, kAtomicExchULL,
                                       kAtomicExchFloat, kAtomicExchDouble);
  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "atomicExch_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  int expected_error_count{8};
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
