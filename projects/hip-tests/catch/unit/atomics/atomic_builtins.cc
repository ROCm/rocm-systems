/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

#include "atomic_builtins_kernels_rtc.hh"

/**
 * @addtogroup __hip_atomic_fetch_add __hip_atomic_fetch_add
 * @{
 * @ingroup AtomicsTest
 */

void AtomicBuiltinsRTCWrapper(const char* program_source, int expected_errors_num,
                              int expected_warnings_num) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, program_source, "atomics_builtins_kernels.cc", 0,
                                   nullptr, nullptr));

  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};
  int warning_count{0};

  std::string error_message{"error:"};
  std::string warning_message{"warning:"};

  size_t npos_e = log.find(error_message, 0);
  while (npos_e != std::string::npos) {
    ++error_count;
    npos_e = log.find(error_message, npos_e + 1);
  }

  size_t npos_w = log.find(warning_message, 0);
  while (npos_w != std::string::npos) {
    ++warning_count;
    npos_w = log.find(warning_message, npos_w + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_errors_num);
  REQUIRE(warning_count == expected_warnings_num);
}

/**
 * Test Description
 * ------------------------
 *    - Compiles atomic builtins while passing parameters that shall cause:
 *        -# Compiler warnings
 *        -# Compiler errors
 * Test source
 * ------------------------
 *    - unit/atomics/atomic_builtins.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_AtomicBuiltins_Negative_Parameters_RTC) {
  AtomicBuiltinsRTCWrapper(kBuiltinStore, 5, 5);
  AtomicBuiltinsRTCWrapper(kBuiltinLoad, 4, 4);
  AtomicBuiltinsRTCWrapper(kBuiltinCompExWeak, 5, 7);
  AtomicBuiltinsRTCWrapper(kBuiltinCompExStrong, 5, 7);
  AtomicBuiltinsRTCWrapper(kBuiltinExchange, 5, 2);
  AtomicBuiltinsRTCWrapper(kBuiltinFetchAdd, 5, 2);
  AtomicBuiltinsRTCWrapper(kBuiltinFetchAnd, 7, 2);
  AtomicBuiltinsRTCWrapper(kBuiltinFetchOr, 7, 2);
  AtomicBuiltinsRTCWrapper(kBuiltinFetchXor, 7, 2);
  AtomicBuiltinsRTCWrapper(kBuiltinFetchMax, 5, 2);
  AtomicBuiltinsRTCWrapper(kBuiltinFetchMin, 5, 2);
}

/**
 * End doxygen group AtomicsTest.
 * @}
 */
