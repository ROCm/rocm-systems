// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_FP_MATH_PROVIDER_PROBE_H_
#define ROCJITSU_TESTS_FP_MATH_PROVIDER_PROBE_H_

#include "rocjitsu/base/rj_compiler.h"

#include <stdint.h>

// Private BUILD_TESTING-only entry point in the actual librocjitsu.so image.
typedef struct rj_test_fp_math_result {
  uint64_t first_init_ns;
  uint64_t elapsed_ns;
  uintptr_t instruction_callback_addr;
  uintptr_t wave_callback_addr;
  uint64_t exec;
  uint32_t tier; // 0=scalar, 1=v3, 2=v4; auto may report v4 without adopting it.
  uint32_t kind; // 0=scalar, 1=AVX2x8, 2=AVX512x8, 3=AVX512x16.
  uint32_t qualified_model;
  uint32_t fenv_preserved;
  uint32_t output_count;
  uint32_t output_words[64];
  uint32_t expected_words[64];
  uint32_t mismatches;
  uint32_t executed;
  char error[256];
} rj_test_fp_math_result;

// operation: 0=EXP, 1=LOG. wave_size: 32 or 64.
// scenario: 0=full EXEC, 1=partial, 2=sparse, 3=empty, 4=in-place,
//           5=partial in-place, 6=SGPR broadcast, 7=inline +1.0 broadcast.
// Only scenario 0 accepts timed iterations; it never overwrites its source.
#ifdef __cplusplus
extern "C" RJ_API_EXPORT int rj_test_fp_math_probe(uint32_t operation, uint32_t scenario,
                                                   uint32_t wave_size, uint32_t iterations,
                                                   rj_test_fp_math_result *out) noexcept;
#endif

#endif // ROCJITSU_TESTS_FP_MATH_PROVIDER_PROBE_H_
