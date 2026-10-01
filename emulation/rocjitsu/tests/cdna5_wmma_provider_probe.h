// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_CDNA5_WMMA_PROVIDER_PROBE_H_
#define ROCJITSU_TESTS_CDNA5_WMMA_PROVIDER_PROBE_H_

#include "rocjitsu/base/rj_compiler.h"

#include <stdint.h>

// Private, BUILD_TESTING-only ABI. The implementation is compiled into
// librocjitsu.so, so it decodes and executes with that image's ISA backend.
enum { RJ_TEST_WMMA_MAX_OUTPUT_WORDS = 8 * 32, RJ_TEST_WMMA_MAX_VGPRS = 256 };

typedef struct rj_test_cdna5_wmma_result {
  uint64_t elapsed_ns;
  // Decoder creation plus one decode. Only the first call in a fresh process
  // contains the provider's cold load/selection cost.
  uint64_t first_decode_ns;
  uintptr_t callback_addr;
  uint32_t output_count;
  uint32_t output_words[RJ_TEST_WMMA_MAX_OUTPUT_WORDS];
  uint32_t executed;
  // Union of observed accesses by logical VGPR. Event counts can legitimately
  // differ between scalar and vector implementations of the same instruction.
  uint64_t read_lanes[RJ_TEST_WMMA_MAX_VGPRS];
  uint64_t write_lanes[RJ_TEST_WMMA_MAX_VGPRS];
  uint8_t read_bytes[RJ_TEST_WMMA_MAX_VGPRS];
  uint8_t write_bytes[RJ_TEST_WMMA_MAX_VGPRS];
  char error[160];
} rj_test_cdna5_wmma_result;

// form: 0=F32/F16, 1=F16/F16, 2=F32/BF16, 3=BF16/BF16,
//       4=BF16F32/BF16 (all 16x16x32).
// scenario: 0=independent operands (benchmark), 1=dst aliases A,
//           2=dst aliases B, 3=dst aliases C, 4=non-finite input,
//           5=negate C, 6=absolute C, 7=absolute then negate C,
//           8=inline constant +1.0 C. Modifier scenarios are valid only for
//           the F32-output and BF16F32 forms (0, 2, and 4).
//           9..14=distinct signed NaN payloads in A/B/C.
// Nonzero iterations are accepted only for scenario 0, whose source windows
// remain unchanged across executions. Returns zero on success.
#ifdef __cplusplus
extern "C" RJ_API_EXPORT int rj_test_cdna5_wmma_probe(uint32_t form, uint32_t scenario,
                                                      uint32_t iterations,
                                                      rj_test_cdna5_wmma_result *out) noexcept;
// Observe one execution in the actual shared image, excluding setup and output
// inspection. Disjoint D is poisoned to detect accidental accumulator reads.
extern "C" RJ_API_EXPORT int rj_test_cdna5_wmma_observe(uint32_t form, uint32_t scenario,
                                                        rj_test_cdna5_wmma_result *out) noexcept;
#endif

#endif // ROCJITSU_TESTS_CDNA5_WMMA_PROVIDER_PROBE_H_
