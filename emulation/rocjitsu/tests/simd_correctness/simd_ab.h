// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Scalar-vs-SIMD A/B harness for the *_simd_correctness_test suites.
//
// `util::force_scalar()` is process-wide and immutable (read once from
// RJ_FORCE_SCALAR), so a single test process exercises exactly one mode and
// can no longer compare the scalar and SIMD execute paths in-process. The
// equivalence check instead runs the suite twice and diffs a deterministic
// per-case result dump:
//
//   RJ_FORCE_SCALAR=1 RJ_SIMD_DUMP=scalar.txt rocjitsu_tests --gtest_filter=...
//                     RJ_SIMD_DUMP=simd.txt   rocjitsu_tests --gtest_filter=...
//   diff <(sort scalar.txt) <(sort simd.txt)   # any divergence == failure
//
// (wired as the `simd_ab_diff` CTest entry). Each correctness test calls
// `record()` with its computed destination words; the two runs must emit
// byte-identical dumps unless the SIMD fast path diverged from the scalar
// generated body -- which is exactly what the diff catches.

#ifndef ROCJITSU_TESTS_SIMD_AB_H_
#define ROCJITSU_TESTS_SIMD_AB_H_

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace simd_ab {

/// True when RJ_SIMD_DUMP names a file: this process is one half of an A/B
/// run and each case's results must be recorded for the external diff. When
/// false (a plain `ctest` invocation) record() is a no-op and the tests rely
/// on their in-process structural invariants only.
bool dumping();

/// Append one case's per-lane destination words to the dump, tagged by the
/// current gtest test name plus `sublabel` and the `exec` mask. Deterministic
/// and ordered; identical across the scalar and SIMD runs unless the SIMD
/// path diverged. No-op when !dumping().
void record(std::string_view sublabel, uint64_t exec, const uint32_t *dst, std::size_t n);

} // namespace simd_ab

#endif // ROCJITSU_TESTS_SIMD_AB_H_
