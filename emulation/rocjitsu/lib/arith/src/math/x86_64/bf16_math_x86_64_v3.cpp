// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 BF16 native math and inherited-model packing with shared modifiers.
#include "common/compiler.h"
#include "common/x86_64/alu_x86_64_v3.h"
#include "math/x86_64/bf16_math.h"
#include "math/x86_64/tanh_loose_impl.h"
#include "math/x86_64/unary_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {
// Widen the selected BF16 source halves and apply ABS/NEG.
[[gnu::always_inline]] inline __m256 load_input(const uint32_t *a, uint32_t mode) {
  auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
  bits = mode & RJ_ARITH_ALU_HIGH_A ? _mm256_and_si256(bits, _mm256_set1_epi32(-65536))
                                    : _mm256_slli_epi32(bits, 16);
  const auto keep = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  const auto flip = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_and_si256(bits, keep), flip));
}

// Apply inherited BF16 scaling/clamping, truncate, and preserve all unwritten storage.
[[gnu::always_inline]] inline void store_output(uint32_t *d, __m256 value, uint32_t mode,
                                                uint32_t exec_mask) {
  const float scales[] = {1, 2, 4, 0.5f};
  const auto scale = _mm256_set1_ps(scales[(mode >> 6) & 3]);
  const auto keep_destination = _mm256_set1_epi32(mode & RJ_ARITH_ALU_HIGH_D ? 65535 : -65536);
  auto bits = _mm256_castps_si256(value);
  if (mode & RJ_ARITH_ALU_OMOD_MASK) {
    bits = _mm256_castps_si256(_mm256_mul_ps(_mm256_castsi256_ps(bits), scale));
    const auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
    bits = _mm256_andnot_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x800000), magnitude), bits);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    bits = _mm256_castps_si256(clamp_f32(_mm256_castsi256_ps(bits), false));
  }
  // Truncate, never RNE: AVX512BF16 narrowing would change the legacy numeric contract.
  bits = mode & RJ_ARITH_ALU_HIGH_D ? _mm256_and_si256(bits, _mm256_set1_epi32(-65536))
                                    : _mm256_srli_epi32(bits, 16);
  bits = _mm256_or_si256(
      bits,
      _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(d)), keep_destination));
  auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask)),
                                          _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
  _mm256_maskstore_epi32(reinterpret_cast<int *>(d), lane_exec_mask, bits);
}

// Reuse the FP32 vector algorithms; BF16 guest FP controls do not select another model.
template <Bf16MathOp Op> [[gnu::always_inline]] inline __m256 native_value(__m256 value) {
  if constexpr (Op == Bf16MathOp::Tanh) {
    return _mm256_castsi256_ps((__m256i)tanh_vector<F8, I8>((I8)_mm256_castps_si256(value)));
  } else {
    constexpr Unary op = Op == Bf16MathOp::Rcp    ? Unary::Rcp
                         : Op == Bf16MathOp::Rsq  ? Unary::Rsq
                         : Op == Bf16MathOp::Sqrt ? Unary::Sqrt
                         : Op == Bf16MathOp::Exp  ? Unary::Exp
                                                  : Unary::Log;
    static_assert(Op == Bf16MathOp::Rcp || Op == Bf16MathOp::Rsq || Op == Bf16MathOp::Sqrt ||
                  Op == Bf16MathOp::Exp || Op == Bf16MathOp::Log);
    value = flush_denorm_f32(value);
    return flush_denorm_f32(unary_value<op>(value));
  }
}
} // namespace

// Evaluate a selected FP32 core without status; stage the source before any writes.
int bf16_math_x86_64_v3(Bf16MathCore core, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                        uint32_t *d, const uint32_t *a) {
  uint32_t input[32], output[32]{};
  for (int lane = 0; lane < 32; lane += 8) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(input + lane),
                        _mm256_castps_si256(load_input(a + lane, mode)));
  }
  const uint32_t *sources[] = {input};
  uint32_t *destinations[] = {output};
  const auto semantics = (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY
                             ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                             : RJ_ARITH_SEMANTICS_LOOSE;
  if (int error = core((flags & RJ_ARITH_CPU_MASK) | semantics, exec_mask, 0, destinations, sources,
                       nullptr)) {
    return error;
  }
  for (int lane = 0; lane < 32; lane += 8) {
    store_output(
        d + lane,
        _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(output + lane))),
        mode, exec_mask >> lane);
  }
  return RJ_ARITH_SUCCESS;
}

// Fuse source decoding and BF16 publication around the shared native arithmetic.
template <Bf16MathOp Op>
void bf16_math_loose_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  RJ_ARITH_UNROLL(4)
  for (int lane = 0; lane < 32; lane += 8) {
    store_output(d + lane, native_value<Op>(load_input(a + lane, mode)), mode, exec_mask >> lane);
  }
}

template void bf16_math_loose_x86_64_v3<Bf16MathOp::Rcp>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *);
template void bf16_math_loose_x86_64_v3<Bf16MathOp::Rsq>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *);
template void bf16_math_loose_x86_64_v3<Bf16MathOp::Sqrt>(uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *);
template void bf16_math_loose_x86_64_v3<Bf16MathOp::Exp>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *);
template void bf16_math_loose_x86_64_v3<Bf16MathOp::Log>(uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *);
template void bf16_math_loose_x86_64_v3<Bf16MathOp::Tanh>(uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *);

} // namespace rocjitsu::arith
