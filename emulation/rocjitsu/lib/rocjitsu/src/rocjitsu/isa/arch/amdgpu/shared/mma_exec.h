// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_

/// @file Shared Matrix Multiply-Accumulate (MMA) register mapping and execution.
///
/// Implements the GFX9 MFMA register layout formulas from the AMD Matrix
/// Instruction Calculator (InstCalcGfx9). Shared across CDNA1-4 (all use the
/// same GFX9 encoding family for MFMA).
///
/// AccMode governs how the accumulator source (src2) is resolved:
///   - Unified:  CDNA3/4 — VGPR and AccVGPR share a single file; encoding
///               range 256-511 = VGPR, 768-1023 = AccVGPR (alias).
///   - Separate: CDNA2 — dedicated AccVGPR file, src2 ranges differ.
///   - VgprOnly: CDNA1 — no AccVGPR; src2 is always a VGPR or constant.
///
/// Key conventions:
///   - Output D[i][j]: i is the register dimension (matrix column),
///     j is the lane dimension (matrix row). Call output_loc with
///     i=column, j=row to get the physical (vgpr_offset, lane).
///   - Input A[row][k]: use input_loc(dim=M, K, B, i=row, k, b, bits).
///     Input B[col][k]: use input_loc(dim=N, K, B, i=col, k, b, bits).

#include "rocjitsu/isa/arch/amdgpu/shared/mma/common.h"

#include "rocjitsu/isa/arch/amdgpu/shared/mma/select.h"

namespace rocjitsu {
namespace amdgpu {
/// Generic MFMA execute for f32 output: D = C + A x B.
///
/// All inputs are read before any outputs are written to avoid WAR hazards
/// when destination registers overlap source registers.
///
/// @param cbsz  A-matrix broadcast block size (0 = no broadcast).
/// @param abid  A-matrix broadcast source block ID.
/// @param blgp  B-matrix lane group permutation pattern.
template <typename ExtractA, typename ExtractB>
void exec_f32_mixed(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t a_bits,
                    uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                    ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR, uint32_t cbsz = 0,
                    uint32_t abid = 0, uint32_t blgp = 0) {
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);

  constexpr size_t MAX_AB = 2048;      // max M*K over all MFMA shapes
  constexpr size_t MAX_BSTRIDE = 4096; // max K*stride over all MFMA shapes
  constexpr size_t MAX_C = 1024;       // max M*stride over all MFMA shapes
  static_assert((MAX_AB + MAX_BSTRIDE + MAX_C) * sizeof(float) <= 48 * 1024,
                "MFMA staging buffers exceed the 48 KiB stack budget");

  auto stage_operands = [&](uint32_t block, uint32_t b_stride, float *a_values, float *b_values) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t k = 0; k < K; ++k) {
        auto al = input_loc(M, K, B, row, k, block, a_bits);
        if (cbsz != 0)
          al.lane = permute_a_lane(al.lane, cbsz, abid);
        a_values[static_cast<size_t>(row) * K + k] = ea(cu, s0, physicalize_loc(al, wf));
      }
    }
    for (uint32_t k = 0; k < K; ++k) {
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = input_loc(N, K, B, col, k, block, b_bits);
        if (blgp != 0)
          bl.lane = permute_b_lane(bl.lane, blgp);
        b_values[static_cast<size_t>(k) * b_stride + col] = eb(cu, s1, physicalize_loc(bl, wf));
      }
    }
  };

  // Scalar reference: D[i][j] = C[i][j] + sum_k A[i][k] * B[k][j], accumulated
  // per output in K order (non-fused multiply-add).
  auto run_scalar = [&]() {
    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    // Real ISA shapes use bounded per-block stack storage. Preserve support
    // for direct callers with larger dimensions through one overflow buffer.
    alignas(64) float a_stack[MAX_AB];
    alignas(64) float b_stack[MAX_BSTRIDE];
    std::vector<float> overflow;
    float *a_values = a_stack;
    float *b_values = b_stack;
    if (a_count > MAX_AB || b_count > MAX_BSTRIDE) {
      overflow.resize(a_count + b_count);
      a_values = overflow.data();
      b_values = overflow.data() + a_count;
    }

    for (uint32_t b = 0; b < B; ++b) {
      stage_operands(b, N, a_values, b_values);
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          // AMD convention: i=row (register dimension), j=col (lane dimension).
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          float acc =
              (const_acc != ACC_FROM_VGPR)
                  ? std::bit_cast<float>(const_acc)
                  : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
          for (uint32_t k = 0; k < K; ++k) {
            acc += a_values[static_cast<size_t>(row) * K + k] *
                   b_values[static_cast<size_t>(k) * N + col];
          }
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
        }
      }
    }
  };

  // SIMD fast path. Works for any MFMA shape and any f32-producing extract
  // (f16/bf16/fp8/bf8/f32). Per block: hoist A into row-major (row,k), B into
  // row-major (k,col), and C into (row,col) dense f32 buffers (lane
  // permutation folded in during the hoist), then run the dense MxNxK matmul
  // as native-width FMA rows over the N (column) dimension. Uses fused FMA,
  // matching the GFX9 MFMA hardware's single-rounding MACs (the scalar path
  // above is non-fused; results agree to a few ULP). N columns that don't
  // fill a full SIMD lane group fall to a scalar (fused) tail.
  if constexpr (util::has_stdx_simd) {
    // Pad the column (N) leading dimension up to a SIMD-width multiple so
    // every matmul row starts W-aligned: the inner loop then uses aligned
    // loads/stores for any N, and the staging buffers live on the stack
    // (no per-call heap allocation). MAX_* bound every real MFMA shape;
    // anything larger (or a forced-scalar run) falls back to the scalar path.
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * stride > MAX_C) {
      run_scalar();
    } else {
      // Zero-initialized as the uniform staging-buffer convention (see the
      // zero-init policy on wmma_simd_matmul). C is pre-seeded just below, so the
      // init is redundant here, but matching the WMMA paths keeps one rule.
      alignas(64) float Abuf[MAX_AB] = {};
      alignas(64) float Bbuf[MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        stage_operands(b, stride, Abuf, Bbuf);
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            Cbuf[row * stride + col] =
                (const_acc != ACC_FROM_VGPR)
                    ? std::bit_cast<float>(const_acc)
                    : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
          }
        wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            results.push_back(
                {out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[row * stride + col])});
          }
      }
    }
  } else {
    run_scalar();
  }

  bool has_nan = false;
  for (const auto &r : results) {
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
    float fval = std::bit_cast<float>(r.val);
    if (std::isnan(fval) || std::isinf(fval))
      has_nan = true;
  }
  if (has_nan) {
    util::Logger::vm([&](auto &os) {
      os << std::format("MFMA_NAN_DETECTED dst=v{} s0=v{} s1=v{} s2=v{} M={} N={} K={}", dst, s0,
                        s1, s2, M, N, K);
      for (const auto &r : results) {
        float fval = std::bit_cast<float>(r.val);
        if (std::isnan(fval) || std::isinf(fval))
          os << std::format("\n[rj log VM]   reg={} lane={} val={:#x}({}) "
                            "a=[{:#x},{:#x}] b=[{:#x},{:#x}]",
                            r.reg, r.lane, r.val, fval, RegisterAccess(cu).read_vgpr(s0, r.lane),
                            RegisterAccess(cu).read_vgpr(s0 + 1, r.lane),
                            RegisterAccess(cu).read_vgpr(s1, r.lane),
                            RegisterAccess(cu).read_vgpr(s1 + 1, r.lane));
      }
    });
  }
  util::Logger::vm([&](auto &os) {
    static thread_local uint64_t mfma_count = 0;
    if (++mfma_count > 30)
      return;
    os << std::format("MFMA_F32 #{} M={} N={} K={} B={} dst=v{} s0=v{} s1=v{} s2=v{}", mfma_count,
                      M, N, K, B, dst, s0, s1, s2);
    for (uint32_t ln : {0u, 1u, 4u, 8u, 16u, 31u, 32u, 48u, 63u}) {
      os << std::format(
          "\n[rj log VM]   L{}: s0=[{:#x},{:#x},{:#x},{:#x}]"
          " s1=[{:#x},{:#x},{:#x},{:#x}]"
          " out=[{:#x},{:#x},{:#x},{:#x}]",
          ln, RegisterAccess(cu).read_vgpr(s0, ln), RegisterAccess(cu).read_vgpr(s0 + 1, ln),
          RegisterAccess(cu).read_vgpr(s0 + 2, ln), RegisterAccess(cu).read_vgpr(s0 + 3, ln),
          RegisterAccess(cu).read_vgpr(s1, ln), RegisterAccess(cu).read_vgpr(s1 + 1, ln),
          RegisterAccess(cu).read_vgpr(s1 + 2, ln), RegisterAccess(cu).read_vgpr(s1 + 3, ln),
          RegisterAccess(cu).read_vgpr(dst, ln), RegisterAccess(cu).read_vgpr(dst + 1, ln),
          RegisterAccess(cu).read_vgpr(dst + 2, ln), RegisterAccess(cu).read_vgpr(dst + 3, ln));
    }
  });
}

template <typename ExtractA, typename ExtractB>
void exec_f32(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t in_bits,
              uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
              uint32_t const_acc = ACC_FROM_VGPR, uint32_t cbsz = 0, uint32_t abid = 0,
              uint32_t blgp = 0) {
  exec_f32_mixed(cu, M, N, K, B, in_bits, in_bits, dst, s0, s1, s2, ea, eb, const_acc, cbsz, abid,
                 blgp);
}

/// Column (N) leading-dimension pitch rounding A/B/C buffers up to a SIMD-width
/// multiple, so every matmul row starts W-aligned. Bounds every real WMMA shape
/// (M <= 32, N <= 16, K <= 128); anything larger falls back to the scalar path.
constexpr size_t WMMA_SIMD_MAX_AB = 4096;      // max M*K
constexpr size_t WMMA_SIMD_MAX_BSTRIDE = 4096; // max K*stride
constexpr size_t WMMA_SIMD_MAX_C = 1024;       // max M*stride
// Combined stack frame for the WMMA/SWMMAC staging buffers (currently 28 KiB for
// the float and int32 paths); tripwire against silent stack-frame growth.
static_assert((WMMA_SIMD_MAX_AB + WMMA_SIMD_MAX_BSTRIDE + WMMA_SIMD_MAX_C) * sizeof(float) <=
                  48 * 1024,
              "WMMA SIMD staging buffers exceed the 48 KiB stack budget");

/// GFX9-layout packed 16-bit output execution: D = C + A x B, result packed f16/bf16.
/// Uses GFX9 input_loc for inputs and output_loc_16 (derived from output_loc_32)
/// for outputs. The accumulator is read/written in packed 16-bit format.
template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
void exec_packed16_gfx9(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t in_bits,
                        uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea,
                        ExtractB eb, ReadAcc read_acc, PackResult pack_result,
                        uint32_t const_acc = ACC_FROM_VGPR) {
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = physicalize_packed_out(output_loc_16(M, N, row, col, b), wf);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
        for (uint32_t k = 0; k < K; ++k) {
          auto al = physicalize_loc(input_loc(M, K, B, row, k, b, in_bits), wf);
          auto bl = physicalize_loc(input_loc(N, K, B, col, k, b, in_bits), wf);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
      }
    }
  }

  // Determine physical VGPR count for the packed output.
  uint32_t max_reg = 0;
  for (const auto &r : results)
    if (r.reg > max_reg)
      max_reg = r.reg;
  uint32_t dst_regs = max_reg + 1;

  std::vector<uint32_t> words(dst_regs * wf, 0);
  std::vector<uint8_t> masks(dst_regs * wf, 0);
  for (const auto &r : results) {
    uint32_t idx = r.reg * wf + r.lane;
    uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < wf; ++lane) {
      uint32_t idx = reg * wf + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = RegisterAccess(cu).read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      RegisterAccess(cu).write_vgpr(dst + reg, lane, word);
    }
  }
}

/// Convenience wrappers for GFX9 packed f16/bf16 output.
template <typename ExtractA, typename ExtractB>
void exec_f16_gfx9(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t in_bits,
                   uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                   uint32_t const_acc = ACC_FROM_VGPR) {
  exec_packed16_gfx9(
      cu, M, N, K, B, in_bits, dst, s0, s1, s2, ea, eb,
      [](auto &c, uint32_t base, uint32_t lane, uint32_t sub) -> float {
        uint32_t raw = RegisterAccess(c).read_vgpr(base, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float v) -> uint16_t { return util::f32_to_f16(v); }, const_acc);
}

template <typename ExtractA, typename ExtractB>
void exec_bf16_gfx9(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t in_bits,
                    uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                    uint32_t const_acc = ACC_FROM_VGPR) {
  exec_packed16_gfx9(
      cu, M, N, K, B, in_bits, dst, s0, s1, s2, ea, eb,
      [](auto &c, uint32_t base, uint32_t lane, uint32_t sub) -> float {
        uint32_t raw = RegisterAccess(c).read_vgpr(base, lane);
        return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float v) -> uint16_t { return util::f32_to_bf16(v); }, const_acc);
}

inline uint32_t pack_i32_acc(int64_t acc, bool clamp) {
  if (clamp) {
    acc = std::clamp(acc, static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                     static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  }
  return static_cast<uint32_t>(acc);
}

/// GFX9-layout i32 output with configurable input bit-width and extractors.
/// When clamp=false, accumulates in uint32_t (defined wrapping, matching
/// hardware). When clamp=true, accumulates in int64_t and saturates to
/// int32_t range via pack_i32_acc.
template <typename ExtractA, typename ExtractB>
void exec_i32_mixed(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t in_bits,
                    uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                    uint32_t const_acc = ACC_FROM_VGPR, bool clamp = false, uint32_t cbsz = 0,
                    uint32_t abid = 0, uint32_t blgp = 0) {
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
        auto products = [&](uint32_t k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          if (cbsz != 0)
            al.lane = permute_a_lane(al.lane, cbsz, abid);
          if (blgp != 0)
            bl.lane = permute_b_lane(bl.lane, blgp);
          return std::pair{ea(cu, s0, physicalize_loc(al, wf)),
                           eb(cu, s1, physicalize_loc(bl, wf))};
        };
        uint32_t val;
        if (clamp) {
          int64_t acc = (const_acc != ACC_FROM_VGPR)
                            ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                            : static_cast<int64_t>(static_cast<int32_t>(
                                  RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane)));
          for (uint32_t k = 0; k < K; ++k) {
            auto [a, b_val] = products(k);
            acc += static_cast<int64_t>(a) * static_cast<int64_t>(b_val);
          }
          val = pack_i32_acc(acc, true);
        } else {
          uint32_t acc = (const_acc != ACC_FROM_VGPR)
                             ? const_acc
                             : RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane);
          for (uint32_t k = 0; k < K; ++k) {
            auto [a, b_val] = products(k);
            acc += static_cast<uint32_t>(a * b_val);
          }
          val = acc;
        }
        results.push_back({out.reg, out.lane, val});
      }
    }
  }
  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_f32_mixed(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t a_bits,
                         uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                         ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                         uint32_t c_modifier = 0, uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        float acc =
            (const_acc != ACC_FROM_VGPR)
                ? std::bit_cast<float>(const_acc)
                : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
        acc = apply_wmma_c_modifier(acc, c_modifier);
        for (uint32_t k = 0; k < K; ++k) {
          auto al = gfx12_wmma_a_input_loc(wave_size, M, K, row, k, a_bits, b_bits);
          auto bl = gfx12_wmma_b_input_loc(wave_size, N, K, col, k, a_bits, b_bits);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
      }
    }
  };

  // SIMD fast path: hoist A/B/C into dense f32 buffers, then run the dense
  // MxNxK matmul as native-width FMA rows over the N (column) dimension.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      auto reads = read_mixed_matrix_fast_path_regions(cu, s0, s1, s2, static_cast<uint64_t>(M) * K,
                                                       a_bits, static_cast<uint64_t>(N) * K, b_bits,
                                                       static_cast<uint64_t>(M) * N,
                                                       /*acc_bits=*/32, const_acc, wave_size);
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          Cbuf[row * stride + col] =
              apply_wmma_c_modifier((const_acc != ACC_FROM_VGPR)
                                        ? std::bit_cast<float>(const_acc)
                                        : std::bit_cast<float>(reads.acc->lane(out.reg, out.lane)),
                                    c_modifier);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k)
          Abuf[row * K + k] =
              ea(reads.a, s0, gfx12_wmma_a_input_loc(wave_size, M, K, row, k, a_bits, b_bits));
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col)
          Bbuf[k * stride + col] =
              eb(reads.b, s1, gfx12_wmma_b_input_loc(wave_size, N, K, col, k, a_bits, b_bits));
      wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[row * stride + col])});
        }
    }
  } else {
    run_scalar();
  }

  auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wave_size);
  for (const auto &r : results)
    writes.set_lane(r.reg, r.lane, r.val);
}

// GFX11 F16/BF16 WMMA executes adjacent K pairs as DOT2, starting at K=0.
// Packed outputs narrow after every pair. Preserve raw NaN bits and stage all
// outputs before writes, including D overlap with A, B or C.
template <bool Bf16, bool Packed = false>
void exec_gfx11_wmma_dot2(auto &cu, uint32_t wave_size, uint32_t dst, uint32_t s0, uint32_t s1,
                          uint32_t s2, std::optional<uint32_t> const_acc, uint32_t neg,
                          uint32_t neg_hi, uint32_t opsel = 0, bool fp16_ovfl = false) {
  require_gfx11_wmma_wave_size(wave_size);
  std::array<uint32_t, 256> results;
  RegisterAccess regs(cu);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = gfx11_wmma_output_loc_32(wave_size, 16, 16, row, col);
      const uint32_t lane_group = out.lane / 16;
      uint32_t acc = const_acc ? *const_acc : regs.read_vgpr(s2 + out.reg, out.lane);
      if constexpr (Packed)
        acc = (acc >> (16 * opsel)) & 0xffff;
      if (neg_hi & 4)
        acc &= Packed ? 0x7fffu : 0x7fffffffu;
      if (neg & 4)
        acc ^= Packed ? 0x8000u : 0x80000000u;
      for (uint32_t k = 0; k < 16; k += 2) {
        const auto al = gfx11_wmma_input_loc(16, 16, row, k, 16, lane_group);
        const auto bl = gfx11_wmma_input_loc(16, 16, col, k, 16, lane_group);
        uint32_t a = regs.read_vgpr(s0 + al.vgpr_offset, al.lane);
        uint32_t b = regs.read_vgpr(s1 + bl.vgpr_offset, bl.lane);
        a ^= ((neg & 1) << 15) | ((neg_hi & 1) << 31);
        b ^= ((neg & 2) << 14) | ((neg_hi & 2) << 30);
        if constexpr (Packed)
          acc = gfx11_dot2_packed16<Bf16>(uint16_t(a), uint16_t(b), uint16_t(a >> 16),
                                          uint16_t(b >> 16), uint16_t(acc), fp16_ovfl);
        else
          acc = gfx11_dot2_f32<Bf16>(uint16_t(a), uint16_t(b), uint16_t(a >> 16), uint16_t(b >> 16),
                                     acc);
      }
      results[row * 16 + col] = acc;
    }
  }
  for (uint32_t row = 0; row < 16; ++row)
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = gfx11_wmma_output_loc_32(wave_size, 16, 16, row, col);
      uint32_t value = results[row * 16 + col];
      if constexpr (Packed) {
        const unsigned shift = 16 * opsel;
        value = (regs.read_vgpr(dst + out.reg, out.lane) & ~(0xffffu << shift)) | (value << shift);
      }
      regs.write_vgpr(dst + out.reg, out.lane, value);
    }
}

// GFX12 F16/BF16 WMMA rounds after each group of four products; packed outputs
// narrow at each group as well.
// In wave32, documented K bits 2 and 3 occupy the lane and register selectors,
// respectively; gfx12_wmma_input_loc uses contiguous per-lane K chunks.
template <bool Bf16, bool Packed = false>
void exec_gfx12_wmma_dot4(auto &cu, uint32_t wave_size, uint32_t dst, uint32_t s0, uint32_t s1,
                          uint32_t s2, std::optional<uint32_t> const_acc, uint32_t neg,
                          uint32_t neg_hi, bool fp16_ovfl = false) {
  require_gfx12_wmma_wave_size(wave_size);
  std::array<uint32_t, 256> results;
  RegisterAccess regs(cu);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = gfx12_wmma_output_loc_32(wave_size, 16, 16, row, col);
      uint32_t acc =
          const_acc ? *const_acc : regs.read_vgpr(s2 + (Packed ? out.reg / 2 : out.reg), out.lane);
      if constexpr (Packed)
        acc = (acc >> (16 * (out.reg % 2))) & 0xffff;
      if (neg_hi & 4)
        acc &= Packed ? 0x7fffu : 0x7fffffffu;
      if (neg & 4)
        acc ^= Packed ? 0x8000u : 0x80000000u;
      for (uint32_t k = 0; k < 16; k += 4) {
        std::array<uint16_t, 4> a, b;
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t logical_k = k + j;
          const uint32_t physical_k =
              wave_size == 32 ? (logical_k & 3) | ((logical_k & 4) << 1) | ((logical_k & 8) >> 1)
                              : logical_k;
          const auto al = gfx12_wmma_input_loc(wave_size, 16, 16, row, physical_k, 16);
          const auto bl = gfx12_wmma_input_loc(wave_size, 16, 16, col, physical_k, 16);
          a[j] = uint16_t(regs.read_vgpr(s0 + al.vgpr_offset, al.lane) >> (16 * al.sub_element));
          b[j] = uint16_t(regs.read_vgpr(s1 + bl.vgpr_offset, bl.lane) >> (16 * bl.sub_element));
          if ((al.sub_element ? neg_hi : neg) & 1)
            a[j] ^= 0x8000u;
          if ((bl.sub_element ? neg_hi : neg) & 2)
            b[j] ^= 0x8000u;
        }
        if constexpr (Packed)
          acc = gfx12_dot4_packed16<Bf16>(a, b, uint16_t(acc), fp16_ovfl);
        else
          acc = gfx12_dot_f32<Bf16>(a, b, acc);
      }
      results[row * 16 + col] = acc;
    }
  }
  for (uint32_t row = 0; row < 16; ++row)
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = gfx12_wmma_output_loc_32(wave_size, 16, 16, row, col);
      if constexpr (Packed) {
        if (row % 2 == 0)
          regs.write_vgpr(dst + out.reg / 2, out.lane,
                          results[row * 16 + col] | (results[(row + 1) * 16 + col] << 16));
      } else
        regs.write_vgpr(dst + out.reg, out.lane, results[row * 16 + col]);
    }
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_f32_mixed(auto &cu, uint32_t wave_size, uint32_t M, uint32_t N, uint32_t K,
                               uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0,
                               uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                               uint32_t const_acc = ACC_FROM_VGPR, uint32_t c_modifier = 0) {
  require_gfx11_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = gfx11_wmma_output_loc_32(wave_size, M, N, row, col);
      uint32_t lane_group = out.lane / N;
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
      acc = apply_wmma_c_modifier(acc, c_modifier);
      for (uint32_t k = 0; k < K; ++k) {
        auto al = gfx11_wmma_input_loc(M, K, row, k, a_bits, lane_group);
        auto bl = gfx11_wmma_input_loc(N, K, col, k, b_bits, lane_group);
        acc += ea(cu, s0, al) * eb(cu, s1, bl);
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }

  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_f32(auto &cu, uint32_t wave_size, uint32_t M, uint32_t N, uint32_t K,
                         uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                         ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                         uint32_t c_modifier = 0) {
  exec_gfx11_wmma_f32_mixed(cu, wave_size, M, N, K, in_bits, in_bits, dst, s0, s1, s2, ea, eb,
                            const_acc, c_modifier);
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
void exec_gfx11_wmma_packed16(auto &cu, uint32_t wave_size, uint32_t M, uint32_t N, uint32_t K,
                              uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                              uint32_t opsel, ExtractA ea, ExtractB eb, ReadAcc read_acc,
                              PackResult pack_result, uint32_t const_acc = ACC_FROM_VGPR) {
  require_gfx11_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = gfx11_wmma_output_loc_16(wave_size, M, N, row, col, opsel);
      uint32_t lane_group = out.lane / N;
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
      for (uint32_t k = 0; k < K; ++k) {
        auto al = gfx11_wmma_input_loc(M, K, row, k, in_bits, lane_group);
        auto bl = gfx11_wmma_input_loc(N, K, col, k, in_bits, lane_group);
        acc += ea(cu, s0, al) * eb(cu, s1, bl);
      }
      results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
    }
  }

  uint32_t max_reg = 0;
  for (const auto &r : results)
    max_reg = std::max(max_reg, r.reg);
  const uint32_t dst_regs = max_reg + 1;

  std::vector<uint32_t> words(dst_regs * wave_size, 0);
  std::vector<uint8_t> masks(dst_regs * wave_size, 0);
  for (const auto &r : results) {
    uint32_t idx = r.reg * wave_size + r.lane;
    uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      uint32_t idx = reg * wave_size + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = RegisterAccess(cu).read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      RegisterAccess(cu).write_vgpr(dst + reg, lane, word);
    }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_f16(auto &cu, uint32_t wave_size, uint32_t M, uint32_t N, uint32_t K,
                         uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                         uint32_t opsel, ExtractA ea, ExtractB eb,
                         uint32_t const_acc = ACC_FROM_VGPR) {
  exec_gfx11_wmma_packed16(
      cu, wave_size, M, N, K, in_bits, dst, s0, s1, s2, opsel, ea, eb,
      [](auto &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = RegisterAccess(cu).read_vgpr(reg, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_f16(val); }, const_acc);
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_bf16(auto &cu, uint32_t wave_size, uint32_t M, uint32_t N, uint32_t K,
                          uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                          uint32_t opsel, ExtractA ea, ExtractB eb,
                          uint32_t const_acc = ACC_FROM_VGPR) {
  exec_gfx11_wmma_packed16(
      cu, wave_size, M, N, K, in_bits, dst, s0, s1, s2, opsel, ea, eb,
      [](auto &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = RegisterAccess(cu).read_vgpr(reg, lane);
        return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_bf16(val); }, const_acc);
}

template <typename ExtractA, typename ExtractB, typename ScaleAWord, typename ScaleBWord>
void exec_wmma_f32_scaled_mixed(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t a_bits,
                                uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                                uint32_t s2, ExtractA ea, ExtractB eb, uint32_t const_acc,
                                ScaleAWord scale_a_word, ScaleBWord scale_b_word,
                                uint32_t matrix_a_scale, uint32_t matrix_b_scale,
                                uint32_t matrix_a_scale_fmt, uint32_t matrix_b_scale_fmt,
                                bool scale16 = false, uint32_t c_modifier = 0) {
  require_wmma_wave32(cu);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);
  const uint32_t num_scale_blocks = scale16 ? 8u : 4u;

  auto scale_for = [](uint64_t scale_word, uint32_t scale_byte, uint32_t scale_fmt) -> float {
    return decode_wmma_scale_byte(static_cast<uint8_t>((scale_word >> (scale_byte * 8)) & 0xffu),
                                  scale_fmt);
  };

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        float acc =
            (const_acc != ACC_FROM_VGPR)
                ? std::bit_cast<float>(const_acc)
                : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
        acc = apply_wmma_c_modifier(acc, c_modifier);
        const uint64_t a_scale_word =
            scale_a_word(wmma_a_scale_lane(M, K, row, matrix_a_scale, a_bits, b_bits));
        const uint64_t b_scale_word = scale_b_word(wmma_scale_lane(col, matrix_b_scale));
        for (uint32_t block = 0; block < num_scale_blocks; ++block) {
          float block_sum = 0.0f;
          for (uint32_t k = 0; k < K; ++k) {
            if (wmma_block_scale_byte(k, scale16) != block)
              continue;
            auto al = wmma_block_scaled_a_input_loc(M, K, row, k, a_bits);
            auto bl = wmma_block_scaled_b_input_loc(N, K, col, k, b_bits);
            block_sum = std::fma(ea(cu, s0, al), eb(cu, s1, bl), block_sum);
          }
          block_sum *= scale_for(a_scale_word, block, matrix_a_scale_fmt);
          block_sum *= scale_for(b_scale_word, block, matrix_b_scale_fmt);
          acc += block_sum;
        }
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
      }
    }
  };

  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      auto reads = read_mixed_matrix_fast_path_regions(cu, s0, s1, s2, static_cast<uint64_t>(M) * K,
                                                       a_bits, static_cast<uint64_t>(N) * K, b_bits,
                                                       static_cast<uint64_t>(M) * N,
                                                       /*acc_bits=*/32, const_acc, /*wf_size=*/32);
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cacc[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = wmma_output_loc_32(M, N, row, col);
          Cacc[row * N + col] =
              apply_wmma_c_modifier((const_acc != ACC_FROM_VGPR)
                                        ? std::bit_cast<float>(const_acc)
                                        : std::bit_cast<float>(reads.acc->lane(out.reg, out.lane)),
                                    c_modifier);
        }
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t k = 0; k < K; ++k) {
          auto al = wmma_block_scaled_a_input_loc(M, K, row, k, a_bits);
          Abuf[row * K + k] = ea(reads.a, s0, al);
        }
      }
      for (uint32_t col = 0; col < N; ++col) {
        for (uint32_t k = 0; k < K; ++k) {
          auto bl = wmma_block_scaled_b_input_loc(N, K, col, k, b_bits);
          Bbuf[k * stride + col] = eb(reads.b, s1, bl);
        }
      }
      for (uint32_t row = 0; row < M; ++row) {
        const uint64_t a_scale_word =
            scale_a_word(wmma_a_scale_lane(M, K, row, matrix_a_scale, a_bits, b_bits));
        for (uint32_t block = 0; block < num_scale_blocks; ++block) {
          uint32_t col = 0;
          alignas(64) float block_sums[64];
          for (; col + W <= N; col += W) {
            util::native<float> block_sum(0.0f);
            for (uint32_t k = 0; k < K; ++k) {
              if (wmma_block_scale_byte(k, scale16) != block)
                continue;
              util::native<float> a(Abuf[row * K + k]);
              util::native<float> bv;
              bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
              block_sum = util::stdx::fma(a, bv, block_sum);
            }
            block_sum.copy_to(block_sums, util::stdx::vector_aligned);
            for (uint32_t j = 0; j < W; ++j) {
              float scaled = block_sums[j] * scale_for(a_scale_word, block, matrix_a_scale_fmt);
              const uint64_t b_scale_word = scale_b_word(wmma_scale_lane(col + j, matrix_b_scale));
              scaled *= scale_for(b_scale_word, block, matrix_b_scale_fmt);
              Cacc[row * N + col + j] += scaled;
            }
          }
          for (; col < N; ++col) {
            float block_sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
              if (wmma_block_scale_byte(k, scale16) == block)
                block_sum = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], block_sum);
            }
            block_sum *= scale_for(a_scale_word, block, matrix_a_scale_fmt);
            const uint64_t b_scale_word = scale_b_word(wmma_scale_lane(col, matrix_b_scale));
            block_sum *= scale_for(b_scale_word, block, matrix_b_scale_fmt);
            Cacc[row * N + col] += block_sum;
          }
        }
      }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = wmma_output_loc_32(M, N, row, col);
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cacc[row * N + col])});
        }
    }
  } else {
    run_scalar();
  }

  auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, /*wf_size=*/32);
  for (const auto &r : results)
    writes.set_lane(r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_f32(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst,
                   uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                   uint32_t const_acc = ACC_FROM_VGPR, uint32_t c_modifier = 0,
                   uint32_t wave_size = WMMA_WAVE32) {
  exec_wmma_f32_mixed(cu, M, N, K, in_bits, in_bits, dst, s0, s1, s2, ea, eb, const_acc, c_modifier,
                      wave_size);
}

/// Fast path for v_wmma_f32_16x16x32_f16 (gfx1250, wave32) — the WMMA analogue
/// of exec_f32_mfma_16x16x32_f16. Compile-time M/N/K let the compiler fully
/// unroll the 16-row x 32-K matmul into straight-line AVX-512 FMAs; the f16
/// inputs are bulk-converted once with F16C (one vector op per 16 halves)
/// instead of branchy per-element extract_f16; VGPRs are accessed through
/// observed register-access regions and the result scatters directly (no
/// Result staging vector). Falls back to the generic exec_wmma_f32 without
/// AVX-512 / under force-scalar.
inline void exec_wmma_f32_16x16x32_f16(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                       uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                                       uint32_t c_modifier = 0) {
  constexpr uint32_t M = 16, N = 16, K = 32, in_bits = 16;
  if constexpr (!util::has_stdx_simd) {
    exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
                  const_acc, c_modifier);
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
                    const_acc, c_modifier);
      return;
    }
    require_wmma_wave32(cu);
    const uint32_t wf = cu.wf_size();
    auto reads = read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32,
                                             const_acc, wf);
    auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
    alignas(64) float A_buf[M * K]; // A[row][k]
    alignas(64) float B_buf[K * N]; // B[k][col]
    alignas(64) float C_buf[M * N]; // C[row][col]
    alignas(64) uint32_t C_words[M * N];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    // A and B each occupy 8 VGPRs x wf lanes = 2*8*wf packed f16 (16 f16/lane
    // for K=32). Bulk-convert the region to f32 once, then the hoist is a pure
    // f32 index-shuffle (f16 j of word w sub s -> flat (w*wf+lane)*2+s).
    constexpr uint32_t NUM_IN_REGS = 8;
    const uint32_t n_halves = 2 * NUM_IN_REGS * wf;
    alignas(64) float A_f32[2 * NUM_IN_REGS * 64];
    alignas(64) float B_f32[2 * NUM_IN_REGS * 64];
    convert_f16_matrix_region(reads.a, A_f32, n_halves);
    convert_f16_matrix_region(reads.b, B_f32, n_halves);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        C_buf[row * N + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(C_words[out.reg * wf + out.lane]),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    // Dense 16x32 * 32x16 -> 16x16 matmul, 16-lane stdx FMA per row.
    for (uint32_t row = 0; row < M; ++row) {
      util::native<float> c_row;
      c_row.copy_from(&C_buf[row * N], util::stdx::vector_aligned);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<float> a_bcast(A_buf[row * K + k]);
        util::native<float> b_row;
        b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
        c_row = util::stdx::fma(a_bcast, b_row, c_row);
      }
      c_row.copy_to(&C_buf[row * N], util::stdx::vector_aligned);
    }
    // Scatter directly back to VGPRs (no Result staging vector).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        writes.set_linear_word(out.reg * wf + out.lane,
                               std::bit_cast<uint32_t>(C_buf[row * N + col]));
      }
  }
}

/// Fast path for v_wmma_f32_16x16x32_bf16 (gfx1250, wave32). Identical to
/// exec_wmma_f32_16x16x32_f16 except the bulk input convert is the bf16
/// zero-extend (no F16C needed). Falls back to the generic exec_wmma_f32
/// without AVX-512 / under force-scalar.
inline void exec_wmma_f32_16x16x32_bf16(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                        uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                                        uint32_t c_modifier = 0) {
  constexpr uint32_t M = 16, N = 16, K = 32, in_bits = 16;
  if constexpr (!util::has_stdx_simd) {
    exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16, amdgpu::extract_bf16,
                  const_acc, c_modifier);
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16,
                    amdgpu::extract_bf16, const_acc, c_modifier);
      return;
    }
    require_wmma_wave32(cu);
    const uint32_t wf = cu.wf_size();
    auto reads = read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32,
                                             const_acc, wf);
    auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
    alignas(64) float A_buf[M * K]; // A[row][k]
    alignas(64) float B_buf[K * N]; // B[k][col]
    alignas(64) float C_buf[M * N]; // C[row][col]
    alignas(64) uint32_t C_words[M * N];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    // A and B each occupy 8 VGPRs x wf lanes = 2*8*wf packed bf16 (16 bf16/lane
    // for K=32). Bulk-convert the region to f32 once, then the hoist is a pure
    // f32 index-shuffle (bf16 j of word w sub s -> flat (w*wf+lane)*2+s).
    constexpr uint32_t NUM_IN_REGS = 8;
    const uint32_t n_halves = 2 * NUM_IN_REGS * wf;
    alignas(64) float A_f32[2 * NUM_IN_REGS * 64];
    alignas(64) float B_f32[2 * NUM_IN_REGS * 64];
    convert_bf16_matrix_region(reads.a, A_f32, n_halves);
    convert_bf16_matrix_region(reads.b, B_f32, n_halves);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        C_buf[row * N + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(C_words[out.reg * wf + out.lane]),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    // Dense 16x32 * 32x16 -> 16x16 matmul, 16-lane stdx FMA per row.
    for (uint32_t row = 0; row < M; ++row) {
      util::native<float> c_row;
      c_row.copy_from(&C_buf[row * N], util::stdx::vector_aligned);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<float> a_bcast(A_buf[row * K + k]);
        util::native<float> b_row;
        b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
        c_row = util::stdx::fma(a_bcast, b_row, c_row);
      }
      c_row.copy_to(&C_buf[row * N], util::stdx::vector_aligned);
    }
    // Scatter directly back to VGPRs (no Result staging vector).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        writes.set_linear_word(out.reg * wf + out.lane,
                               std::bit_cast<uint32_t>(C_buf[row * N + col]));
      }
  }
}

// CDNA3 uses the AMD FNUZ format for fp8/bf8
/// Fast path for the dense f32-out fp8/bf8-input WMMA shapes
/// (v_wmma_f32_16x16x{64,128}_{fp8,bf8}_{fp8,bf8}, gfx1250, wave32). Identical
/// to the f16/bf16 specializations except the bulk input convert is the f8 LUT
/// gather; A/B formats are compile-time selected.
/// Falls back to the generic exec_wmma_f32_mixed without AVX-512 / under
/// force-scalar.
template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
void exec_wmma_f32_f8_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                           uint32_t const_acc = ACC_FROM_VGPR, uint32_t c_modifier = 0) {
  constexpr uint32_t in_bits = 8;
  static_assert(N % 16 == 0, "specialized f8 WMMA assumes N is a multiple of the zmm width");
  constexpr auto ea = f8_extract_fn<A_FP8, FNUZ>();
  constexpr auto eb = f8_extract_fn<B_FP8, FNUZ>();
  auto fallback = [&]() {
    exec_wmma_f32_mixed(cu, M, N, K, in_bits, in_bits, dst, s0, s1, s2, ea, eb, const_acc,
                        c_modifier);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    auto reads = read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32,
                                             const_acc, wf);
    auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
    alignas(64) float A_buf[M * K]; // A[row][k]
    alignas(64) float B_buf[K * N]; // B[k][col]
    alignas(64) float C_buf[M * N]; // C[row][col]
    alignas(64) uint32_t C_words[M * N];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    // Bulk-convert each whole packed f8 region to f32 once, then the hoist is a
    // pure f32 index-shuffle (byte of word w lane l sub s -> (w*wf+l)*4+s).
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    convert_f8_matrix_region<A_FP8, FNUZ>(reads.a, A_f32, M * K);
    convert_f8_matrix_region<B_FP8, FNUZ>(reads.b, B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        C_buf[row * N + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(C_words[out.reg * wf + out.lane]),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_a_input_loc(M, K, row, k, in_bits, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_b_input_loc(N, K, col, k, in_bits, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
      }
    // Dense MxKxN matmul, W-lane (zmm) stdx FMA over N (N/W chunks per row).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Scatter directly back to VGPRs (no Result staging vector).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        writes.set_linear_word(out.reg * wf + out.lane,
                               std::bit_cast<uint32_t>(C_buf[row * N + col]));
      }
  }
}

/// Return whether an f32-accumulating matrix output row divides into complete
/// native SIMD chunks.
constexpr bool mma_f32_native_width_supported(uint32_t n, uint32_t width) {
  return width > 1 && n % width == 0;
}

/// Fast path for the f32-input WMMA shapes (v_wmma_f32_*_f32). f32 inputs, so no
/// F16C convert — the hoist reads each operand word straight through observed
/// register-access regions. Compile-time M/N/K fully unroll the matmul, looped
/// over N in native-width chunks. The specialization is bypassed when host SIMD
/// is unavailable, the native width is one or does not evenly divide N, or
/// force-scalar is enabled. The generic exec_wmma_f32 path may still use
/// native-width SIMD with a scalar tail when SIMD is available but N is not
/// divisible by the native width.
template <uint32_t M, uint32_t N, uint32_t K>
void exec_wmma_f32_f32_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                            uint32_t const_acc = ACC_FROM_VGPR, uint32_t c_modifier = 0) {
  constexpr uint32_t in_bits = 32;
  if constexpr (!util::has_stdx_simd) {
    exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f32, amdgpu::extract_f32,
                  const_acc, c_modifier);
    return;
  } else {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    if (util::force_scalar() || !mma_f32_native_width_supported(N, W)) {
      exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f32, amdgpu::extract_f32,
                    const_acc, c_modifier);
      return;
    }
    require_wmma_wave32(cu);
    const uint32_t wf = cu.wf_size();
    auto reads = read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32,
                                             const_acc, wf);
    auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    alignas(64) uint32_t A_words[M * K];
    alignas(64) uint32_t B_words[N * K];
    alignas(64) uint32_t C_words[M * N];
    copy_matrix_region_words(reads.a, A_words);
    copy_matrix_region_words(reads.b, B_words);
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        C_buf[row * N + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(C_words[out.reg * wf + out.lane]),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = std::bit_cast<float>(A_words[al.vgpr_offset * wf + al.lane]);
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = std::bit_cast<float>(B_words[bl.vgpr_offset * wf + bl.lane]);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        writes.set_linear_word(out.reg * wf + out.lane,
                               std::bit_cast<uint32_t>(C_buf[row * N + col]));
      }
  }
}

/// AVX-512 fast path shared by the five gfx1250 16x16x64 16-bit SWMMAC forms.
///
/// The generic SWMMAC helpers repeatedly gather A, B, and sparse metadata via
/// RegisterAccess for every output element.  This specialization snapshots the
/// complete architectural inputs once, bulk-widens the packed 16-bit inputs,
/// resolves the sparse K selection once per A element, and accumulates all 16
/// columns of an output row in one native SIMD vector.
///
/// BF16F32 has an asymmetric D/C operand: C occupies eight F32 VGPRs while D
/// occupies four packed-BF16 VGPRs.  Keep accumulator and result formats
/// separate so that all eight C registers are snapshotted before only the low
/// four destination registers are written.
RJ_NOINLINE inline void exec_swmmac_16x16x64_16bit_fast_body(
    auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base, uint32_t index_base,
    SwmmacK64Input input_format, SwmmacK64Accumulator accumulator_format,
    SwmmacK64Result result_format, uint32_t const_acc, bool fp16_ovfl) {
  constexpr uint32_t SPEC_M = 16;
  constexpr uint32_t SPEC_N = 16;
  constexpr uint32_t SPEC_K = 64;
  constexpr uint32_t COMPRESSED_K = SPEC_K / 2;
  constexpr uint32_t WF = WMMA_WAVE32;
  constexpr uint32_t IN_BITS = 16;
  constexpr uint32_t INDEX_ENTRIES = 16;

  const uint32_t acc_bits =
      accumulator_format == SwmmacK64Accumulator::F32 ? uint32_t{32} : uint32_t{16};
  auto reads = read_mixed_matrix_fast_path_regions(
      cu, s0, s1, acc_base, static_cast<uint64_t>(SPEC_M) * COMPRESSED_K, IN_BITS,
      static_cast<uint64_t>(SPEC_N) * SPEC_K, IN_BITS, static_cast<uint64_t>(SPEC_M) * SPEC_N,
      acc_bits, const_acc, WF);
  RegisterAccess regs(cu);
  auto index_read = regs.read_vgpr_region(index_base, /*reg_count=*/1, mfma_full_lane_mask(WF));

  // Snapshot every source before acquiring the destination write view.  The
  // instruction permits D to alias A, B, C, or the sparse index register.
  alignas(64) float a_physical[SPEC_M * COMPRESSED_K];
  alignas(64) float b_physical[SPEC_N * SPEC_K];
  alignas(64) uint32_t c_words[SPEC_M * SPEC_N] = {};
  alignas(64) uint32_t index_words[WF] = {};
  if (input_format == SwmmacK64Input::F16) {
    convert_f16_matrix_region(reads.a, a_physical, SPEC_M * COMPRESSED_K);
    convert_f16_matrix_region(reads.b, b_physical, SPEC_N * SPEC_K);
  } else {
    convert_bf16_matrix_region(reads.a, a_physical, SPEC_M * COMPRESSED_K);
    convert_bf16_matrix_region(reads.b, b_physical, SPEC_N * SPEC_K);
  }
  if (reads.acc)
    copy_matrix_region_words(*reads.acc, c_words);
  index_read.copy_to({index_words, WF});

  alignas(64) float a_matrix[SPEC_M * COMPRESSED_K];
  alignas(64) float b_matrix[SPEC_K * SPEC_N];
  alignas(64) float c_matrix[SPEC_M * SPEC_N];
  alignas(64) uint32_t dense_k[SPEC_M * COMPRESSED_K];

  for (uint32_t row = 0; row < SPEC_M; ++row) {
    for (uint32_t ck = 0; ck < COMPRESSED_K; ++ck) {
      const auto a_loc = swmmac_a_input_loc(WF, SPEC_M, SPEC_K, row, ck, IN_BITS);
      a_matrix[row * COMPRESSED_K + ck] =
          a_physical[(a_loc.vgpr_offset * WF + a_loc.lane) * 2 + a_loc.sub_element];
      const auto index_loc = swmmac_index_loc(WF, SPEC_M, SPEC_K, IN_BITS, row, ck, INDEX_ENTRIES);
      dense_k[row * COMPRESSED_K + ck] =
          swmmac_dense_k(index_words[index_loc.lane], ck, index_loc.local_compressed_k);
    }
  }
  for (uint32_t k = 0; k < SPEC_K; ++k) {
    for (uint32_t col = 0; col < SPEC_N; ++col) {
      const auto b_loc = swmmac_b_input_loc(WF, SPEC_N, SPEC_K, col, k, IN_BITS);
      b_matrix[k * SPEC_N + col] =
          b_physical[(b_loc.vgpr_offset * WF + b_loc.lane) * 2 + b_loc.sub_element];
    }
  }

  auto initial_acc = [&](uint32_t row, uint32_t col) {
    if (const_acc != ACC_FROM_VGPR) {
      return std::bit_cast<float>(const_acc);
    }
    if (accumulator_format == SwmmacK64Accumulator::F32) {
      const auto out = wmma_output_loc_32(SPEC_M, SPEC_N, row, col);
      return std::bit_cast<float>(c_words[out.reg * WF + out.lane]);
    }
    const auto out = wmma_output_loc_16(SPEC_M, SPEC_N, row, col);
    const uint32_t raw = c_words[out.reg * WF + out.lane];
    const uint16_t packed = static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFFu);
    return accumulator_format == SwmmacK64Accumulator::F16 ? util::f16_to_f32(packed)
                                                           : util::bf16_to_f32(packed);
  };
  for (uint32_t row = 0; row < SPEC_M; ++row)
    for (uint32_t col = 0; col < SPEC_N; ++col)
      c_matrix[row * SPEC_N + col] = initial_acc(row, col);

  // Preserve compressed-K accumulation order while processing all 16 output
  // columns in parallel.  On an AVX-512 build native<float> is one ZMM.
  for (uint32_t row = 0; row < SPEC_M; ++row) {
    util::native<float> c_row;
    c_row.copy_from(&c_matrix[row * SPEC_N], util::stdx::vector_aligned);
    for (uint32_t ck = 0; ck < COMPRESSED_K; ++ck) {
      util::native<float> a_broadcast(a_matrix[row * COMPRESSED_K + ck]);
      util::native<float> b_row;
      b_row.copy_from(&b_matrix[dense_k[row * COMPRESSED_K + ck] * SPEC_N],
                      util::stdx::vector_aligned);
      c_row = util::native_fma(a_broadcast, b_row, c_row);
    }
    const uint64_t nan_lanes = util::simd_mask_to_bits(util::stdx::isnan(c_row));
    c_row.copy_to(&c_matrix[row * SPEC_N], util::stdx::vector_aligned);
    // Packed and scalar host FMAs can choose different NaN payload sources.
    // Recompute only exceptional lanes with the deterministic scalar helper.
    uint64_t pending_nan_lanes = nan_lanes;
    while (pending_nan_lanes != 0) {
      const uint32_t col = static_cast<uint32_t>(std::countr_zero(pending_nan_lanes));
      pending_nan_lanes &= pending_nan_lanes - 1;
      float acc = initial_acc(row, col);
      for (uint32_t ck = 0; ck < COMPRESSED_K; ++ck) {
        const uint32_t k = dense_k[row * COMPRESSED_K + ck];
        acc = matrix_fma(a_matrix[row * COMPRESSED_K + ck], b_matrix[k * SPEC_N + col], acc);
      }
      c_matrix[row * SPEC_N + col] = acc;
    }
  }

  const uint32_t result_bits = result_format == SwmmacK64Result::F32 ? uint32_t{32} : uint32_t{16};
  auto writes = write_wmma_output_region(cu, dst, SPEC_M, SPEC_N, result_bits, WF);
  if (result_format == SwmmacK64Result::F32) {
    for (uint32_t row = 0; row < SPEC_M; ++row)
      for (uint32_t col = 0; col < SPEC_N; ++col) {
        const auto out = wmma_output_loc_32(SPEC_M, SPEC_N, row, col);
        writes.set_linear_word(out.reg * WF + out.lane,
                               std::bit_cast<uint32_t>(c_matrix[row * SPEC_N + col]));
      }
  } else {
    constexpr uint32_t PACKED_REGS = (SPEC_M * SPEC_N / WF) / 2;
    alignas(64) uint32_t packed_words[PACKED_REGS * WF] = {};
    for (uint32_t row = 0; row < SPEC_M; ++row)
      for (uint32_t col = 0; col < SPEC_N; ++col) {
        const auto out = wmma_output_loc_16(SPEC_M, SPEC_N, row, col);
        // CDNA5 SWMMAC packed results use fixed round-to-nearest-even.
        const uint16_t value = result_format == SwmmacK64Result::F16
                                   ? wmma_round_f16(c_matrix[row * SPEC_N + col], fp16_ovfl)
                                   : wmma_round_bf16_rne(c_matrix[row * SPEC_N + col], fp16_ovfl);
        packed_words[out.reg * WF + out.lane] |= static_cast<uint32_t>(value)
                                                 << (out.sub_element * 16);
      }
    for (uint32_t reg = 0; reg < PACKED_REGS; ++reg)
      for (uint32_t lane = 0; lane < WF; ++lane)
        writes.set_linear_word(reg * WF + lane, packed_words[reg * WF + lane]);
  }
}

/// Keep all eligibility checks outside the stack-heavy implementation so
/// forced-scalar and non-K64 instructions do not allocate or probe its scratch
/// frame before falling back to the generic path.
[[gnu::always_inline]] inline bool
try_exec_swmmac_16x16x64_16bit(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits,
                               uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                               uint32_t index_base, uint32_t index_entries, uint32_t index_key,
                               SwmmacK64Input input_format, SwmmacK64Accumulator accumulator_format,
                               SwmmacK64Result result_format, uint32_t const_acc = ACC_FROM_VGPR,
                               uint32_t wave_size = WMMA_WAVE32, bool fp16_ovfl = false) {
  if constexpr (util::has_stdx_simd) {
    if (util::force_scalar() || util::native<float>::size() != 16 || M != 16 || N != 16 ||
        K != 64 || in_bits != 16 || index_entries != 16 || index_key != 0 ||
        wave_size != WMMA_WAVE32 || cu.wf_size() != WMMA_WAVE32)
      return false;

    exec_swmmac_16x16x64_16bit_fast_body(cu, dst, s0, s1, acc_base, index_base, input_format,
                                         accumulator_format, result_format, const_acc, fp16_ovfl);
    return true;
  } else {
    return false;
  }
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f32_mixed(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t a_bits,
                           uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                           uint32_t acc_base, uint32_t index_base, uint32_t index_entries,
                           uint32_t index_key, ExtractA ea, ExtractB eb,
                           uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  const uint32_t compressed_k = K / 2;

  // dense_k depends on (row, ck) via the row's metadata, not on col — so the
  // gathered B column is shared across all col for a fixed row. Resolve it once.
  auto dense_k_for = [&](uint32_t row, uint32_t ck) -> uint32_t {
    const auto index_loc = swmmac_index_loc(wave_size, M, K, a_bits, row, ck, index_entries);
    const uint64_t index_set =
        read_swmmac_index_set(cu, index_base, index_loc.lane, index_entries, index_key);
    return swmmac_dense_k(index_set, ck, index_loc.local_compressed_k);
  };

  auto initial_acc_for = [&](uint32_t row, uint32_t col) {
    if (const_acc != ACC_FROM_VGPR)
      return std::bit_cast<float>(const_acc);
    const auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
    return std::bit_cast<float>(RegisterAccess(cu).read_vgpr(acc_base + out.reg, out.lane));
  };

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        float acc = initial_acc_for(row, col);
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          auto al = swmmac_a_input_loc(wave_size, M, K, row, ck, a_bits);
          auto bl = swmmac_b_input_loc(wave_size, N, K, col, dense_k_for(row, ck), b_bits);
          acc = swmmac_scalar_mac<ExtractA, ExtractB>(ea(cu, s0, al), eb(cu, s1, bl), acc);
        }
        if (std::isnan(acc)) [[unlikely]] {
          acc = swmmac_replay_nan_vgpr(cu, M, N, K, a_bits, b_bits, s0, s1, index_base,
                                       index_entries, index_key, row, col, ea, eb,
                                       initial_acc_for(row, col), wave_size);
        }
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
      }
    }
  };

  // SIMD fast path: because the B gather is row-dependent, hoist each row's A
  // (compressed_k) and B (compressed_k x col, with dense_k folded in) into dense
  // buffers, then run that single row through the shared matmul core (M=1).
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(compressed_k) > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(compressed_k) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col)
          Cbuf[col] = initial_acc_for(row, col);
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          Abuf[ck] = ea(cu, s0, swmmac_a_input_loc(wave_size, M, K, row, ck, a_bits));
          const uint32_t dense_k = dense_k_for(row, ck);
          for (uint32_t col = 0; col < N; ++col)
            Bbuf[ck * stride + col] =
                eb(cu, s1, swmmac_b_input_loc(wave_size, N, K, col, dense_k, b_bits));
        }
        wmma_simd_matmul<float>(1, N, compressed_k, W, stride, Abuf, Bbuf, Cbuf);
        for (uint32_t col = 0; col < N; ++col) {
          if (std::isnan(Cbuf[col])) [[unlikely]] {
            Cbuf[col] = swmmac_replay_nan_buffers(Abuf, Bbuf, compressed_k, stride, col,
                                                  initial_acc_for(row, col));
          }
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[col])});
        }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f32(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst,
                     uint32_t s0, uint32_t s1, uint32_t acc_base, uint32_t index_base,
                     uint32_t index_entries, uint32_t index_key, ExtractA ea, ExtractB eb,
                     uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32) {
  using A = std::remove_cvref_t<ExtractA>;
  using B = std::remove_cvref_t<ExtractB>;
  if constexpr (std::is_same_v<A, ExtractF16> && std::is_same_v<B, ExtractF16>) {
    if (try_exec_swmmac_16x16x64_16bit(cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base,
                                       index_entries, index_key, SwmmacK64Input::F16,
                                       SwmmacK64Accumulator::F32, SwmmacK64Result::F32, const_acc,
                                       wave_size))
      return;
  } else if constexpr (std::is_same_v<A, ExtractBf16> && std::is_same_v<B, ExtractBf16>) {
    if (try_exec_swmmac_16x16x64_16bit(cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base,
                                       index_entries, index_key, SwmmacK64Input::BF16,
                                       SwmmacK64Accumulator::F32, SwmmacK64Result::F32, const_acc,
                                       wave_size))
      return;
  }
  if (try_exec_swmmac_16x16x128_8bit(cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base,
                                     index_entries, index_key, ea, eb, SwmmacK128Result::F32,
                                     const_acc, wave_size))
    return;
  exec_swmmac_f32_mixed(cu, M, N, K, in_bits, in_bits, dst, s0, s1, acc_base, index_base,
                        index_entries, index_key, ea, eb, const_acc, wave_size);
}

/// Execute sparse BF16F32 with its architectural mixed-width D/C contract:
/// eight F32 accumulator registers are read and four packed-BF16 result
/// registers are written. The encoding names the eight-register C view; the
/// generated operand metadata exposes that tied read separately from the
/// four-register D definition. Runtime execution must leave dst+4 through
/// dst+7 untouched.
template <typename ExtractA, typename ExtractB>
void exec_swmmac_bf16f32(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits,
                         uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                         uint32_t index_base, uint32_t index_entries, uint32_t index_key,
                         ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                         uint32_t wave_size = WMMA_WAVE32, bool fp16_ovfl = false) {
  using A = std::remove_cvref_t<ExtractA>;
  using B = std::remove_cvref_t<ExtractB>;
  if constexpr (std::is_same_v<A, ExtractBf16> && std::is_same_v<B, ExtractBf16>) {
    if (try_exec_swmmac_16x16x64_16bit(cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base,
                                       index_entries, index_key, SwmmacK64Input::BF16,
                                       SwmmacK64Accumulator::F32, SwmmacK64Result::BF16, const_acc,
                                       wave_size, fp16_ovfl))
      return;
  }

  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);
  const uint32_t compressed_k = K / 2;

  auto dense_k_for = [&](uint32_t row, uint32_t ck) -> uint32_t {
    const auto index_loc = swmmac_index_loc(wave_size, M, K, in_bits, row, ck, index_entries);
    const uint64_t index_set =
        read_swmmac_index_set(cu, index_base, index_loc.lane, index_entries, index_key);
    return swmmac_dense_k(index_set, ck, index_loc.local_compressed_k);
  };

  auto initial_acc_for = [&](uint32_t row, uint32_t col) {
    if (const_acc != ACC_FROM_VGPR)
      return std::bit_cast<float>(const_acc);
    const auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
    return std::bit_cast<float>(RegisterAccess(cu).read_vgpr(acc_base + out.reg, out.lane));
  };

  // Stage every result before writing so D may alias any input, including the
  // upper half of the F32 accumulator tuple. Accumulate a row's independent
  // columns together to hide the latency of the architecturally required fused
  // BF16 MAC chain while retaining each output's compressed-K order.
  if (N != 16)
    throw util::UnimplementedInst("BF16F32 SWMMAC requires N=16");
  std::array<float, 16> accumulators{};
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col)
      accumulators[col] = initial_acc_for(row, col);
    for (uint32_t ck = 0; ck < compressed_k; ++ck) {
      const auto a_loc = swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits);
      const float a = ea(cu, s0, a_loc);
      const uint32_t dense_k = dense_k_for(row, ck);
      for (uint32_t col = 0; col < N; ++col) {
        const auto b_loc = swmmac_b_input_loc(wave_size, N, K, col, dense_k, in_bits);
        accumulators[col] = std::fma(a, eb(cu, s1, b_loc), accumulators[col]);
      }
    }
    for (uint32_t col = 0; col < N; ++col) {
      float acc = accumulators[col];
      if (std::isnan(acc)) [[unlikely]] {
        acc = swmmac_replay_nan_vgpr(cu, M, N, K, in_bits, in_bits, s0, s1, index_base,
                                     index_entries, index_key, row, col, ea, eb,
                                     initial_acc_for(row, col), wave_size);
      }
      const auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
      results.push_back({out.reg, out.lane, out.sub_element, wmma_round_bf16_rne(acc, fp16_ovfl)});
    }
  }

  const uint32_t dst_regs = ((M * N) / wave_size + 1) / 2;
  std::vector<uint32_t> words(dst_regs * wave_size, 0);
  std::vector<uint8_t> masks(dst_regs * wave_size, 0);
  for (const auto &r : results) {
    const uint32_t idx = r.reg * wave_size + r.lane;
    const uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      const uint32_t idx = reg * wave_size + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        const uint32_t old = RegisterAccess(cu).read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      RegisterAccess(cu).write_vgpr(dst + reg, lane, word);
    }
  }
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
void exec_wmma_packed16(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits,
                        uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea,
                        ExtractB eb, ReadAcc read_acc, PackResult pack_result,
                        uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
        for (uint32_t k = 0; k < K; ++k) {
          auto al = gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits);
          auto bl = gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
      }
    }
  };

  // SIMD fast path: run the f32 matmul vectorized over N into a dense grid, then
  // pack each result to 16 bits via the caller's pack_result. The masked 2-per-
  // word scatter below is unchanged.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
          Cbuf[row * stride + col] = (const_acc != ACC_FROM_VGPR)
                                         ? std::bit_cast<float>(const_acc)
                                         : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k)
          Abuf[row * K + k] = ea(cu, s0, gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits));
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col)
          Bbuf[k * stride + col] =
              eb(cu, s1, gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits));
      wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
          results.push_back(
              {out.reg, out.lane, out.sub_element, pack_result(Cbuf[row * stride + col])});
        }
    }
  } else {
    run_scalar();
  }

  uint32_t dst_regs = ((M * N) / wave_size + 1) / 2;
  std::vector<uint32_t> words(dst_regs * wave_size, 0);
  std::vector<uint8_t> masks(dst_regs * wave_size, 0);
  for (const auto &r : results) {
    uint32_t idx = r.reg * wave_size + r.lane;
    uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      uint32_t idx = reg * wave_size + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = RegisterAccess(cu).read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      RegisterAccess(cu).write_vgpr(dst + reg, lane, word);
    }
  }
}

inline void exec_wmma_bf16f32_16x16x32_bf16(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                            uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                                            uint32_t c_modifier = 0) {
  constexpr uint32_t M = 16, N = 16, K = 32, in_bits = 16;
  auto run_scalar = [&]() {
    require_wmma_wave32(cu);
    struct Result {
      uint32_t reg;
      uint32_t lane;
      uint32_t sub_element;
      uint16_t val;
    };
    std::vector<Result> results;
    results.reserve(M * N);

    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto cout = wmma_output_loc_32(M, N, row, col);
        const float initial_acc =
            (const_acc != ACC_FROM_VGPR)
                ? std::bit_cast<float>(const_acc)
                : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + cout.reg, cout.lane));
        float acc = apply_wmma_c_modifier(initial_acc, c_modifier);
        for (uint32_t k = 0; k < K; ++k) {
          auto al = wmma_input_loc(M, K, row, k, in_bits);
          auto bl = wmma_input_loc(N, K, col, k, in_bits);
          acc = std::fma(extract_bf16(cu, s0, al), extract_bf16(cu, s1, bl), acc);
        }
        if (std::isnan(acc)) {
          acc = apply_wmma_c_modifier(initial_acc, c_modifier);
          for (uint32_t k = 0; k < K; ++k) {
            auto al = wmma_input_loc(M, K, row, k, in_bits);
            auto bl = wmma_input_loc(N, K, col, k, in_bits);
            acc = matrix_fma(extract_bf16(cu, s0, al), extract_bf16(cu, s1, bl), acc);
          }
        }
        auto out = wmma_output_loc_16(M, N, row, col);
        results.push_back({out.reg, out.lane, out.sub_element, util::f32_to_bf16(acc)});
      }
    }

    uint32_t dst_regs = ((M * N) / WMMA_WAVE32 + 1) / 2;
    std::vector<uint32_t> words(dst_regs * WMMA_WAVE32, 0);
    std::vector<uint8_t> masks(dst_regs * WMMA_WAVE32, 0);
    for (const auto &r : results) {
      uint32_t idx = r.reg * WMMA_WAVE32 + r.lane;
      uint32_t shift = r.sub_element * 16;
      words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
      masks[idx] |= 1u << r.sub_element;
    }
    for (uint32_t reg = 0; reg < dst_regs; ++reg) {
      for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
        uint32_t idx = reg * WMMA_WAVE32 + lane;
        uint32_t word = words[idx];
        if (masks[idx] != 0x3u) {
          uint32_t old = RegisterAccess(cu).read_vgpr(dst + reg, lane);
          if ((masks[idx] & 0x1u) == 0)
            word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
          if ((masks[idx] & 0x2u) == 0)
            word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
        }
        RegisterAccess(cu).write_vgpr(dst + reg, lane, word);
      }
    }
  };

  if constexpr (!util::has_stdx_simd) {
    run_scalar();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      run_scalar();
      return;
    }

    require_wmma_wave32(cu);
    const uint32_t wf = cu.wf_size();
    auto reads = read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32,
                                             const_acc, wf);
    auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/16, wf);

    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[K * N];
    alignas(64) uint32_t C_words[M * N];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    convert_bf16_matrix_region(reads.a, A_f32, M * K);
    convert_bf16_matrix_region(reads.b, B_f32, K * N);

    auto initial_acc = [&](uint32_t row, uint32_t col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      return apply_wmma_c_modifier((const_acc != ACC_FROM_VGPR)
                                       ? std::bit_cast<float>(const_acc)
                                       : std::bit_cast<float>(C_words[out.reg * wf + out.lane]),
                                   c_modifier);
    };
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col)
        C_buf[row * N + col] = initial_acc(row, col);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto in = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_f32[(in.vgpr_offset * wf + in.lane) * 2 + in.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto in = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_f32[(in.vgpr_offset * wf + in.lane) * 2 + in.sub_element];
      }

    for (uint32_t row = 0; row < M; ++row) {
      util::native<float> c_row;
      c_row.copy_from(&C_buf[row * N], util::stdx::vector_aligned);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<float> b_row;
        b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
        c_row = util::native_fma(util::native<float>(A_buf[row * K + k]), b_row, c_row);
      }
      const uint64_t nan_lanes = util::simd_mask_to_bits(util::stdx::isnan(c_row));
      c_row.copy_to(&C_buf[row * N], util::stdx::vector_aligned);
      // A packed host FMA may propagate a different NaN operand than scalar
      // FMA.  Recompute only exceptional lanes with the shared source-priority
      // helper; finite rows retain the full-width AVX-512 path.
      uint64_t pending_nan_lanes = nan_lanes;
      while (pending_nan_lanes != 0) {
        const uint32_t col = static_cast<uint32_t>(std::countr_zero(pending_nan_lanes));
        pending_nan_lanes &= pending_nan_lanes - 1;
        float acc = initial_acc(row, col);
        for (uint32_t k = 0; k < K; ++k)
          acc = matrix_fma(A_buf[row * K + k], B_buf[k * N + col], acc);
        C_buf[row * N + col] = acc;
      }
    }

    // wmma_output_loc_16 pairs adjacent rows in each destination register.
    // Pack two complete rows per vector: even rows become the low halves and
    // odd rows become the high halves.  This is the scalar BF16 truncation
    // contract (high 16 bits of each f32), not round-to-nearest-even.
    constexpr uint32_t DST_REGS = 4;
    alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32];
    for (uint32_t pair = 0; pair < M / 2; ++pair) {
      util::native<float> even_row;
      util::native<float> odd_row;
      even_row.copy_from(&C_buf[(2 * pair) * N], util::stdx::vector_aligned);
      odd_row.copy_from(&C_buf[(2 * pair + 1) * N], util::stdx::vector_aligned);
      auto packed = (std::bit_cast<util::native<uint32_t>>(even_row) >> 16) |
                    (std::bit_cast<util::native<uint32_t>>(odd_row) & 0xFFFF0000u);
      packed.copy_to(&words[(pair % DST_REGS) * WMMA_WAVE32 + (pair / DST_REGS) * N],
                     util::stdx::vector_aligned);
    }
    for (uint32_t reg = 0; reg < DST_REGS; ++reg)
      for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane)
        writes.set_linear_word(reg * wf + lane, words[reg * WMMA_WAVE32 + lane]);
  }
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
void exec_swmmac_packed16(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits,
                          uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                          uint32_t index_base, uint32_t index_entries, uint32_t index_key,
                          ExtractA ea, ExtractB eb, ReadAcc read_acc, PackResult pack_result,
                          uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  const uint32_t compressed_k = K / 2;

  auto dense_k_for = [&](uint32_t row, uint32_t ck) -> uint32_t {
    const auto index_loc = swmmac_index_loc(wave_size, M, K, in_bits, row, ck, index_entries);
    const uint64_t index_set =
        read_swmmac_index_set(cu, index_base, index_loc.lane, index_entries, index_key);
    return swmmac_dense_k(index_set, ck, index_loc.local_compressed_k);
  };

  auto initial_acc_for = [&](uint32_t row, uint32_t col) {
    if (const_acc != ACC_FROM_VGPR)
      return std::bit_cast<float>(const_acc);
    const auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
    return read_acc(cu, acc_base + out.reg, out.lane, out.sub_element);
  };

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
        float acc = initial_acc_for(row, col);
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          auto al = swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits);
          auto bl = swmmac_b_input_loc(wave_size, N, K, col, dense_k_for(row, ck), in_bits);
          acc = swmmac_scalar_mac<ExtractA, ExtractB>(ea(cu, s0, al), eb(cu, s1, bl), acc);
        }
        if (std::isnan(acc)) [[unlikely]] {
          acc = swmmac_replay_nan_vgpr(cu, M, N, K, in_bits, in_bits, s0, s1, index_base,
                                       index_entries, index_key, row, col, ea, eb,
                                       initial_acc_for(row, col), wave_size);
        }
        results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
      }
    }
  };

  // SIMD fast path: per-row gather (dense_k is row-dependent) into dense f32
  // buffers, single row through the matmul core (M=1), pack to 16 bits.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(compressed_k) > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(compressed_k) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col)
          Cbuf[col] = initial_acc_for(row, col);
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          Abuf[ck] = ea(cu, s0, swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits));
          const uint32_t dense_k = dense_k_for(row, ck);
          for (uint32_t col = 0; col < N; ++col)
            Bbuf[ck * stride + col] =
                eb(cu, s1, swmmac_b_input_loc(wave_size, N, K, col, dense_k, in_bits));
        }
        wmma_simd_matmul<float>(1, N, compressed_k, W, stride, Abuf, Bbuf, Cbuf);
        for (uint32_t col = 0; col < N; ++col) {
          if (std::isnan(Cbuf[col])) [[unlikely]] {
            Cbuf[col] = swmmac_replay_nan_buffers(Abuf, Bbuf, compressed_k, stride, col,
                                                  initial_acc_for(row, col));
          }
          auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
          results.push_back({out.reg, out.lane, out.sub_element, pack_result(Cbuf[col])});
        }
      }
    }
  } else {
    run_scalar();
  }

  uint32_t dst_regs = ((M * N) / wave_size + 1) / 2;
  std::vector<uint32_t> words(dst_regs * wave_size, 0);
  std::vector<uint8_t> masks(dst_regs * wave_size, 0);
  for (const auto &r : results) {
    uint32_t idx = r.reg * wave_size + r.lane;
    uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      uint32_t idx = reg * wave_size + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = RegisterAccess(cu).read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      RegisterAccess(cu).write_vgpr(dst + reg, lane, word);
    }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_f16(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst,
                   uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                   uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32,
                   bool fp16_ovfl = false) {
  exec_wmma_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, s2, ea, eb,
      [](auto &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = RegisterAccess(cu).read_vgpr(reg, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [fp16_ovfl](float val) { return wmma_round_f16(val, fp16_ovfl); }, const_acc, wave_size);
}

/// Fast path for the f16-output WMMA shapes (v_wmma_f16_*_f16). Like the f32-out
/// f16 specialization for the input convert + matmul, but the result is packed
/// to 16 bits (2 per dst word) and written through the WMMA 16-bit output map.
/// Falls back to generic exec_wmma_f16 without AVX-512 / under force-scalar.
template <uint32_t M, uint32_t N, uint32_t K>
void exec_wmma_f16_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                        uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t in_bits = 16;
  static_assert(N % 16 == 0, "specialized f16 WMMA assumes N is a multiple of the zmm width");
  auto fallback = [&]() {
    exec_wmma_f16(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
                  const_acc);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    auto reads = read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/16,
                                             const_acc, wf);
    auto writes = readwrite_wmma_output_region(cu, dst, M, N, /*output_bits=*/16, wf);
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    alignas(64) uint32_t C_words[M * N];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    convert_f16_matrix_region(reads.a, A_f32, M * K);
    convert_f16_matrix_region(reads.b, B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        if (const_acc != ACC_FROM_VGPR) {
          C_buf[row * N + col] = std::bit_cast<float>(const_acc);
        } else {
          uint32_t raw = C_words[out.reg * wf + out.lane];
          C_buf[row * N + col] =
              util::f16_to_f32(static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFF));
        }
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Pack f32 results to f16, two per dst word (the WMMA 16-bit output map).
    constexpr uint32_t DST_REGS = ((M * N) / WMMA_WAVE32 + 1) / 2;
    alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32] = {};
    alignas(64) uint8_t masks[DST_REGS * WMMA_WAVE32] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        uint32_t idx = out.reg * WMMA_WAVE32 + out.lane;
        uint32_t shift = out.sub_element * 16;
        uint16_t v = util::f32_to_f16(C_buf[row * N + col]);
        words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(v) << shift);
        masks[idx] |= 1u << out.sub_element;
      }
    for (uint32_t reg = 0; reg < DST_REGS; ++reg)
      for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
        uint32_t idx = reg * WMMA_WAVE32 + lane;
        uint32_t word = words[idx];
        if (masks[idx] != 0x3u) {
          uint32_t old = writes.linear_word(reg * wf + lane);
          if ((masks[idx] & 0x1u) == 0)
            word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
          if ((masks[idx] & 0x2u) == 0)
            word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
        }
        writes.set_linear_word(reg * wf + lane, word);
      }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f16(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst,
                     uint32_t s0, uint32_t s1, uint32_t acc_base, uint32_t index_base,
                     uint32_t index_entries, uint32_t index_key, ExtractA ea, ExtractB eb,
                     uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32,
                     bool fp16_ovfl = false) {
  using A = std::remove_cvref_t<ExtractA>;
  using B = std::remove_cvref_t<ExtractB>;
  if constexpr (std::is_same_v<A, ExtractF16> && std::is_same_v<B, ExtractF16>) {
    if (try_exec_swmmac_16x16x64_16bit(cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base,
                                       index_entries, index_key, SwmmacK64Input::F16,
                                       SwmmacK64Accumulator::F16, SwmmacK64Result::F16, const_acc,
                                       wave_size, fp16_ovfl))
      return;
  }
  if (try_exec_swmmac_16x16x128_8bit(cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base,
                                     index_entries, index_key, ea, eb, SwmmacK128Result::F16,
                                     const_acc, wave_size, fp16_ovfl))
    return;
  exec_swmmac_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base, index_entries, index_key, ea, eb,
      [](auto &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = RegisterAccess(cu).read_vgpr(reg, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [fp16_ovfl](float val) { return wmma_round_f16(val, fp16_ovfl); }, const_acc, wave_size);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_bf16(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst,
                    uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                    uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32) {
  exec_wmma_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, s2, ea, eb,
      [](auto &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = RegisterAccess(cu).read_vgpr(reg, lane);
        return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_bf16(val); }, const_acc, wave_size);
}

/// Fast path for the bf16-output WMMA shapes (v_wmma_bf16_*_bf16). Like the
/// f16-out specialization but with the bf16 bulk input convert (zero-extend)
/// and bf16 truncation on output. Falls back to generic exec_wmma_bf16 without
/// AVX-512 / under force-scalar.
template <uint32_t M, uint32_t N, uint32_t K>
void exec_wmma_bf16_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                         uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t in_bits = 16;
  static_assert(N % 16 == 0, "specialized bf16 WMMA assumes N is a multiple of the zmm width");
  auto fallback = [&]() {
    exec_wmma_bf16(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16,
                   amdgpu::extract_bf16, const_acc);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    auto reads = read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/16,
                                             const_acc, wf);
    auto writes = readwrite_wmma_output_region(cu, dst, M, N, /*output_bits=*/16, wf);
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    alignas(64) uint32_t C_words[M * N];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    convert_bf16_matrix_region(reads.a, A_f32, M * K);
    convert_bf16_matrix_region(reads.b, B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        if (const_acc != ACC_FROM_VGPR) {
          C_buf[row * N + col] = std::bit_cast<float>(const_acc);
        } else {
          uint32_t raw = C_words[out.reg * wf + out.lane];
          C_buf[row * N + col] =
              util::bf16_to_f32(static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFF));
        }
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Pack f32 results to bf16 (truncation), two per dst word.
    constexpr uint32_t DST_REGS = ((M * N) / WMMA_WAVE32 + 1) / 2;
    alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32] = {};
    alignas(64) uint8_t masks[DST_REGS * WMMA_WAVE32] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        uint32_t idx = out.reg * WMMA_WAVE32 + out.lane;
        uint32_t shift = out.sub_element * 16;
        uint16_t v = util::f32_to_bf16(C_buf[row * N + col]);
        words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(v) << shift);
        masks[idx] |= 1u << out.sub_element;
      }
    for (uint32_t reg = 0; reg < DST_REGS; ++reg)
      for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
        uint32_t idx = reg * WMMA_WAVE32 + lane;
        uint32_t word = words[idx];
        if (masks[idx] != 0x3u) {
          uint32_t old = writes.linear_word(reg * wf + lane);
          if ((masks[idx] & 0x1u) == 0)
            word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
          if ((masks[idx] & 0x2u) == 0)
            word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
        }
        writes.set_linear_word(reg * wf + lane, word);
      }
  }
}

/// Fast path for the dense f16-out fp8/bf8-input WMMA shapes
/// (v_wmma_f16_16x16x{64,128}_{fp8,bf8}_{fp8,bf8}). Like the f16-out f16
/// specialization for the matmul + packed 16-bit output map, but the bulk
/// input convert is the f8 LUT gather; A/B formats are compile-time selected.
/// Falls back to the generic exec_wmma_f16 without AVX-512 / under
/// force-scalar.
template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
void exec_wmma_f16_f8_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                           uint32_t const_acc = ACC_FROM_VGPR, bool fp16_ovfl = false) {
  constexpr uint32_t in_bits = 8;
  static_assert(N % 16 == 0, "specialized f8 WMMA assumes N is a multiple of the zmm width");
  constexpr auto ea = f8_extract_fn<A_FP8, FNUZ>();
  constexpr auto eb = f8_extract_fn<B_FP8, FNUZ>();
  auto fallback = [&]() {
    exec_wmma_f16(cu, M, N, K, in_bits, dst, s0, s1, s2, ea, eb, const_acc, WMMA_WAVE32, fp16_ovfl);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    auto reads = read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/16,
                                             const_acc, wf);
    auto writes = readwrite_wmma_output_region(cu, dst, M, N, /*output_bits=*/16, wf);
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    alignas(64) uint32_t C_words[M * N];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    convert_f8_matrix_region<A_FP8, FNUZ>(reads.a, A_f32, M * K);
    convert_f8_matrix_region<B_FP8, FNUZ>(reads.b, B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        if (const_acc != ACC_FROM_VGPR) {
          C_buf[row * N + col] = std::bit_cast<float>(const_acc);
        } else {
          uint32_t raw = C_words[out.reg * wf + out.lane];
          C_buf[row * N + col] =
              util::f16_to_f32(static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFF));
        }
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_a_input_loc(M, K, row, k, in_bits, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_b_input_loc(N, K, col, k, in_bits, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Pack f32 results to f16, two per dst word (the WMMA 16-bit output map).
    constexpr uint32_t DST_REGS = ((M * N) / WMMA_WAVE32 + 1) / 2;
    alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32] = {};
    alignas(64) uint8_t masks[DST_REGS * WMMA_WAVE32] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        uint32_t idx = out.reg * WMMA_WAVE32 + out.lane;
        uint32_t shift = out.sub_element * 16;
        uint16_t v = wmma_round_f16(C_buf[row * N + col], fp16_ovfl);
        words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(v) << shift);
        masks[idx] |= 1u << out.sub_element;
      }
    for (uint32_t reg = 0; reg < DST_REGS; ++reg)
      for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
        uint32_t idx = reg * WMMA_WAVE32 + lane;
        uint32_t word = words[idx];
        if (masks[idx] != 0x3u) {
          uint32_t old = writes.linear_word(reg * wf + lane);
          if ((masks[idx] & 0x1u) == 0)
            word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
          if ((masks[idx] & 0x2u) == 0)
            word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
        }
        writes.set_linear_word(reg * wf + lane, word);
      }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_bf16(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst,
                      uint32_t s0, uint32_t s1, uint32_t acc_base, uint32_t index_base,
                      uint32_t index_entries, uint32_t index_key, ExtractA ea, ExtractB eb,
                      uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32,
                      bool fp16_ovfl = false) {
  using A = std::remove_cvref_t<ExtractA>;
  using B = std::remove_cvref_t<ExtractB>;
  auto read_acc = [](auto &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
    uint32_t raw = RegisterAccess(cu).read_vgpr(reg, lane);
    return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
  };
  if constexpr (std::is_same_v<A, ExtractBf16> && std::is_same_v<B, ExtractBf16>) {
    if (try_exec_swmmac_16x16x64_16bit(cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base,
                                       index_entries, index_key, SwmmacK64Input::BF16,
                                       SwmmacK64Accumulator::BF16, SwmmacK64Result::BF16, const_acc,
                                       wave_size, fp16_ovfl))
      return;
    if (M == 16 && N == 16 && K == 64 && in_bits == 16 && index_entries == 16) {
      // Match fixed RNE when RJ_FORCE_SCALAR selects the oracle path.
      exec_swmmac_packed16(
          cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base, index_entries, index_key, ea, eb,
          read_acc, [fp16_ovfl](float val) { return wmma_round_bf16_rne(val, fp16_ovfl); },
          const_acc, wave_size);
      return;
    }
  }
  exec_swmmac_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base, index_entries, index_key, ea, eb,
      read_acc, [](float val) { return util::f32_to_bf16(val); }, const_acc, wave_size);
}

template <typename ExtractA, typename ExtractB, typename ScaleBlock>
void exec_f32_scaled_impl(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t a_bits,
                          uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                          ExtractA ea, ExtractB eb, ScaleBlock scale_block_sum, uint32_t const_acc,
                          uint32_t c_modifier) {
  constexpr uint32_t BLOCK_K = 32;
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  uint32_t num_blocks = (K + BLOCK_K - 1) / BLOCK_K;

  auto run_scalar = [&]() {
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          float acc =
              (const_acc != ACC_FROM_VGPR)
                  ? std::bit_cast<float>(const_acc)
                  : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
          acc = apply_wmma_c_modifier(acc, c_modifier);
          for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            float block_sum = 0.0f;
            uint32_t k_start = blk * BLOCK_K;
            uint32_t k_end = std::min(k_start + BLOCK_K, K);
            for (uint32_t k = k_start; k < k_end; ++k) {
              auto al = mfma_scale_f8f6f4_input_loc(M, K, row, k, a_bits);
              auto bl = mfma_scale_f8f6f4_input_loc(N, K, col, k, b_bits);
              const float a = ea(cu, s0, physicalize_loc(al, wf));
              const float b_val = eb(cu, s1, physicalize_loc(bl, wf));
              block_sum = std::fma(a, b_val, block_sum);
            }
            acc += scale_block_sum(block_sum, row, col, b, blk);
          }
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
        }
      }
    }
  };

  // SIMD fast path: hoist A/B into dense f32 buffers (lane permutation folded
  // in), then for each row accumulate each K-block's partial product as
  // native-width FMA rows over the N (column) dimension. The per-output E8M0
  // scale + ldexp accumulation stays scalar (cheap: O(num_blocks) per output
  // vs O(K) MACs). A scalar tail covers trailing N columns.
  if constexpr (util::has_stdx_simd) {
    // Column-padded, stack-allocated, aligned B loads. See exec_f32_mixed.
    // Cacc is touched scalar (per-output ldexp) so it keeps an N pitch.
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    constexpr size_t MAX_AB = 2048;
    constexpr size_t MAX_BSTRIDE = 4096;
    constexpr size_t MAX_C = 1024;
    static_assert((MAX_AB + MAX_BSTRIDE + MAX_C) * sizeof(float) <= 48 * 1024,
                  "MFMA SIMD staging buffers exceed the 48 KiB stack budget");
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * N > MAX_C) {
      run_scalar();
    } else {
      auto reads = read_mixed_matrix_fast_path_regions(
          cu, s0, s1, s2, static_cast<uint64_t>(M) * K * B, a_bits,
          static_cast<uint64_t>(N) * K * B, b_bits, static_cast<uint64_t>(M) * N * B,
          /*acc_bits=*/32, const_acc, wf);
      // Zero-initialized staging buffers (uniform convention; see
      // wmma_simd_matmul). Cacc is pre-seeded below.
      alignas(64) float Abuf[MAX_AB] = {};
      alignas(64) float Bbuf[MAX_BSTRIDE] = {};
      alignas(64) float Cacc[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t k = 0; k < K; ++k) {
            auto al = mfma_scale_f8f6f4_input_loc(M, K, row, k, a_bits);
            Abuf[row * K + k] = ea(reads.a, s0, physicalize_loc(al, wf));
          }
        for (uint32_t k = 0; k < K; ++k)
          for (uint32_t col = 0; col < N; ++col) {
            auto bl = mfma_scale_f8f6f4_input_loc(N, K, col, k, b_bits);
            Bbuf[k * stride + col] = eb(reads.b, s1, physicalize_loc(bl, wf));
          }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            Cacc[row * N + col] = apply_wmma_c_modifier(
                (const_acc != ACC_FROM_VGPR)
                    ? std::bit_cast<float>(const_acc)
                    : std::bit_cast<float>(reads.acc->lane(out.reg, out.lane)),
                c_modifier);
          }
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            uint32_t k_start = blk * BLOCK_K;
            uint32_t k_end = std::min(k_start + BLOCK_K, K);
            uint32_t col = 0;
            alignas(64) float bs[64];
            for (; col + W <= N; col += W) {
              util::native<float> acc(0.0f);
              for (uint32_t k = k_start; k < k_end; ++k) {
                util::native<float> a(Abuf[row * K + k]);
                util::native<float> bv;
                bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
                acc = util::stdx::fma(a, bv, acc);
              }
              acc.copy_to(bs, util::stdx::vector_aligned);
              for (uint32_t j = 0; j < W; ++j)
                Cacc[row * N + col + j] += scale_block_sum(bs[j], row, col + j, b, blk);
            }
            for (; col < N; ++col) {
              float block_sum = 0.0f;
              for (uint32_t k = k_start; k < k_end; ++k)
                block_sum = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], block_sum);
              Cacc[row * N + col] += scale_block_sum(block_sum, row, col, b, blk);
            }
          }
        }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cacc[row * N + col])});
          }
      }
    }
  } else {
    run_scalar();
  }

  auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
  for (const auto &r : results)
    writes.set_lane(r.reg, r.lane, r.val);
}

inline uint8_t mfma_inline_scale_e8m0(uint32_t selector) {
  switch (selector) {
  case 240: // +0.5f
  case 241: // -0.5f
    return 126;
  case 242: // +1.0f
  case 243: // -1.0f
    return 127;
  case 244: // +2.0f
  case 245: // -2.0f
    return 128;
  case 246: // +4.0f
  case 247: // -4.0f
    return 129;
  case 248: // 1/(2*pi)
    return 124;
  default:
    return 0;
  }
}

inline uint8_t read_mfma_scale_e8m0(auto &cu, uint32_t vgpr_base, uint32_t selector, uint32_t lane,
                                    uint32_t byte_index) {
  if (selector >= 240 && selector <= 248)
    return mfma_inline_scale_e8m0(selector);
  uint8_t byte_mask = static_cast<uint8_t>(1u << byte_index);
  uint32_t raw = RegisterAccess(cu).read_vgpr(src_base(vgpr_base, selector), lane, byte_mask);
  return static_cast<uint8_t>((raw >> (byte_index * 8)) & 0xffu);
}

/// Scaled MFMA for mixed-format f8f6f4: A and B may have different bit widths.
/// cbsz/blgp are used as format selectors (not lane permutations).
template <typename ExtractA, typename ExtractB>
void exec_f32_scaled_mixed(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                           uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                           uint32_t s2, ExtractA ea, ExtractB eb, uint32_t const_acc,
                           uint32_t vgpr_base, uint32_t scale_a, uint32_t scale_b,
                           uint32_t scale_a_byte, uint32_t scale_b_byte, uint32_t c_modifier = 0) {
  auto scale_block_sum = [&](float block_sum, uint32_t row, uint32_t col, uint32_t,
                             uint32_t blk) -> float {
    uint8_t sa_e8m0 = read_mfma_scale_e8m0(cu, vgpr_base, scale_a, M * blk + row, scale_a_byte);
    uint8_t sb_e8m0 = read_mfma_scale_e8m0(cu, vgpr_base, scale_b, N * blk + col, scale_b_byte);
    if (sa_e8m0 == 0xffu || sb_e8m0 == 0xffu)
      return std::numeric_limits<float>::quiet_NaN();
    int scale_exp = static_cast<int>(sa_e8m0) + static_cast<int>(sb_e8m0) - 254;
    return std::ldexp(block_sum, scale_exp);
  };
  exec_f32_scaled_impl(cu, M, N, K, B, a_bits, b_bits, dst, s0, s1, s2, ea, eb, scale_block_sum,
                       const_acc, c_modifier);
}

/// MFMA execute for i32 output with i8 input: D = C + A x B.
inline void exec_i32_i8(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t dst,
                        uint32_t s0, uint32_t s1, uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                        uint32_t cbsz = 0, uint32_t abid = 0, uint32_t blgp = 0) {
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);

  auto run_scalar = [&]() {
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          uint32_t acc = (const_acc != ACC_FROM_VGPR)
                             ? const_acc
                             : RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane);
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 8);
            auto bl = input_loc(N, K, B, col, k, b, 8);
            if (cbsz != 0)
              al.lane = permute_a_lane(al.lane, cbsz, abid);
            if (blgp != 0)
              bl.lane = permute_b_lane(bl.lane, blgp);
            acc += static_cast<uint32_t>(extract_i8(cu, s0, physicalize_loc(al, wf)) *
                                         extract_i8(cu, s1, physicalize_loc(bl, wf)));
          }
          results.push_back({out.reg, out.lane, acc});
        }
      }
    }
  };

  // SIMD fast path mirrors exec_f32_mixed: hoist A/B/C into dense int32
  // buffers, then run the matmul as native-width int32 multiply-accumulate
  // over the N dimension. Integer MAC is exact, so the SIMD and scalar paths
  // are bit-identical. A scalar tail handles trailing N columns.
  if constexpr (util::has_stdx_simd) {
    // Column-padded, stack-allocated, aligned. See exec_f32_mixed.
    constexpr uint32_t W = static_cast<uint32_t>(util::native<int32_t>::size());
    constexpr size_t MAX_AB = 2048;
    constexpr size_t MAX_BSTRIDE = 4096;
    constexpr size_t MAX_C = 1024;
    static_assert((MAX_AB + MAX_BSTRIDE + MAX_C) * sizeof(int32_t) <= 48 * 1024,
                  "MFMA SIMD staging buffers exceed the 48 KiB stack budget");
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * stride > MAX_C) {
      run_scalar();
    } else {
      // Zero-initialized staging buffers (uniform convention; see
      // wmma_simd_matmul). Cbuf is pre-seeded below.
      alignas(64) int32_t Abuf[MAX_AB] = {};
      alignas(64) int32_t Bbuf[MAX_BSTRIDE] = {};
      alignas(64) int32_t Cbuf[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            Cbuf[row * stride + col] =
                (const_acc != ACC_FROM_VGPR)
                    ? static_cast<int32_t>(const_acc)
                    : static_cast<int32_t>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
          }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 8);
            if (cbsz != 0)
              al.lane = permute_a_lane(al.lane, cbsz, abid);
            Abuf[row * K + k] = extract_i8(cu, s0, physicalize_loc(al, wf));
          }
        for (uint32_t k = 0; k < K; ++k)
          for (uint32_t col = 0; col < N; ++col) {
            auto bl = input_loc(N, K, B, col, k, b, 8);
            if (blgp != 0)
              bl.lane = permute_b_lane(bl.lane, blgp);
            Bbuf[k * stride + col] = extract_i8(cu, s1, physicalize_loc(bl, wf));
          }
        for (uint32_t row = 0; row < M; ++row) {
          uint32_t col = 0;
          for (; col + W <= N; col += W) {
            util::native<int32_t> c;
            c.copy_from(&Cbuf[row * stride + col], util::stdx::vector_aligned);
            for (uint32_t k = 0; k < K; ++k) {
              util::native<int32_t> a(Abuf[row * K + k]);
              util::native<int32_t> bv;
              bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
              c += a * bv;
            }
            c.copy_to(&Cbuf[row * stride + col], util::stdx::vector_aligned);
          }
          for (; col < N; ++col) {
            uint32_t acc = static_cast<uint32_t>(Cbuf[row * stride + col]);
            for (uint32_t k = 0; k < K; ++k)
              acc += static_cast<uint32_t>(Abuf[row * K + k] * Bbuf[k * stride + col]);
            Cbuf[row * stride + col] = static_cast<int32_t>(acc);
          }
        }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            results.push_back({out.reg, out.lane, static_cast<uint32_t>(Cbuf[row * stride + col])});
          }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_i32(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst,
                   uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb, bool clamp,
                   uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        int64_t acc = (const_acc != ACC_FROM_VGPR)
                          ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                          : static_cast<int64_t>(static_cast<int32_t>(
                                RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane)));
        for (uint32_t k = 0; k < K; ++k) {
          auto al = gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits);
          auto bl = gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits);
          acc += static_cast<int64_t>(ea(cu, s0, al)) * static_cast<int64_t>(eb(cu, s1, bl));
        }
        results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
      }
    }
  };

  // SIMD fast path: the K-sum of products is exact in int32 for WMMA's i8/i4
  // inputs at K <= 128 (max |sum| ~2M << 2^31). The int32 accumulator is added
  // in 64-bit at pack time so pack_i32_acc saturates exactly like the scalar
  // int64 reference even when C sits near INT32_MAX/INT32_MIN.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<int32_t>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) int32_t Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) int32_t Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) int32_t Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k)
          Abuf[row * K + k] = ea(cu, s0, gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits));
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col)
          Bbuf[k * stride + col] =
              eb(cu, s1, gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits));
      wmma_simd_matmul<int32_t>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          int64_t acc = (const_acc != ACC_FROM_VGPR)
                            ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                            : static_cast<int64_t>(static_cast<int32_t>(
                                  RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane)));
          acc += static_cast<int64_t>(Cbuf[row * stride + col]);
          results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
        }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_i32(auto &cu, uint32_t wave_size, uint32_t M, uint32_t N, uint32_t K,
                         uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                         ExtractA ea, ExtractB eb, bool clamp, uint32_t const_acc = ACC_FROM_VGPR) {
  require_gfx11_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = gfx11_wmma_output_loc_32(wave_size, M, N, row, col);
      uint32_t lane_group = out.lane / N;
      int64_t acc = (const_acc != ACC_FROM_VGPR)
                        ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                        : static_cast<int64_t>(static_cast<int32_t>(
                              RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane)));
      for (uint32_t k = 0; k < K; ++k) {
        auto al = gfx11_wmma_input_loc(M, K, row, k, in_bits, lane_group);
        auto bl = gfx11_wmma_input_loc(N, K, col, k, in_bits, lane_group);
        acc += static_cast<int64_t>(ea(cu, s0, al)) * static_cast<int64_t>(eb(cu, s1, bl));
      }
      results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
    }
  }

  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

inline void exec_wmma_i32_i8(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t dst,
                             uint32_t s0, uint32_t s1, uint32_t s2,
                             uint32_t const_acc = ACC_FROM_VGPR) {
  exec_wmma_i32(cu, M, N, K, 8, dst, s0, s1, s2, extract_i8, extract_i8, false, const_acc);
}

/// Fast path for v_wmma_i32_16x16x64_iu8 (gfx1250, wave32, dense). Same recipe
/// as the f16/bf16 specializations: bulk sign-/zero-extend the packed i8
/// inputs (per the A/B sign selects from the neg bits) to int32 once, then run
/// a constexpr-unrolled int32 matmul over observed register-access regions. The K-sum
/// of products is exact in int32 (|sum| <= 64 * 16384) and is added to the
/// int32 accumulator in 64-bit at pack time, so clamp saturation matches the
/// scalar int64 reference bit for bit; with clamp off both wrap mod 2^32.
/// Falls back to the generic exec_wmma_i32 without AVX-512 / under
/// force-scalar.
inline void exec_wmma_i32_16x16x64_iu8(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                       uint32_t s2, bool a_signed, bool b_signed, bool clamp,
                                       uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t M = 16, N = 16, K = 64, in_bits = 8;
  auto fallback = [&]() {
    auto extract_a = [a_signed](auto &cu, uint32_t base, const InputLoc &loc) {
      return a_signed ? extract_i8(cu, base, loc) : extract_u8(cu, base, loc);
    };
    auto extract_b = [b_signed](auto &cu, uint32_t base, const InputLoc &loc) {
      return b_signed ? extract_i8(cu, base, loc) : extract_u8(cu, base, loc);
    };
    exec_wmma_i32(cu, M, N, K, in_bits, dst, s0, s1, s2, extract_a, extract_b, clamp, const_acc);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    const uint32_t wf = cu.wf_size();
    auto reads = read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, in_bits, /*acc_bits=*/32,
                                             const_acc, wf);
    // Accumulate in unsigned 32-bit (wrap is well-defined; identical mod 2^32
    // to the intended signed wrap), sign-restore via int32 cast at pack time.
    alignas(64) uint32_t A_buf[M * K]; // A[row][k] (sign-/zero-extended bits)
    alignas(64) uint32_t B_buf[K * N]; // B[k][col] (sign-/zero-extended bits)
    alignas(64) uint32_t S_buf[M * N]; // sum-of-products, accumulator added at pack
    alignas(64) uint32_t C_words[M * N];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    // Bulk-extend the packed byte regions to int32 once, then the hoist is a
    // pure i32 index-shuffle (byte j of word w lane l sub s -> (w*wf+l)*4+s).
    alignas(64) int32_t A_i32[M * K];
    alignas(64) int32_t B_i32[N * K];
    if (a_signed)
      convert_i8_matrix_region(reads.a, A_i32, M * K);
    else
      convert_u8_matrix_region(reads.a, A_i32, M * K);
    if (b_signed)
      convert_i8_matrix_region(reads.b, B_i32, N * K);
    else
      convert_u8_matrix_region(reads.b, B_i32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_i32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_i32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
      }
    // Dense 16x64 * 64x16 -> 16x16 product-sum, 16-lane stdx u32 MAC per row
    // (unsigned to avoid signed-overflow UB; bits match the signed math).
    for (uint32_t row = 0; row < M; ++row) {
      util::native<uint32_t> s_row(0);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<uint32_t> a_bcast(A_buf[row * K + k]);
        util::native<uint32_t> b_row;
        b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
        s_row += a_bcast * b_row;
      }
      s_row.copy_to(&S_buf[row * N], util::stdx::vector_aligned);
    }
    // Add the accumulator in 64-bit and pack (saturating when clamp is set),
    // scattering directly back to VGPRs.
    auto writes = write_wmma_output_region(cu, dst, M, N, /*output_bits=*/32, wf);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        int64_t acc =
            (const_acc != ACC_FROM_VGPR)
                ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                : static_cast<int64_t>(static_cast<int32_t>(C_words[out.reg * wf + out.lane]));
        acc += static_cast<int64_t>(static_cast<int32_t>(S_buf[row * N + col]));
        writes.set_linear_word(out.reg * wf + out.lane, pack_i32_acc(acc, clamp));
      }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_i32(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst,
                     uint32_t s0, uint32_t s1, uint32_t acc_base, uint32_t index_base,
                     uint32_t index_entries, uint32_t index_key, ExtractA ea, ExtractB eb,
                     bool clamp, uint32_t const_acc = ACC_FROM_VGPR,
                     uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  const uint32_t compressed_k = K / 2;

  auto dense_k_for = [&](uint32_t row, uint32_t ck) -> uint32_t {
    const auto index_loc = swmmac_index_loc(wave_size, M, K, in_bits, row, ck, index_entries);
    const uint64_t index_set =
        read_swmmac_index_set(cu, index_base, index_loc.lane, index_entries, index_key);
    return swmmac_dense_k(index_set, ck, index_loc.local_compressed_k);
  };

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        int64_t acc = (const_acc != ACC_FROM_VGPR)
                          ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                          : static_cast<int64_t>(static_cast<int32_t>(
                                RegisterAccess(cu).read_vgpr(acc_base + out.reg, out.lane)));
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          auto al = swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits);
          auto bl = swmmac_b_input_loc(wave_size, N, K, col, dense_k_for(row, ck), in_bits);
          acc += static_cast<int64_t>(ea(cu, s0, al)) * static_cast<int64_t>(eb(cu, s1, bl));
        }
        results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
      }
    }
  };

  // SIMD fast path: per-row gather, single row through the int32 matmul core
  // (M=1). The K-sum of products is exact in int32 for i8/i4 inputs (no
  // overflow); the int32 accumulator is added in 64-bit at pack time so
  // pack_i32_acc saturates exactly like the scalar int64 reference even when
  // C sits near INT32_MAX/INT32_MIN.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<int32_t>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(compressed_k) > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(compressed_k) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) int32_t Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) int32_t Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) int32_t Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col)
          Cbuf[col] = 0;
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          Abuf[ck] = ea(cu, s0, swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits));
          const uint32_t dense_k = dense_k_for(row, ck);
          for (uint32_t col = 0; col < N; ++col)
            Bbuf[ck * stride + col] =
                eb(cu, s1, swmmac_b_input_loc(wave_size, N, K, col, dense_k, in_bits));
        }
        wmma_simd_matmul<int32_t>(1, N, compressed_k, W, stride, Abuf, Bbuf, Cbuf);
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          int64_t acc = (const_acc != ACC_FROM_VGPR)
                            ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                            : static_cast<int64_t>(static_cast<int32_t>(
                                  RegisterAccess(cu).read_vgpr(acc_base + out.reg, out.lane)));
          acc += static_cast<int64_t>(Cbuf[col]);
          results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
        }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

inline void exec_swmmac_i32_i8(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t dst,
                               uint32_t s0, uint32_t s1, uint32_t acc_base, uint32_t index_base,
                               uint32_t index_entries, uint32_t index_key,
                               uint32_t const_acc = ACC_FROM_VGPR) {
  exec_swmmac_i32(cu, M, N, K, 8, dst, s0, s1, acc_base, index_base, index_entries, index_key,
                  extract_i8, extract_i8, false, const_acc);
}

/// Execute gfx1251 wave32 V_WMMA_F64_16X16X4_F64.
///
/// Public evidence for the logical matrix operation, vector widths, modifiers,
/// and register tuple sizes comes from LLVM's F64_F64X4_WMMA_w32 profile and
/// gfx1251_asm_wmma_w32.s. The physical mapping combines section 7.12.2 of the
/// public CDNA5 ISA reference with the checked-in FMT_WMMA_AB_16X4_F64 and
/// FMT_WMMA_DC_16X16_F64 component layouts:
///   A[row][k] / B[col][k]: lane = row-or-col + 16*(k/2), VGPR = 2*(k%2)
///   C/D[row][col]: lane = col + 16*(row/8), VGPR = 2*(row%8)
///
/// Matrix reuse bits are scheduling hints and therefore do not enter this
/// functional helper. All source and accumulator values are staged before any
/// destination write so overlapping C/D and A/B tuples preserve read-before-
/// write behavior. CDNA5 also requires EXEC to contain all 32 wave lanes.
inline void exec_wmma_f64_16x16x4_f64(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                      uint64_t const_acc, uint32_t neg, uint32_t neg_hi,
                                      [[maybe_unused]] uint64_t exec_mask) {
  // LLVM marks GFX12 WMMA as ReadsModeReg=0. Until physical gfx1251 validation
  // establishes a more specific fixed arithmetic policy, use the IEEE baseline:
  // round-to-nearest-even and preserve input/output denormals.
  constexpr uint32_t kRoundNearestEven = 0;
  constexpr uint32_t kPreserveInputOutputDenormals = 3;
  // TODO(hanchung): Validate gfx1251 F64 WMMA lane/register mapping and fixed
  // arithmetic behavior against a physical gfx1251 KFD/ROCm execution capture.
  constexpr uint32_t M = 16;
  constexpr uint32_t N = 16;
  constexpr uint32_t K = 4;
  const uint64_t kFullExec = mfma_full_lane_mask(WMMA_WAVE32);
  assert(is_gfx1251_wmma_execution_state_valid(cu.wf_size(), exec_mask) &&
         "gfx1251 F64 WMMA requires wave32 with EXEC set to all ones");

  auto toggle_sign = [](uint64_t value, bool toggle) {
    return toggle ? value ^ (uint64_t{1} << 63) : value;
  };
  auto modify_acc = [](uint64_t value, uint32_t modifier) {
    if (modifier & 0x2u)
      value &= ~(uint64_t{1} << 63);
    if (modifier & 0x1u)
      value ^= uint64_t{1} << 63;
    return value;
  };

  std::array<uint64_t, M * K> a{};
  std::array<uint64_t, K * N> b{};
  std::array<uint64_t, M * N> result{};
  {
    auto reads =
        read_wmma_fast_path_regions(cu, s0, s1, s2, M, N, K, /*data_bits=*/64, /*acc_bits=*/64,
                                    const_acc == ACC_FROM_VGPR ? ACC_FROM_VGPR : 0u, WMMA_WAVE32);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        const auto loc = wmma_input_loc(M, K, row, k, /*data_bits=*/64);
        a[row * K + k] = toggle_sign(reads.a.lane64(loc.vgpr_offset, loc.lane), (neg & 0x1u) != 0);
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        const auto loc = wmma_input_loc(N, K, col, k, /*data_bits=*/64);
        b[k * N + col] = toggle_sign(reads.b.lane64(loc.vgpr_offset, loc.lane), (neg & 0x2u) != 0);
      }

    const uint32_t c_modifier = wmma_c_modifier(neg, neg_hi);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        const auto out = wmma_output_loc_64(M, N, row, col);
        uint64_t acc =
            const_acc == ACC_FROM_VGPR ? reads.acc->lane64(out.reg, out.lane) : const_acc;
        acc = modify_acc(acc, c_modifier);
        for (uint32_t k = 0; k < K; ++k)
          acc = fp_mode::fma_f64(a[row * K + k], b[k * N + col], acc, kRoundNearestEven,
                                 kPreserveInputOutputDenormals);
        result[row * N + col] = acc;
      }
  }

  RegisterAccess regs(cu);
  auto writes = regs.write_vgpr_region(dst, /*reg_count=*/16, kFullExec);
  for (uint32_t row = 0; row < M; ++row)
    for (uint32_t col = 0; col < N; ++col) {
      const auto out = wmma_output_loc_64(M, N, row, col);
      const uint64_t value = result[row * N + col];
      writes.set_lane(out.reg, out.lane, static_cast<uint32_t>(value));
      writes.set_lane(out.reg + 1, out.lane, static_cast<uint32_t>(value >> 32));
    }
}

/// MFMA execute for f64 output with f64 input: D = C + A x B.
inline void exec_f64(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t dst,
                     uint32_t s0, uint32_t s1, uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                     uint32_t neg = 0) {
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t lo;
    uint32_t hi;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  auto apply_neg = [neg](double value, uint32_t bit) {
    return (neg & bit) ? std::bit_cast<double>(std::bit_cast<uint64_t>(value) ^ (uint64_t{1} << 63))
                       : value;
  };

  auto run_scalar = [&]() {
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          // AMD convention: i=row (register dimension), j=col (lane dimension).
          auto out = output_loc_64(M, N, row, col, b);
          double acc;
          if (const_acc != ACC_FROM_VGPR) {
            acc = static_cast<double>(std::bit_cast<float>(const_acc));
          } else {
            uint32_t lo = RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane);
            uint32_t hi = RegisterAccess(cu).read_vgpr(s2 + out.reg + 1, out.lane);
            acc = std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
          }
          acc = apply_neg(acc, 0x4u);
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 64);
            auto bl = input_loc(N, K, B, col, k, b, 64);
            acc +=
                apply_neg(extract_f64(cu, s0, al), 0x1u) * apply_neg(extract_f64(cu, s1, bl), 0x2u);
          }
          uint64_t bits = std::bit_cast<uint64_t>(acc);
          results.push_back(
              {out.reg, out.lane, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)});
        }
      }
    }
  };

  // SIMD fast path mirrors exec_f32_mixed with native<double> lanes (8-wide on
  // AVX-512) and fused FMA, matching the GFX9 f64 MFMA single-rounding MACs.
  // A scalar (fused) tail covers the trailing N columns.
  if constexpr (util::has_stdx_simd) {
    // Column-padded, stack-allocated, aligned. See exec_f32_mixed. MAX_BSTRIDE
    // is half the f32 cap because native<double> packs half as many lanes.
    constexpr uint32_t W = static_cast<uint32_t>(util::native<double>::size());
    constexpr size_t MAX_AB = 2048;
    constexpr size_t MAX_BSTRIDE = 2048;
    constexpr size_t MAX_C = 1024;
    static_assert((MAX_AB + MAX_BSTRIDE + MAX_C) * sizeof(double) <= 48 * 1024,
                  "MFMA SIMD staging buffers exceed the 48 KiB stack budget");
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * stride > MAX_C) {
      run_scalar();
    } else {
      // Zero-initialized staging buffers (uniform convention; see
      // wmma_simd_matmul). Cbuf is pre-seeded below.
      alignas(64) double Abuf[MAX_AB] = {};
      alignas(64) double Bbuf[MAX_BSTRIDE] = {};
      alignas(64) double Cbuf[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = output_loc_64(M, N, row, col, b);
            if (const_acc != ACC_FROM_VGPR) {
              Cbuf[row * stride + col] = static_cast<double>(std::bit_cast<float>(const_acc));
            } else {
              uint32_t lo = RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane);
              uint32_t hi = RegisterAccess(cu).read_vgpr(s2 + out.reg + 1, out.lane);
              Cbuf[row * stride + col] =
                  std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
            }
            Cbuf[row * stride + col] = apply_neg(Cbuf[row * stride + col], 0x4u);
          }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 64);
            Abuf[row * K + k] = apply_neg(extract_f64(cu, s0, al), 0x1u);
          }
        for (uint32_t k = 0; k < K; ++k)
          for (uint32_t col = 0; col < N; ++col) {
            auto bl = input_loc(N, K, B, col, k, b, 64);
            Bbuf[k * stride + col] = apply_neg(extract_f64(cu, s1, bl), 0x2u);
          }
        for (uint32_t row = 0; row < M; ++row) {
          uint32_t col = 0;
          for (; col + W <= N; col += W) {
            util::native<double> c;
            c.copy_from(&Cbuf[row * stride + col], util::stdx::vector_aligned);
            for (uint32_t k = 0; k < K; ++k) {
              util::native<double> a(Abuf[row * K + k]);
              util::native<double> bv;
              bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
              c = util::stdx::fma(a, bv, c);
            }
            c.copy_to(&Cbuf[row * stride + col], util::stdx::vector_aligned);
          }
          for (; col < N; ++col) {
            double acc = Cbuf[row * stride + col];
            for (uint32_t k = 0; k < K; ++k)
              acc = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], acc);
            Cbuf[row * stride + col] = acc;
          }
        }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = output_loc_64(M, N, row, col, b);
            uint64_t bits = std::bit_cast<uint64_t>(Cbuf[row * stride + col]);
            results.push_back({out.reg, out.lane, static_cast<uint32_t>(bits),
                               static_cast<uint32_t>(bits >> 32)});
          }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results) {
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.lo);
    RegisterAccess(cu).write_vgpr(dst + r.reg + 1, r.lane, r.hi);
  }
}

/// SMFMAC 16x16x32 f16/bf16 (CDNA3 mai-insts). K=32, 8 sparse groups.
/// A = v2 (4 halves/lane), B = v4 (8 halves/lane), D = v4 f32.
template <typename Extract>
void exec_smfmac_f32_16x16x32_f16(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, Extract ex) {
  if (smfmac_try_avx512<SmfmacLayout::Cdna3F16, 16, 16, 32>(cu, dst, s0, s1, idx_base, ex, ex))
    return;
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(16 * 16);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      auto out = output_loc_32(16, 16, row, col, 0);
      float acc = std::bit_cast<float>(RegisterAccess(cu).read_vgpr(dst + out.reg, out.lane));
      float Bcol[32];
      for (int g = 0; g < 4; ++g)
        for (int e = 0; e < 8; ++e)
          Bcol[8 * g + e] = ex(cu, s1, e, g * 16 + col);
      for (int q = 0; q < 8; ++q) {
        int laneA = (q / 2) * 16 + row;
        uint32_t idxval = RegisterAccess(cu).read_vgpr(idx_base, laneA);
        int field = (idxval >> (4 * (q % 2))) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          float av = ex(cu, s0, (2 * q + s) % 4, laneA);
          acc = matrix_fma(av, Bcol[4 * q + (s == 0 ? p0 : p1)], acc);
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 32x32x16 f16/bf16 (CDNA3 mai-insts). K=16, 4 sparse groups.
/// A = v2 (4 halves/lane), B = v4 (8 halves/lane), D = v16 f32.
template <typename Extract>
void exec_smfmac_f32_32x32x16_f16(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, Extract ex) {
  if (smfmac_try_avx512<SmfmacLayout::Cdna3F16, 32, 32, 16>(cu, dst, s0, s1, idx_base, ex, ex))
    return;
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(32 * 32);
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t col = 0; col < 32; ++col) {
      auto out = output_loc_32(32, 32, row, col, 0);
      float acc = std::bit_cast<float>(RegisterAccess(cu).read_vgpr(dst + out.reg, out.lane));
      uint32_t jlow = col % 16, jhi = col / 16;
      float Bcol[16];
      for (int kgrp = 0; kgrp < 2; ++kgrp) {
        uint32_t b_lane = 16 * (jhi + 2 * kgrp) + jlow;
        for (int e = 0; e < 8; ++e)
          Bcol[8 * kgrp + e] = ex(cu, s1, e, b_lane);
      }
      for (int q = 0; q < 4; ++q) {
        int laneA = (q / 2) * 32 + row;
        uint32_t idxval = RegisterAccess(cu).read_vgpr(idx_base, laneA);
        int field = (idxval >> (4 * (q % 2))) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          float av = ex(cu, s0, (2 * q + s) % 4, laneA);
          acc = matrix_fma(av, Bcol[4 * q + (s == 0 ? p0 : p1)], acc);
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 16x16x64 f16/bf16 (gfx950-insts). K=64, 16 sparse groups.
/// A = v4 (8 halves/lane), B = v8 (16 halves/lane), D = v4 f32.
template <typename Extract>
void exec_smfmac_f32_16x16x64_f16(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, Extract ex) {
  if (smfmac_try_avx512<SmfmacLayout::Cdna4F16, 16, 16, 64>(cu, dst, s0, s1, idx_base, ex, ex))
    return;
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(16 * 16);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      auto out = output_loc_32(16, 16, row, col, 0);
      float acc = std::bit_cast<float>(RegisterAccess(cu).read_vgpr(dst + out.reg, out.lane));
      float Bcol[64];
      for (int g = 0; g < 4; ++g)
        for (int e = 0; e < 16; ++e) {
          int k = 32 * (e / 8) + 8 * g + (e % 8);
          Bcol[k] = ex(cu, s1, e, g * 16 + col);
        }
      for (int q = 0; q < 16; ++q) {
        int laneA = (q / 4) * 16 + row;
        uint32_t idxval = RegisterAccess(cu).read_vgpr(idx_base, laneA);
        int field = (idxval >> (4 * (q % 4))) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          float av = ex(cu, s0, (2 * q + s) % 8, laneA);
          acc = matrix_fma(av, Bcol[4 * q + (s == 0 ? p0 : p1)], acc);
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 32x32x32 f16/bf16 (gfx950-insts). K=32, 8 sparse groups.
/// A = v4 (8 halves/lane), B = v8 (16 halves/lane), D = v16 f32.
template <typename Extract>
void exec_smfmac_f32_32x32x32_f16(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, Extract ex) {
  if (smfmac_try_avx512<SmfmacLayout::Cdna4F16, 32, 32, 32>(cu, dst, s0, s1, idx_base, ex, ex))
    return;
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(32 * 32);
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t col = 0; col < 32; ++col) {
      auto out = output_loc_32(32, 32, row, col, 0);
      float acc = std::bit_cast<float>(RegisterAccess(cu).read_vgpr(dst + out.reg, out.lane));
      float Bcol[32];
      for (int sl = 0; sl < 2; ++sl) {
        uint32_t src = col + 32 * sl;
        int kgrp = sl;
        for (int e = 0; e < 16; ++e) {
          int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
          Bcol[k] = ex(cu, s1, e, src);
        }
      }
      for (int q = 0; q < 8; ++q) {
        int laneA = (q / 4) * 32 + row;
        uint32_t idxval = RegisterAccess(cu).read_vgpr(idx_base, laneA);
        int field = (idxval >> (4 * (q % 4))) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          float av = ex(cu, s0, (2 * q + s) % 8, laneA);
          acc = matrix_fma(av, Bcol[4 * q + (s == 0 ? p0 : p1)], acc);
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 16x16x64 fp8 (CDNA3 fp8-insts). K=64, 16 sparse groups.
/// A = v2 (8 bytes/lane), B = v4 (16 bytes/lane), D = v4 f32.
template <typename ExtractA, typename ExtractB>
void exec_smfmac_f32_16x16x64_fp8(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, ExtractA ea, ExtractB eb) {
  if (smfmac_try_avx512<SmfmacLayout::Cdna3Fp8, 16, 16, 64>(cu, dst, s0, s1, idx_base, ea, eb))
    return;
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(16 * 16);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      auto out = output_loc_32(16, 16, row, col, 0);
      float acc = std::bit_cast<float>(RegisterAccess(cu).read_vgpr(dst + out.reg, out.lane));
      float Bcol[64];
      for (int g = 0; g < 4; ++g)
        for (int e = 0; e < 16; ++e) {
          int k = 32 * (e / 8) + 8 * g + (e % 8);
          Bcol[k] = eb(cu, s1, e, g * 16 + col);
        }
      for (int q = 0; q < 16; ++q) {
        int ga = ((2 * q) % 16) / 4;
        int laneA = ga * 16 + row;
        int idxlane = ((q % 8) / 2) * 16 + row;
        int nb = 2 * (q / 8) + (q % 2);
        uint32_t idxval = RegisterAccess(cu).read_vgpr(idx_base, idxlane);
        int field = (idxval >> (4 * nb)) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          int cc = 2 * q + s;
          int byte = 4 * (cc / 16) + (cc % 16) % 4;
          float av = ea(cu, s0, byte, laneA);
          acc = matrix_fma(av, Bcol[4 * q + (s == 0 ? p0 : p1)], acc);
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 32x32x32 fp8 (CDNA3 fp8-insts). K=32, 8 sparse groups.
/// A = v2 (8 bytes/lane), B = v4 (16 bytes/lane), D = v16 f32.
template <typename ExtractA, typename ExtractB>
void exec_smfmac_f32_32x32x32_fp8(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, ExtractA ea, ExtractB eb) {
  if (smfmac_try_avx512<SmfmacLayout::Cdna3Fp8, 32, 32, 32>(cu, dst, s0, s1, idx_base, ea, eb))
    return;
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(32 * 32);
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t col = 0; col < 32; ++col) {
      auto out = output_loc_32(32, 32, row, col, 0);
      float acc = std::bit_cast<float>(RegisterAccess(cu).read_vgpr(dst + out.reg, out.lane));
      float Bcol[32];
      for (int kgrp = 0; kgrp < 2; ++kgrp) {
        uint32_t b_lane = col + 32 * kgrp;
        for (int e = 0; e < 16; ++e) {
          int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
          Bcol[k] = eb(cu, s1, e, b_lane);
        }
      }
      for (int q = 0; q < 8; ++q) {
        int ga = ((2 * q) % 8) / 4;
        int laneA = ga * 32 + row;
        int idxlane = ((q % 4) / 2) * 32 + row;
        int nb = 2 * (q / 4) + (q % 2);
        uint32_t idxval = RegisterAccess(cu).read_vgpr(idx_base, idxlane);
        int field = (idxval >> (4 * nb)) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          int cc = 2 * q + s;
          int byte = 4 * (cc / 8) + (cc % 8) % 4;
          float av = ea(cu, s0, byte, laneA);
          acc = matrix_fma(av, Bcol[4 * q + (s == 0 ? p0 : p1)], acc);
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 16x16x128 fp8 (gfx950-insts). K=128, 32 sparse groups.
/// A = v4 (16 bytes/lane), B = v8 (32 bytes/lane), D = v4 f32.
template <typename ExtractA, typename ExtractB>
void exec_smfmac_f32_16x16x128_fp8(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                   uint32_t idx_base, ExtractA ea, ExtractB eb) {
  if (smfmac_try_avx512<SmfmacLayout::Cdna4Fp8, 16, 16, 128>(cu, dst, s0, s1, idx_base, ea, eb))
    return;
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(16 * 16);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      auto out = output_loc_32(16, 16, row, col, 0);
      float acc = std::bit_cast<float>(RegisterAccess(cu).read_vgpr(dst + out.reg, out.lane));
      float Bcol[128];
      for (int g = 0; g < 4; ++g)
        for (int e = 0; e < 32; ++e) {
          int k = 32 * (e / 8) + 8 * g + (e % 8);
          Bcol[k] = eb(cu, s1, e, g * 16 + col);
        }
      for (int q = 0; q < 32; ++q) {
        int idxlane = 16 * (2 * (q / 16) + ((q / 4) % 2)) + row;
        int nb = 2 * ((q / 8) % 2) + 4 * ((q % 4) / 2) + ((q % 4) % 2);
        uint32_t idxval = RegisterAccess(cu).read_vgpr(idx_base, idxlane);
        int field = (idxval >> (4 * nb)) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          int cc = 2 * q + s;
          int ga = 2 * ((cc >> 5) & 1) + ((cc >> 3) & 1);
          int hb = 2 * ((cc >> 2) & 1) + ((cc >> 4) & 1);
          int byte = 4 * hb + (cc & 3);
          float av = ea(cu, s0, byte, ga * 16 + row);
          acc = matrix_fma(av, Bcol[4 * q + (s == 0 ? p0 : p1)], acc);
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 32x32x64 fp8 (gfx950-insts). K=64, 16 sparse groups.
/// A = v4 (16 bytes/lane), B = v8 (32 bytes/lane), D = v16 f32.
template <typename ExtractA, typename ExtractB>
void exec_smfmac_f32_32x32x64_fp8(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, ExtractA ea, ExtractB eb) {
  if (smfmac_try_avx512<SmfmacLayout::Cdna4Fp8, 32, 32, 64>(cu, dst, s0, s1, idx_base, ea, eb))
    return;
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(32 * 32);
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t col = 0; col < 32; ++col) {
      auto out = output_loc_32(32, 32, row, col, 0);
      float acc = std::bit_cast<float>(RegisterAccess(cu).read_vgpr(dst + out.reg, out.lane));
      float Bcol[64];
      for (int kgrp = 0; kgrp < 2; ++kgrp) {
        uint32_t b_lane = kgrp * 32 + col;
        for (int e = 0; e < 32; ++e) {
          int k = 16 * (e / 8) + 8 * kgrp + (e % 8);
          Bcol[k] = eb(cu, s1, e, b_lane);
        }
      }
      for (int q = 0; q < 16; ++q) {
        int idxlane = 32 * (q / 8) + row;
        int nb = (q % 2) + 2 * ((q / 4) % 2) + 4 * ((q / 2) % 2);
        uint32_t idxval = RegisterAccess(cu).read_vgpr(idx_base, idxlane);
        int field = (idxval >> (4 * nb)) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          int cc = 2 * q + s;
          int ga = (cc >> 4) & 1;
          int hb = 2 * ((cc >> 2) & 1) + ((cc >> 3) & 1);
          int byte = 4 * hb + (cc & 3);
          float av = ea(cu, s0, byte, ga * 32 + row);
          acc = matrix_fma(av, Bcol[4 * q + (s == 0 ? p0 : p1)], acc);
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
}

/// Fast path for the f32-input MFMA shapes (v_mfma_f32_*_f32). Like the f16
/// specialization but the inputs are already f32, so there is no F16C convert —
/// the hoist reads each operand word straight through observed register-access
/// regions. BATCH covers the batched shapes (e.g. 32x32x1x2). Compile-time
/// M/N/K/BATCH fully unroll the matmul; it loops over N in native-width chunks.
/// Falls back to the generic exec_f32 without host SIMD, when the native width
/// is one or does not divide N, for non-wave64 execution, under force-scalar,
/// or with cbsz/blgp.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH>
void exec_f32_mfma_f32_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                            uint32_t const_acc, uint32_t cbsz, uint32_t abid, uint32_t blgp) {
  constexpr uint32_t in_bits = 32;
  if constexpr (!util::has_stdx_simd) {
    exec_f32(cu, M, N, K, BATCH, in_bits, dst, s0, s1, s2, amdgpu::extract_f32, amdgpu::extract_f32,
             const_acc, cbsz, abid, blgp);
    return;
  } else {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    if (util::force_scalar() || cbsz != 0 || blgp != 0 || !mma_f32_native_width_supported(N, W) ||
        cu.wf_size() != 64) {
      exec_f32(cu, M, N, K, BATCH, in_bits, dst, s0, s1, s2, amdgpu::extract_f32,
               amdgpu::extract_f32, const_acc, cbsz, abid, blgp);
      return;
    }
    const uint32_t wf = cu.wf_size();
    auto reads =
        read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, BATCH, in_bits, const_acc, wf);
    auto writes = write_mfma_acc32_region(cu, dst, M, N, BATCH, wf);
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    static_assert(M * N * BATCH * sizeof(float) <= 8 * 1024,
                  "specialized MFMA result staging exceeds the stack budget");
    alignas(64) float C_buf[M * N * BATCH];
    alignas(64) uint32_t A_words[M * K * BATCH];
    alignas(64) uint32_t B_words[N * K * BATCH];
    alignas(64) uint32_t C_words[M * N * BATCH];
    copy_matrix_region_words(reads.a, A_words);
    copy_matrix_region_words(reads.b, B_words);
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < BATCH; ++b) {
      float *C_batch = &C_buf[b * M * N];
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_batch[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                       ? std::bit_cast<float>(const_acc)
                                       : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, BATCH, row, k, b, in_bits);
          A_buf[row * K + k] = std::bit_cast<float>(A_words[al.vgpr_offset * wf + al.lane]);
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, BATCH, col, k, b, in_bits);
          B_buf[k * N + col] = std::bit_cast<float>(B_words[bl.vgpr_offset * wf + bl.lane]);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<float> c_row;
          c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<float> a_bcast(A_buf[row * K + k]);
            util::native<float> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row = util::stdx::fma(a_bcast, b_row, c_row);
          }
          c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
        }
    }
    // Publish only after every batch has consumed its inputs.
    for (uint32_t b = 0; b < BATCH; ++b)
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          float fv = C_buf[(b * M + row) * N + col];
          writes.set_linear_word(out.reg * wf + out.lane, std::bit_cast<uint32_t>(fv));
          if (std::isnan(fv) || std::isinf(fv))
            has_nan_or_inf = true;
        }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f32", dst,
                          s0, s1, s2, M, N, K);
      });
    }
  }
}

/// Fast path for the f16-input f32 MFMA shapes. The original target,
/// v_mfma_f32_16x16x32_f16, accounts for 488k invocations and about 4B
/// internal MACs per OPT-125M fp16 eager forward. Compile-time M/N/K/B let the
/// compiler fully unroll the inner matmul into straight-line native SIMD FMAs;
/// a runtime-dimension loop is materially slower on this hot path. The path
/// snapshots packed f16 inputs through observed register-access regions,
/// bulk-converts them into dense f32 buffers, and publishes staged results only
/// after every operand read. It falls back to the generic exec_f32 when:
///   - <experimental/simd> is unavailable
///   - host native_simd<float> has no usable width for the output columns
///   - cbsz/blgp lane permutation is non-default
///   - RJ_FORCE_SCALAR is set
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
void exec_f32_mfma_f16_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                            uint32_t const_acc, uint32_t cbsz, uint32_t abid, uint32_t blgp) {
  constexpr uint32_t B = BATCH, in_bits = 16;
  if constexpr (!util::has_stdx_simd) {
    exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
             const_acc, cbsz, abid, blgp);
    return;
  } else {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    if (util::force_scalar() || cbsz != 0 || blgp != 0 || !mma_f32_native_width_supported(N, W) ||
        cu.wf_size() != 64) {
      exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
               const_acc, cbsz, abid, blgp);
      return;
    }
    const uint32_t wf = cu.wf_size();
    auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
    auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
    alignas(64) float A_buf[M * K]; // A[row][k] (one batch block)
    alignas(64) float B_buf[K * N]; // B[k][col] (one batch block)
    static_assert(M * N * B * sizeof(float) <= 8 * 1024,
                  "specialized MFMA result staging exceeds the stack budget");
    alignas(64) float C_buf[M * N * B]; // C[batch][row][col]
    alignas(64) uint32_t C_words[M * N * B];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    // A/B occupy M*K*B and N*K*B packed f16 over their VGPRs. Bulk-convert each
    // whole region to f32 once with F16C (one vector op per 16 halves) instead
    // of branchy per-element f16_to_f32, then the hoist is a pure f32
    // index-shuffle (f16 of word w lane l sub s -> flat (w*wf+l)*2+s).
    alignas(64) float A_f32[M * K * B];
    alignas(64) float B_f32[N * K * B];
    convert_f16_matrix_region(reads.a, A_f32, M * K * B);
    convert_f16_matrix_region(reads.b, B_f32, N * K * B);
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < B; ++b) {
      float *C_batch = &C_buf[b * M * N];
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_batch[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                       ? std::bit_cast<float>(const_acc)
                                       : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
        }
      // Dense MxKxN matmul, W-lane native SIMD FMA over N (N/W chunks per row).
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<float> c_row;
          c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<float> a_bcast(A_buf[row * K + k]);
            util::native<float> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row = util::stdx::fma(a_bcast, b_row, c_row);
          }
          c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
        }
    }
    // Publish only after every batch has consumed its inputs. This preserves
    // instruction-level snapshot semantics when dst overlaps a later batch's
    // source or accumulator registers.
    for (uint32_t b = 0; b < B; ++b)
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          float fv = C_buf[(b * M + row) * N + col];
          writes.set_linear_word(out.reg * wf + out.lane, std::bit_cast<uint32_t>(fv));
          if (std::isnan(fv) || std::isinf(fv))
            has_nan_or_inf = true;
        }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f16", dst,
                          s0, s1, s2, M, N, K);
      });
    }
  }
}

/// Fast path for the bf16-input MFMA shapes (v_mfma_f32_*_bf16). Identical to
/// the f16 specialization except the bulk input convert is the bf16 zero-extend
/// (no F16C needed). Falls back to the generic exec_f32 without usable host
/// SIMD, under force-scalar, or with cbsz/blgp.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
void exec_f32_mfma_bf16_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                             uint32_t const_acc, uint32_t cbsz, uint32_t abid, uint32_t blgp) {
  constexpr uint32_t B = BATCH, in_bits = 16;
  if constexpr (!util::has_stdx_simd) {
    exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16, amdgpu::extract_bf16,
             const_acc, cbsz, abid, blgp);
    return;
  } else {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    if (util::force_scalar() || cbsz != 0 || blgp != 0 || !mma_f32_native_width_supported(N, W) ||
        cu.wf_size() != 64) {
      exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16, amdgpu::extract_bf16,
               const_acc, cbsz, abid, blgp);
      return;
    }
    const uint32_t wf = cu.wf_size();
    auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
    auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
    alignas(64) float A_buf[M * K]; // A[row][k] (one batch block)
    alignas(64) float B_buf[K * N]; // B[k][col] (one batch block)
    static_assert(M * N * B * sizeof(float) <= 8 * 1024,
                  "specialized MFMA result staging exceeds the stack budget");
    alignas(64) float C_buf[M * N * B]; // C[batch][row][col]
    alignas(64) uint32_t C_words[M * N * B];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    // Bulk-convert the packed bf16 regions to f32 once (zero-extend + shift),
    // then the hoist is a pure f32 index-shuffle.
    alignas(64) float A_f32[M * K * B];
    alignas(64) float B_f32[N * K * B];
    convert_bf16_matrix_region(reads.a, A_f32, M * K * B);
    convert_bf16_matrix_region(reads.b, B_f32, N * K * B);
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < B; ++b) {
      float *C_batch = &C_buf[b * M * N];
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_batch[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                       ? std::bit_cast<float>(const_acc)
                                       : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
        }
      // Dense MxKxN matmul, W-lane native SIMD FMA over N (N/W chunks per row).
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<float> c_row;
          c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<float> a_bcast(A_buf[row * K + k]);
            util::native<float> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row = util::stdx::fma(a_bcast, b_row, c_row);
          }
          c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
        }
    }
    // Publish only after every batch has consumed its inputs.
    for (uint32_t b = 0; b < B; ++b)
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          float fv = C_buf[(b * M + row) * N + col];
          writes.set_linear_word(out.reg * wf + out.lane, std::bit_cast<uint32_t>(fv));
          if (std::isnan(fv) || std::isinf(fv))
            has_nan_or_inf = true;
        }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_bf16",
                          dst, s0, s1, s2, M, N, K);
      });
    }
  }
}

/// Fast path for the dense (B=1) fp8/bf8-input MFMA shapes
/// (v_mfma_f32_{16x16x32,32x32x16}_{fp8,bf8}_{fp8,bf8}). Identical to the
/// f16/bf16 specializations except the bulk input convert is the f8 LUT gather
/// (bit-exact with extract_fp8/extract_bf8 by construction); A/B formats are
/// compile-time selected. Falls back to the generic exec_f32 without AVX-512 /
/// under force-scalar / with cbsz|blgp.
template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
void exec_f32_mfma_f8_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                           uint32_t const_acc, uint32_t cbsz, uint32_t abid, uint32_t blgp) {
  constexpr uint32_t B = 1, in_bits = 8;
  static_assert(N % 16 == 0, "specialized f8 MFMA assumes N is a multiple of the zmm width");
  constexpr auto ea = f8_extract_fn<A_FP8, FNUZ>();
  constexpr auto eb = f8_extract_fn<B_FP8, FNUZ>();
  if constexpr (!util::has_stdx_simd) {
    exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, ea, eb, const_acc, cbsz, abid, blgp);
    return;
  } else {
    if (util::force_scalar() || cbsz != 0 || blgp != 0 || util::native<float>::size() != 16 ||
        cu.wf_size() != 64) {
      exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, ea, eb, const_acc, cbsz, abid, blgp);
      return;
    }
    constexpr uint32_t W = 16; // guaranteed by the native<float>::size()==16 guard above
    const uint32_t wf = cu.wf_size();
    auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
    auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
    alignas(64) float A_buf[M * K]; // A[row][k]
    alignas(64) float B_buf[K * N]; // B[k][col]
    alignas(64) float C_buf[M * N]; // C[row][col]
    alignas(64) uint32_t C_words[M * N];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    // Bulk-convert the packed f8 regions to f32 once through the LUTs, then the
    // hoist is a pure f32 index-shuffle (byte of word w lane l sub s ->
    // (w*wf+l)*4+s).
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    convert_f8_matrix_region<A_FP8, FNUZ>(reads.a, A_f32, M * K);
    convert_f8_matrix_region<B_FP8, FNUZ>(reads.b, B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, 0);
        C_buf[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                   ? std::bit_cast<float>(const_acc)
                                   : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = input_loc(M, K, B, row, k, 0, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = input_loc(N, K, B, col, k, 0, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
      }
    // Dense MxKxN matmul, W-lane (zmm) stdx FMA over N (N/W chunks per row).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Scatter directly back to VGPRs (no Result staging vector).
    bool has_nan_or_inf = false;
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, 0);
        float fv = C_buf[row * N + col];
        writes.set_linear_word(out.reg * wf + out.lane, std::bit_cast<uint32_t>(fv));
        if (std::isnan(fv) || std::isinf(fv))
          has_nan_or_inf = true;
      }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f8", dst,
                          s0, s1, s2, M, N, K);
      });
    }
  }
}

/// Fast path for the dense i8-input MFMA shapes (v_mfma_i32_*_i8), including
/// the batched (BATCH>1) variants. Same structure as the f16 specialization
/// but integer: the packed i8 inputs are bulk sign-extended to int32 once,
/// the hoist is a pure i32 index-shuffle, and the matmul runs as
/// constexpr-unrolled zmm-wide int32 MACs. There is no clamp on the i8 MFMA
/// path: both scalar and SIMD accumulate (and wrap) in 32 bits, so they agree
/// bit for bit on every output (SIMD wraps in uint32 to keep the wrap
/// well-defined). Falls back to the generic exec_i32_i8 without AVX-512 /
/// under force-scalar.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
void exec_i32_mfma_i8_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                           uint32_t const_acc) {
  constexpr uint32_t B = BATCH, in_bits = 8;
  static_assert(N % 16 == 0, "specialized i8 MFMA assumes N is a multiple of the zmm width");
  if constexpr (!util::has_stdx_simd) {
    exec_i32_i8(cu, M, N, K, B, dst, s0, s1, s2, const_acc);
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16 || cu.wf_size() != 64) {
      exec_i32_i8(cu, M, N, K, B, dst, s0, s1, s2, const_acc);
      return;
    }
    constexpr uint32_t W = 16; // guaranteed by the native<float>::size()==16 guard above
    const uint32_t wf = cu.wf_size();
    auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
    auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
    // Accumulate in unsigned 32-bit (wrap is well-defined and identical mod
    // 2^32 to the intended signed wrap), so the SIMD path has no signed-
    // overflow UB.
    alignas(64) uint32_t A_buf[M * K]; // A[row][k] (sign-extended bits, one batch block)
    alignas(64) uint32_t B_buf[K * N]; // B[k][col] (sign-extended bits, one batch block)
    static_assert(M * N * B * sizeof(uint32_t) <= 8 * 1024,
                  "specialized MFMA result staging exceeds the stack budget");
    alignas(64) uint32_t C_buf[M * N * B]; // C[batch][row][col]
    alignas(64) uint32_t C_words[M * N * B];
    if (reads.acc)
      copy_matrix_region_words(*reads.acc, C_words);
    // Bulk sign-extend the packed i8 regions to int32 once, then the hoist is
    // a pure i32 index-shuffle (byte of word w lane l sub s -> (w*wf+l)*4+s).
    alignas(64) int32_t A_i32[M * K * B];
    alignas(64) int32_t B_i32[N * K * B];
    convert_i8_matrix_region(reads.a, A_i32, M * K * B);
    convert_i8_matrix_region(reads.b, B_i32, N * K * B);
    for (uint32_t b = 0; b < B; ++b) {
      uint32_t *C_batch = &C_buf[b * M * N];
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_batch[row * N + col] =
              (const_acc != ACC_FROM_VGPR) ? const_acc : C_words[out.reg * wf + out.lane];
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          A_buf[row * K + k] = A_i32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          B_buf[k * N + col] = B_i32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
        }
      // Dense MxKxN matmul, W-lane (zmm) stdx u32 MAC over N (N/W chunks per
      // row); unsigned wrap matches the scalar signed accumulation mod 2^32.
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<uint32_t> c_row;
          c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<uint32_t> a_bcast(A_buf[row * K + k]);
            util::native<uint32_t> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row += a_bcast * b_row;
          }
          c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
        }
    }
    // Publish only after every batch has consumed its inputs.
    for (uint32_t b = 0; b < B; ++b)
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          writes.set_linear_word(out.reg * wf + out.lane, C_buf[(b * M + row) * N + col]);
        }
  }
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_
