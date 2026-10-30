// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_FLOAT_EXEC_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_FLOAT_EXEC_H_

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/alu_exceptions.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/data_types.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu::arith {

// Translate guest denormal and IEEE controls without changing host FP controls.
inline uint64_t float32_flags(const Wavefront &wave, uint64_t flags) {
  const uint32_t denorm = wave.fp_denorm_mode_f32();
  if (!(denorm & 1))
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (!(denorm & 2))
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  if (wave.ieee_mode())
    flags |= RJ_ARITH_FP_IEEE_MODE;
  return flags;
}

// Evaluate A * B + C, including explicit accumulator/literal operands. Rejected
// calls commit neither vector results nor optional architectural state.
inline void execute_float_fma32(Wavefront &wave, uint64_t flags, uint64_t modifiers,
                                const Operand &destination, const Operand &lhs, const Operand &rhs,
                                const Operand &addend, bool rdna3, bool dx9,
                                bool report_fp_status) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  flags = float32_flags(wave, flags) | (uint64_t(wave.fp_round_mode_f32()) << 24);
  if (rdna3 && !wave.dx10_clamp())
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, exec_mask);
  VectorInput32 b(registers, wave, rhs, exec_mask);
  VectorInput32 c(registers, wave, addend, exec_mask);
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        if (dx9) {
          if (wave.wf_size() == 32) {
            const auto function =
                rdna3 ? rj_arith_v_fma_dx9_zero_f32_w32 : rj_arith_v_fma_dx9_zero_f32_w32_gfx12;
            return function(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data);
          }
          const auto function =
              rdna3 ? rj_arith_v_fma_dx9_zero_f32_w64 : rj_arith_v_fma_dx9_zero_f32_w64_gfx12;
          return function(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data);
        }
        uint32_t *state = &causes;
        if (wave.wf_size() == 32) {
          const auto function = rdna3 ? rj_arith_v_fma_f32_w32 : rj_arith_v_fma_f32_w32_gfx12;
          return function(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data,
                          state);
        }
        const auto function = rdna3 ? rj_arith_v_fma_f32_w64 : rj_arith_v_fma_f32_w64_gfx12;
        return function(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data, state);
      });
  if (success && report_fp_status && !dx9) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Packed inline constants arrive either as FP32 words or already narrowed
// FP16 bits. BF16 narrowing takes the upper 16 bits without rounding.
// Replication broadcasts the resulting low half into both components.
class PackedFloatInput {
public:
  PackedFloatInput(RegisterAccess &registers, Wavefront &wave, const Operand &operand,
                   uint64_t exec_mask, bool narrow, bool replicate, bool bf16 = false,
                   uint8_t source_bytes = 0xF)
      : source_(registers, wave, operand, exec_mask, narrow ? 0xF : source_bytes) {
    data = source_.data;
    if (!narrow && !replicate)
      return;
    for (unsigned lane = 0; lane < wave.wf_size(); ++lane) {
      uint32_t value = source_.data[lane];
      if (narrow)
        value = bf16 ? value >> 16 : util::f32_to_f16(std::bit_cast<float>(value));
      values_[lane] = replicate ? (value & 0xffffu) * 0x10001u : value;
    }
    data = values_.data();
  }

  const uint32_t *data;

private:
  VectorInput32 source_;
  std::array<uint32_t, 64> values_;
};

using FloatDot32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                   const uint32_t *const *, const uint32_t *const *,
                                   const uint32_t *const *, uint32_t *);
using FloatDot64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                   const uint32_t *const *, const uint32_t *const *,
                                   const uint32_t *const *, uint32_t *);

// Mixed FMA consumes one word per source and writes an FP32 word or selected
// FP16 half. Inline FP32 constants selected as FP16 are narrowed and replicated.
// Guest input denormals follow each source format. All sources precede
// destination stores; optional state is committed only after a successful call.
inline void execute_mixed_fma(Wavefront &wave, FloatDot32Function wave32, FloatDot64Function wave64,
                              uint64_t flags, uint64_t modifiers, const Operand &destination,
                              const Operand &lhs, const Operand &rhs, const Operand &third,
                              unsigned narrow_sources, uint8_t destination_bytes, bool rdna3,
                              bool report_fp_status) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  const unsigned half_sources = unsigned((modifiers >> 13) & 7);
  const unsigned denorm32 = wave.fp_denorm_mode_f32();
  const unsigned denorm16 = wave.fp_denorm_mode_f16_f64();
  const bool half_output = destination_bytes != 0xF;
  const unsigned rounding = half_output ? wave.fp_round_mode_f16_f64() : wave.fp_round_mode_f32();
  flags |= uint64_t(rounding) << 24;
  if (rdna3 && !wave.dx10_clamp())
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  if ((denorm32 ^ denorm16) & 1)
    flags |= denorm16 & 1 ? RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16
                          : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16;
  if (!(denorm32 & 1))
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (!((half_output ? denorm16 : denorm32) & 2))
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  if (wave.ieee_mode())
    flags |= RJ_ARITH_FP_IEEE_MODE;
  if (half_output && wave.fp16_ovfl())
    flags |= RJ_ARITH_FP16_OVFL;
  RegisterAccess registers(wave);
  const auto source_bytes = [&](unsigned source) -> uint8_t {
    if (!(half_sources & (1u << source)))
      return 0xF;
    return modifiers & (RJ_ARITH_ALU_HIGH_A << source) ? 0xC : 0x3;
  };
  PackedFloatInput a(registers, wave, lhs, exec_mask, narrow_sources & 1, narrow_sources & 1, false,
                     source_bytes(0));
  PackedFloatInput b(registers, wave, rhs, exec_mask, narrow_sources & 2, narrow_sources & 2, false,
                     source_bytes(1));
  PackedFloatInput c(registers, wave, third, exec_mask, narrow_sources & 4, narrow_sources & 4,
                     false, source_bytes(2));
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data,
                            state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data, state);
      },
      !half_output, destination_bytes);
  if (success && report_fp_status) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Translate FP16 rounding, denormal, overflow and NaN clamp controls.
inline uint64_t float16_flags(const Wavefront &wave, uint64_t flags, bool rdna3) {
  const unsigned denorm = wave.fp_denorm_mode_f16_f64();
  flags |= uint64_t(wave.fp_round_mode_f16_f64()) << 24;
  if (!(denorm & 1))
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (!(denorm & 2))
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  if (wave.ieee_mode())
    flags |= RJ_ARITH_FP_IEEE_MODE;
  if (wave.fp16_ovfl())
    flags |= RJ_ARITH_FP16_OVFL;
  if (rdna3 && !wave.dx10_clamp())
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  return flags;
}

// Execute a three-input FP64 operation with supported guest controls and optional state.
// Acquire every source before stores, including aliased and delegated operands.
inline void execute_float_ternary64(Wavefront &wave, FloatDot32Function wave32,
                                    FloatDot64Function wave64, uint64_t flags, uint64_t modifiers,
                                    const Operand &destination, const Operand &lhs,
                                    const Operand &rhs, const Operand &third, bool rdna3,
                                    bool report_fp_status) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  const uint32_t denorm = wave.fp_denorm_mode_f16_f64();
  if (!(denorm & 1))
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (!(denorm & 2))
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  if (wave.ieee_mode())
    flags |= RJ_ARITH_FP_IEEE_MODE;
  if (rdna3 && !wave.dx10_clamp())
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  RegisterAccess registers(wave);
  VectorInput64 a(registers, wave, lhs, exec_mask);
  VectorInput64 b(registers, wave, rhs, exec_mask);
  VectorInput64 c(registers, wave, third, exec_mask);
  uint32_t causes = 0;
  const bool success = write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32 ? wave32(flags, uint32_t(write_mask), modifiers, d,
                                             a.data.data(), b.data.data(), c.data.data(), state)
                                    : wave64(flags, write_mask, modifiers, d, a.data.data(),
                                             b.data.data(), c.data.data(), state);
      });
  if (success && report_fp_status) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Execute an FP16 operation with three explicit inputs. All selected source
// halves are acquired before committing destination bytes or optional state.
inline void execute_float_ternary16(Wavefront &wave, FloatDot32Function wave32,
                                    FloatDot64Function wave64, uint64_t flags, uint64_t modifiers,
                                    const Operand &destination, const Operand &lhs,
                                    const Operand &rhs, const Operand &third, bool rdna3,
                                    bool report_fp_status) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  flags = float16_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, exec_mask, modifiers & RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3);
  VectorInput32 b(registers, wave, rhs, exec_mask, modifiers & RJ_ARITH_ALU_HIGH_B ? 0xC : 0x3);
  VectorInput32 c(registers, wave, third, exec_mask, modifiers & RJ_ARITH_ALU_HIGH_C ? 0xC : 0x3);
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data,
                            state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data, state);
      },
      /*allow_direct=*/false, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0x3);
  if (success && report_fp_status) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Apply the guest FP16 controls; commit results and optional causes together.
// Source transformation masks contain one bit per operand, in A/B/C order.
template <bool Accumulate>
inline void execute_packed_fma16(Wavefront &wave, uint64_t flags, uint64_t modifiers,
                                 const Operand &destination, const Operand &lhs, const Operand &rhs,
                                 const Operand &addend, bool rdna3, unsigned narrow_sources,
                                 unsigned replicate_sources, bool report_fp_status) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  flags = float16_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  PackedFloatInput a(registers, wave, lhs, exec_mask, narrow_sources & 1, replicate_sources & 1);
  PackedFloatInput b(registers, wave, rhs, exec_mask, narrow_sources & 2, replicate_sources & 2);
  PackedFloatInput c(registers, wave, addend, exec_mask, narrow_sources & 4, replicate_sources & 4);
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        if constexpr (Accumulate) {
          // Staged output must start with the original accumulator. Direct
          // unobserved output already points at that same register storage.
          if (d[0] != c.data)
            std::copy_n(c.data, wave.wf_size(), d[0]);
          if (wave.wf_size() == 32) {
            auto function = rdna3 ? rj_arith_v_pk_fmac_f16_w32 : rj_arith_v_pk_fmac_f16_w32_gfx12;
            return function(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, state);
          }
          auto function = rdna3 ? rj_arith_v_pk_fmac_f16_w64 : rj_arith_v_pk_fmac_f16_w64_gfx12;
          return function(flags, write_mask, modifiers, d, &a.data, &b.data, state);
        } else {
          if (wave.wf_size() == 32) {
            auto function = rdna3 ? rj_arith_v_pk_fma_f16_w32 : rj_arith_v_pk_fma_f16_w32_gfx12;
            return function(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data,
                            state);
          }
          auto function = rdna3 ? rj_arith_v_pk_fma_f16_w64 : rj_arith_v_pk_fma_f16_w64_gfx12;
          return function(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data, state);
        }
      });
  if (success && report_fp_status) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

} // namespace rocjitsu::amdgpu::arith

#endif // ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_FLOAT_EXEC_H_
