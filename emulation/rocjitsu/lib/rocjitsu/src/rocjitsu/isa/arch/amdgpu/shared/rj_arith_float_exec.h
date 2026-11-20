// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/binary_fp_status.h"
#include "rocjitsu/isa/arch/amdgpu/shared/alu_exceptions.h"
#include "rocjitsu/isa/arch/amdgpu/shared/conversion.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu::arith {

// Source observations follow EXEC even when DPP suppresses destination lanes.
// Evaluation and writes use EXEC intersected with the VGPR write mask.

using PseudoScalarFunction = int (*)(uint64_t, uint64_t, uint32_t *, uint32_t, uint32_t *);

// Execute once regardless of EXEC, reading the source before committing the
// complete scalar destination. Rejected calls preserve destination and state.
inline void execute_pseudo_scalar(Wavefront &wave, PseudoScalarFunction function, uint64_t flags,
                                  uint64_t modifiers, const Operand &destination,
                                  const Operand &source, bool half) {
  if (half) {
    const unsigned denorm = wave.fp_denorm_mode_f16_f64();
    if (!(denorm & 1))
      flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
    if (!(denorm & 2))
      flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
    if (wave.fp16_ovfl())
      flags |= RJ_ARITH_FP16_OVFL;
  }
  RegisterAccess registers(wave);
  const uint32_t input = registers.read_scalar(source);
  if (wave.instruction_execution_failed())
    return;
  uint32_t result = 0, causes = 0;
  if (function(flags, modifiers, &result, input, &causes) != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  registers.write_scalar(destination, result);
  wave.set_trapsts(wave.trapsts() | causes);
  wave.raise_alu_causes(causes);
}

// Translate guest FP32 rounding, denormal, IEEE and architecture-specific clamp controls.
inline uint64_t float32_flags(const Wavefront &wave, uint64_t flags, bool rdna3) {
  flags |= uint64_t(wave.fp_round_mode_f32()) << 24;
  if (rdna3 && !wave.dx10_clamp())
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  const uint32_t denorm = wave.fp_denorm_mode_f32();
  if (!(denorm & 1))
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (!(denorm & 2))
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  if (wave.ieee_mode())
    flags |= RJ_ARITH_FP_IEEE_MODE;
  return flags;
}

// Translate the shared FP16/FP64 rounding, denormal, IEEE and clamp controls.
inline uint64_t float16_f64_flags(const Wavefront &wave, uint64_t flags, bool rdna3) {
  flags |= uint64_t(wave.fp_round_mode_f16_f64()) << 24;
  const uint32_t denorm = wave.fp_denorm_mode_f16_f64();
  if (!(denorm & 1))
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  if (!(denorm & 2))
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  if (wave.ieee_mode())
    flags |= RJ_ARITH_FP_IEEE_MODE;
  if (rdna3 && !wave.dx10_clamp())
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  return flags;
}

// Add the FP16-only overflow control to the shared FP16/FP64 MODE fields.
inline uint64_t float16_flags(const Wavefront &wave, uint64_t flags, bool rdna3) {
  flags = float16_f64_flags(wave, flags, rdna3);
  return wave.fp16_ovfl() ? flags | RJ_ARITH_FP16_OVFL : flags;
}

using FloatUnary32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                     const uint32_t *const *, uint32_t *);
using FloatUnary64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                     const uint32_t *const *, uint32_t *);

// Convert selected FP16/integer halves, FP32/integer words and FP64 pairs.
// Pair selectors and byte masks describe layouts; flags describe FP formats.
// Rejected conversions commit neither destination words nor optional state.
inline void execute_float_conversion(Wavefront &wave, FloatUnary32Function wave32,
                                     FloatUnary64Function wave64, uint64_t flags,
                                     uint64_t modifiers, const Operand &destination,
                                     const Operand &source, bool from_double, bool to_double,
                                     uint8_t source_bytes = 0xF, uint8_t destination_bytes = 0xF) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  RegisterAccess registers(wave);
  uint32_t causes = 0;
  const auto convert = [&](const uint32_t *const *a) {
    const auto invoke = [&](uint32_t *const *d, uint64_t write_mask) {
      uint32_t *state = &causes;
      return wave.wf_size() == 32 ? wave32(flags, uint32_t(write_mask), modifiers, d, a, state)
                                  : wave64(flags, write_mask, modifiers, d, a, state);
    };
    return to_double ? write_vector64_result(wave, registers, destination, exec_mask, invoke)
                     : write_vector32_result(wave, registers, destination, exec_mask, invoke,
                                             /*allow_direct=*/destination_bytes == 0xF,
                                             destination_bytes);
  };
  bool success;
  if (from_double) {
    VectorInput64 a(registers, wave, source, wave.exec());
    success = convert(a.data.data());
  } else {
    VectorInput32 a(registers, wave, source, wave.exec(), source_bytes);
    success = convert(&a.data);
  }
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Execute with guest FP modes and routed operands. Compatibility preserves the
// legacy RCP_IFLAG/SQRT status from all EXEC lanes, including suppressed writes.
// Rejected calls commit neither results nor state.
// Exponent extraction returns an integer and ignores guest arithmetic rounding.
inline void execute_float_unary32(Wavefront &wave, FloatUnary32Function wave32,
                                  FloatUnary64Function wave64, uint64_t flags, uint64_t modifiers,
                                  const Operand &destination, const Operand &source, bool rdna3,
                                  bool guest_rounding = false,
                                  bool compatibility_all_exec = false) {
  const bool all_exec = compatibility_all_exec &&
                        (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  const uint64_t exec_mask = all_exec ? wave.exec() : wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float32_flags(wave, flags, rdna3);
  if (!guest_rounding)
    flags &= ~RJ_ARITH_FP_ROUND_MASK;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source, wave.exec());
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, state);
      },
      /*allow_direct=*/(exec_mask & ~wave.vgpr_write_mask()) == 0);
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Execute FP64 unary operations on low/high VGPR pairs. Integral operations
// use their named rounding rule; FRACT subtraction and OMOD use guest rounding.
// FREXP exponent extraction writes one integer word and ignores floating CLAMP.
inline void execute_float_unary64(Wavefront &wave, FloatUnary32Function wave32,
                                  FloatUnary64Function wave64, uint64_t flags, uint64_t modifiers,
                                  const Operand &destination, const Operand &source, bool rdna3,
                                  bool integer_result = false) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float16_f64_flags(wave, flags, rdna3);
  if (integer_result)
    flags &= ~(RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_DX10_CLAMP_OFF);
  RegisterAccess registers(wave);
  VectorInput64 a(registers, wave, source, wave.exec());
  uint32_t causes = 0;
  const auto invoke = [&](uint32_t *const *d, uint64_t write_mask) {
    uint32_t *state = &causes;
    return wave.wf_size() == 32
               ? wave32(flags, uint32_t(write_mask), modifiers, d, a.data.data(), state)
               : wave64(flags, write_mask, modifiers, d, a.data.data(), state);
  };
  const bool success = integer_result
                           ? write_vector32_result(wave, registers, destination, exec_mask, invoke)
                           : write_vector64_result(wave, registers, destination, exec_mask, invoke);
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

using FloatBinary32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                      const uint32_t *const *, const uint32_t *const *, uint32_t *);
using FloatBinary64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                      const uint32_t *const *, const uint32_t *const *, uint32_t *);

// Execute a binary FP64 operation with guest denormal, IEEE and clamp policies.
// Guest rounding is independent of the default host rounding mode.
inline void execute_float_binary64(Wavefront &wave, FloatBinary32Function wave32,
                                   FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                                   const Operand &destination, const Operand &source,
                                   const Operand &rhs, bool rdna3) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float16_f64_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  VectorInput64 a(registers, wave, source, wave.exec());
  VectorInput64 b(registers, wave, rhs, wave.exec());
  uint32_t causes = 0;
  const bool success = write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, a.data.data(), b.data.data(),
                            state)
                   : wave64(flags, write_mask, modifiers, d, a.data.data(), b.data.data(), state);
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Consume an FP64 pair and one integer word, then commit an FP64 pair.
// Guest rounding and clamp policy are translated here, as for other FP64 operations.
inline void execute_float64_word(Wavefront &wave, FloatBinary32Function wave32,
                                 FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                                 const Operand &destination, const Operand &source,
                                 const Operand &integer, bool rdna3, bool ldexp) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float16_f64_flags(wave, flags, rdna3);
  // TRIG_PREOP uses a fixed table policy; LDEXP rounds and clamps by MODE.
  if (!ldexp)
    flags &= ~(RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_DX10_CLAMP_OFF);
  RegisterAccess registers(wave);
  VectorInput64 a(registers, wave, source, wave.exec());
  VectorInput32 b(registers, wave, integer, wave.exec());
  uint32_t causes = 0;
  const bool success = write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, a.data.data(), &b.data,
                            state)
                   : wave64(flags, write_mask, modifiers, d, a.data.data(), &b.data, state);
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Pack two converted values into a full destination word. Half sources read
// only their selected bytes; these fixed-rounding conversions ignore guest MODE.
// Commit vector results and optional state only after a successful rj-arith call.
inline void execute_float_pack(Wavefront &wave, FloatBinary32Function wave32,
                               FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                               const Operand &destination, const Operand &lhs, const Operand &rhs,
                               bool half_sources) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  RegisterAccess registers(wave);
  const uint8_t a_bytes = half_sources ? (modifiers & RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3) : 0xF;
  const uint8_t b_bytes = half_sources ? (modifiers & RJ_ARITH_ALU_HIGH_B ? 0xC : 0x3) : 0xF;
  VectorInput32 a(registers, wave, lhs, wave.exec(), a_bytes);
  VectorInput32 b(registers, wave, rhs, wave.exec(), b_bytes);
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, state);
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Execute a binary operation with guest FP modes and optional accumulated state.
// All sources are acquired before writes, including when destinations alias them.
// MUL keeps the legacy status classifier in Compatibility; the library owns
// explicit mode 3 reporting. Other instructions leave compatibility_mul false.
template <
    ::rocjitsu::rdna4::BinaryFpOperation Gfx12Status = ::rocjitsu::rdna4::BinaryFpOperation::None>
inline void execute_float_binary32(Wavefront &wave, FloatBinary32Function wave32,
                                   FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                                   const Operand &destination, const Operand &lhs,
                                   const Operand &rhs, bool rdna3, bool compatibility_mul = false) {
  const bool report_legacy_mul =
      compatibility_mul && (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  const uint64_t write_mask = wave.exec() & wave.vgpr_write_mask();
  // DPP suppresses destination writes, not the legacy MUL status calculation.
  const uint64_t exec_mask = report_legacy_mul ? wave.exec() : write_mask;
  if (!wave.exec())
    return;
  flags = float32_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  // DPP suppresses results, not ordinary EXEC-participating source reads.
  VectorInput32 a(registers, wave, lhs, wave.exec());
  VectorInput32 b(registers, wave, rhs, wave.exec());
  uint32_t causes = 0;
  if constexpr (Gfx12Status != ::rocjitsu::rdna4::BinaryFpOperation::None) {
    if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
      for (uint64_t active = write_mask; active; active &= active - 1) {
        const unsigned lane = std::countr_zero(active);
        causes |= ::rocjitsu::rdna4::binary_causes<Gfx12Status>(a.data[lane], b.data[lane],
                                                                modifiers, wave.fp_round_mode_f32(),
                                                                wave.fp_denorm_mode_f32());
      }
    }
  }
  if (report_legacy_mul) {
    // Match classify_mul_f32_vop3's OMOD policy, using the already acquired
    // operands so aliasing is safe and observers see no duplicate reads.
    const uint32_t omod =
        fp_mode::effective_omod(wave.cu().arch(), wave.fp_denorm_mode_f32(), wave.ieee_mode(),
                                (modifiers & RJ_ARITH_ALU_OMOD_MASK) / RJ_ARITH_ALU_OMOD_2);
    const float scale = omod == 1 ? 2.0f : omod == 2 ? 4.0f : omod == 3 ? 0.5f : 1.0f;
    const uint32_t suppressed =
        omod && wave.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 ? (1u << 4) | (1u << 5) : 0;
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
      if (!(exec_mask & (1ULL << lane)))
        continue;
      uint32_t x = a.data[lane], y = b.data[lane];
      if (modifiers & RJ_ARITH_ALU_ABS_A)
        x &= 0x7fffffffu;
      if (modifiers & RJ_ARITH_ALU_NEG_A)
        x ^= 0x80000000u;
      if (modifiers & RJ_ARITH_ALU_ABS_B)
        y &= 0x7fffffffu;
      if (modifiers & RJ_ARITH_ALU_NEG_B)
        y ^= 0x80000000u;
      causes |=
          classify_mul_f32(std::bit_cast<float>(x), std::bit_cast<float>(y), scale) & ~suppressed;
    }
  }
  const bool success =
      !wave.instruction_execution_failed() &&
      (!write_mask ||
       write_vector32_result(
           wave, registers, destination, write_mask, [&](uint32_t *const *d, uint64_t write_mask) {
             uint32_t *state = &causes;
             return wave.wf_size() == 32
                        ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, state)
                        : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, state);
           }));
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Evaluate A * B + C, including explicit accumulator/literal operands. Rejected
// calls commit neither vector results nor optional architectural state.
template <bool Dx9, typename Wave32Function, typename Wave64Function>
inline void execute_float_fma32(Wavefront &wave, Wave32Function wave32, Wave64Function wave64,
                                uint64_t flags, uint64_t modifiers, const Operand &destination,
                                const Operand &lhs, const Operand &rhs, const Operand &addend,
                                bool rdna3) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float32_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, wave.exec());
  VectorInput32 b(registers, wave, rhs, wave.exec());
  VectorInput32 c(registers, wave, addend, wave.exec());
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        if constexpr (Dx9) {
          return wave.wf_size() == 32
                     ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data)
                     : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data);
        } else {
          return wave.wf_size() == 32
                     ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data,
                              &causes)
                     : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data, &causes);
        }
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

using FloatTernaryWave32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *, uint32_t *);
using FloatTernaryWave64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                           const uint32_t *const *, const uint32_t *const *,
                                           const uint32_t *const *, uint32_t *);

// Mixed FMA consumes one word per source and writes an FP32 word or selected
// FP16 half. Inline FP32 constants selected as FP16 are narrowed and replicated.
// Guest input denormals follow each source format. All sources precede
// destination stores; optional state is committed only after a successful call.
inline void execute_mixed_fma(Wavefront &wave, FloatTernaryWave32Function wave32,
                              FloatTernaryWave64Function wave64, uint64_t flags, uint64_t modifiers,
                              const Operand &destination, const Operand &lhs, const Operand &rhs,
                              const Operand &third, unsigned narrow_sources,
                              uint8_t destination_bytes, bool rdna3) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
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
  PackedFloatInput a(registers, wave, lhs, wave.exec(), narrow_sources & 1, narrow_sources & 1,
                     false, source_bytes(0));
  PackedFloatInput b(registers, wave, rhs, wave.exec(), narrow_sources & 2, narrow_sources & 2,
                     false, source_bytes(1));
  PackedFloatInput c(registers, wave, third, wave.exec(), narrow_sources & 4, narrow_sources & 4,
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
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Convert A to a byte, insert it at the per-lane position B in the preserved
// word C, and write a full D word. These conversions ignore guest MODE.
// Snapshot every source before committing D or optional exception causes.
inline void execute_byte_pack(Wavefront &wave, FloatTernaryWave32Function wave32,
                              FloatTernaryWave64Function wave64, uint64_t flags, uint64_t modifiers,
                              const Operand &destination, const Operand &value,
                              const Operand &selector, const Operand &preserved) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, value, wave.exec());
  VectorInput32 b(registers, wave, selector, wave.exec());
  VectorInput32 c(registers, wave, preserved, wave.exec());
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data,
                            state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data, state);
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// DOT2 uses architecture-defined accumulation independent of guest MODE. DOT2ACC
// supplies old D as explicit C, so staged output retains the original addend.
inline void execute_dot2_f32(Wavefront &wave, FloatTernaryWave32Function wave32,
                             FloatTernaryWave64Function wave64, uint64_t flags, uint64_t modifiers,
                             const Operand &destination, const Operand &lhs, const Operand &rhs,
                             const Operand &addend, bool bf16, unsigned narrow_sources,
                             unsigned replicate_sources) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  RegisterAccess registers(wave);
  PackedFloatInput a(registers, wave, lhs, wave.exec(), narrow_sources & 1, replicate_sources & 1,
                     bf16);
  PackedFloatInput b(registers, wave, rhs, wave.exec(), narrow_sources & 2, replicate_sources & 2,
                     bf16);
  VectorInput32 c(registers, wave, addend, wave.exec());
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data,
                            state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data, state);
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// True16 DOT2 reads packed A/B and only the selected half of C. Partial write
// views preserve the other destination half without an architectural read of D.
inline void execute_dot2_f16(Wavefront &wave, FloatTernaryWave32Function wave32,
                             FloatTernaryWave64Function wave64, uint64_t flags, uint64_t modifiers,
                             const Operand &destination, const Operand &lhs, const Operand &rhs,
                             const Operand &addend, bool bf16, unsigned narrow_sources,
                             unsigned replicate_sources) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  // Fixed RNE/denormal rules are owned by rj-arith, not the guest MODE register.
  if (wave.fp16_ovfl())
    flags |= RJ_ARITH_FP16_OVFL;
  RegisterAccess registers(wave);
  PackedFloatInput a(registers, wave, lhs, wave.exec(), narrow_sources & 1, replicate_sources & 1,
                     bf16);
  PackedFloatInput b(registers, wave, rhs, wave.exec(), narrow_sources & 2, replicate_sources & 2,
                     bf16);
  // Inline accumulators replicate their formatted low half before OPSEL; raw
  // register and literal carriers retain their independently selected halves.
  const bool replicate_c = replicate_sources & 4;
  PackedFloatInput c(registers, wave, addend, wave.exec(), false, replicate_c, bf16,
                     replicate_c                       ? 0x3
                     : modifiers & RJ_ARITH_ALU_HIGH_C ? 0xC
                                                       : 0x3);
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
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Fixed-rounding cores can still use guest rounding for FRACT or output scaling.
// Routed VOP1 halves and explicit VOP3 selectors both
// preserve the destination's other half, including when source and destination alias.
inline void execute_float_unary16(Wavefront &wave, FloatUnary32Function wave32,
                                  FloatUnary64Function wave64, uint64_t flags, uint64_t modifiers,
                                  const Operand &destination, const Operand &source, bool rdna3,
                                  bool guest_rounding = false) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float16_flags(wave, flags, rdna3);
  if (!guest_rounding)
    flags &= ~RJ_ARITH_FP_ROUND_MASK;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source, wave.exec(),
                  modifiers & RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3);
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, state);
      },
      /*allow_direct=*/false, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0x3);
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

enum class HalfBinaryOperation { None, Add, Sub, Subrev, Mul };

// Two halves sum or multiply exactly in binary64. Classify the unrounded
// operation before NaN quieting, saturation and destination placement.
template <HalfBinaryOperation Op>
inline uint32_t gfx10_half_binary_causes(const Wavefront &wf, const uint32_t *a, const uint32_t *b,
                                         uint64_t exec, uint64_t flags, uint64_t mods,
                                         unsigned raw_omod) {
  using F = fp_format::F16;
  if ((flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_COMPATIBILITY)
    return 0;
  uint32_t causes = 0;
  const auto denorm = wf.fp_denorm_mode_f16_f64();
  const auto round_mode = wf.fp_round_mode_f16_f64();
  const auto omod =
      fp_mode::effective_f16_omod(wf.cu().arch(), denorm, wf.ieee_mode(), false, raw_omod);
  const conversion::FloatRounding rounding{conversion::mode_rounding(round_mode)};
  auto source = [&](uint32_t raw, uint64_t high, uint64_t absolute, uint64_t negate) {
    uint16_t x = raw >> ((mods & high) ? 16 : 0);
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (x & F::kMagnitude) < F::kMinNormal)
      x &= F::kSign;
    if (mods & absolute)
      x &= F::kMagnitude;
    if (mods & negate)
      x ^= F::kSign;
    return x;
  };
  for (; exec; exec &= exec - 1) {
    const unsigned lane = std::countr_zero(exec);
    uint16_t x = source(a[lane], RJ_ARITH_ALU_HIGH_A, RJ_ARITH_ALU_ABS_A, RJ_ARITH_ALU_NEG_A);
    uint16_t y = source(b[lane], RJ_ARITH_ALU_HIGH_B, RJ_ARITH_ALU_ABS_B, RJ_ARITH_ALU_NEG_B);
    if constexpr (Op == HalfBinaryOperation::Sub)
      y ^= F::kSign;
    if constexpr (Op == HalfBinaryOperation::Subrev)
      x ^= F::kSign;
    const unsigned mx = x & F::kMagnitude, my = y & F::kMagnitude;
    if (mx >= F::kInfinity || my >= F::kInfinity) {
      const bool snan =
          (mx > F::kInfinity && !(mx & F::kQuiet)) || (my > F::kInfinity && !(my & F::kQuiet));
      const bool invalid = Op == HalfBinaryOperation::Mul
                               ? (mx == 0 && my == F::kInfinity) || (my == 0 && mx == F::kInfinity)
                               : mx == F::kInfinity && my == F::kInfinity && ((x ^ y) & F::kSign);
      if (snan || invalid)
        causes |= RJ_ARITH_EXCEPTION_INVALID;
      continue;
    }
    if ((mx && mx < F::kMinNormal) || (my && my < F::kMinNormal))
      causes |= RJ_ARITH_EXCEPTION_INPUT_DENORM;
    const double lhs = util::f16_to_f32(x), rhs = util::f16_to_f32(y);
    const double exact = Op == HalfBinaryOperation::Mul ? lhs * rhs : lhs + rhs;
    const auto arithmetic = fp_mode::finish_arithmetic_f16(exact, round_mode, denorm, false);
    causes |= gfx10_f16_output_causes(exact, arithmetic, wf.fp16_ovfl(), omod);
    // IEEE 754 overflow uses the guest-rounded significand with an unbounded
    // exponent, before FP16_OVFL or CLAMP chooses the destination value.
    const uint64_t bits = std::bit_cast<uint64_t>(exact);
    const auto rounded = conversion::round_finite<fp_format::F64, F>(bits, rounding);
    if (rounded >= F::kInfinity)
      causes |= RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT;
    else if (std::fabs(exact) != util::f16_to_f32(static_cast<uint16_t>(rounded))) {
      causes |= RJ_ARITH_EXCEPTION_INEXACT;
      // A tiny inexact result can round up to the smallest encoded normal.
      // Detect tininess after rounding to full half precision, before producing a subnormal.
      if (std::fabs(exact) < 0x1p-14 &&
          conversion::round_finite<fp_format::F64, F>(bits, {rounding.direction, true}) == 0)
        causes |= RJ_ARITH_EXCEPTION_UNDERFLOW;
    }
  }
  return causes;
}

// Ordinary FP16 reads and writes selected halves. VOP2 operand wrappers already
// resolve their halves; VOP3 supplies explicit selectors. Rejected calls commit
// neither destination bytes nor optional state.
template <HalfBinaryOperation Op = HalfBinaryOperation::None>
inline void execute_float_binary16(Wavefront &wave, FloatBinary32Function wave32,
                                   FloatBinary64Function wave64, uint64_t flags, uint64_t modifiers,
                                   const Operand &destination, const Operand &lhs,
                                   const Operand &rhs, bool rdna3) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float16_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, wave.exec(), modifiers & RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3);
  VectorInput32 b(registers, wave, rhs, wave.exec(), modifiers & RJ_ARITH_ALU_HIGH_B ? 0xC : 0x3);
  uint32_t causes = 0;
  if constexpr (Op != HalfBinaryOperation::None)
    causes =
        gfx10_half_binary_causes<Op>(wave, a.data, b.data, exec_mask, flags, modifiers,
                                     (modifiers & RJ_ARITH_ALU_OMOD_MASK) / RJ_ARITH_ALU_OMOD_2);
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, state);
      },
      /*allow_direct=*/false, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0x3);
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// SDWA selects source words before arithmetic, then scales the rounded half,
// clamps it, and merges the result into the selected destination field.
template <HalfBinaryOperation Op = HalfBinaryOperation::None, typename Inst>
inline void execute_sdwa_float_binary16(Inst &inst, Wavefront &wave, FloatBinary32Function wave32,
                                        FloatBinary64Function wave64, const Operand &destination,
                                        const Operand &lhs, const Operand &rhs) {
  const auto exec = wave.exec();
  if (!exec)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, exec);
  VectorInput32 b(registers, wave, rhs, exec);
  if (wave.instruction_execution_failed())
    return;
  std::array<uint32_t, 64> results{};
  uint32_t *output = results.data();
  uint32_t causes = 0;
  const uint64_t flags = float16_flags(wave, wave.cu().arithmetic_flags(), true);
  if constexpr (Op != HalfBinaryOperation::None)
    causes = gfx10_half_binary_causes<Op>(wave, a.data, b.data, exec, flags, 0, inst.sdwa_omod_);
  const int status = wave.wf_size() == 32
                         ? wave32(flags, uint32_t(exec), 0, &output, &a.data, &b.data, &causes)
                         : wave64(flags, exec, 0, &output, &a.data, &b.data, &causes);
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
    if (exec & (uint64_t{1} << lane))
      sdwa::write_lane<sdwa::ResultFormat::F16, sdwa::OutputPolicy::ROUNDED>(
          inst, wave, destination, lane, results[lane]);
  }
  wave.set_trapsts(wave.trapsts() | causes);
  wave.raise_alu_causes(causes);
}

// Execute a three-input FP64 operation with supported guest controls and optional state.
// Acquire every source before stores, including aliased and delegated operands.
inline void execute_float_ternary64(Wavefront &wave, FloatTernaryWave32Function wave32,
                                    FloatTernaryWave64Function wave64, uint64_t flags,
                                    uint64_t modifiers, const Operand &destination,
                                    const Operand &lhs, const Operand &rhs, const Operand &third,
                                    bool rdna3) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float16_f64_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  VectorInput64 a(registers, wave, lhs, wave.exec());
  VectorInput64 b(registers, wave, rhs, wave.exec());
  VectorInput64 c(registers, wave, third, wave.exec());
  uint32_t causes = 0;
  const bool success = write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32 ? wave32(flags, uint32_t(write_mask), modifiers, d,
                                             a.data.data(), b.data.data(), c.data.data(), state)
                                    : wave64(flags, write_mask, modifiers, d, a.data.data(),
                                             b.data.data(), c.data.data(), state);
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Execute a three-input FP32 operation with guest controls and optional state.
// Acquire every source before stores, including aliased and delegated operands.
inline void execute_float_ternary32(Wavefront &wave, FloatTernaryWave32Function wave32,
                                    FloatTernaryWave64Function wave64, uint64_t flags,
                                    uint64_t modifiers, const Operand &destination,
                                    const Operand &lhs, const Operand &rhs, const Operand &third,
                                    bool rdna3) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float32_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, wave.exec());
  VectorInput32 b(registers, wave, rhs, wave.exec());
  VectorInput32 c(registers, wave, third, wave.exec());
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data,
                            state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data, state);
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Execute an FP16 operation with three explicit inputs. All selected source
// halves are acquired before committing destination bytes or optional state.
inline void execute_float_ternary16(Wavefront &wave, FloatTernaryWave32Function wave32,
                                    FloatTernaryWave64Function wave64, uint64_t flags,
                                    uint64_t modifiers, const Operand &destination,
                                    const Operand &lhs, const Operand &rhs, const Operand &third,
                                    bool rdna3) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float16_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, wave.exec(), modifiers & RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3);
  VectorInput32 b(registers, wave, rhs, wave.exec(), modifiers & RJ_ARITH_ALU_HIGH_B ? 0xC : 0x3);
  VectorInput32 c(registers, wave, third, wave.exec(), modifiers & RJ_ARITH_ALU_HIGH_C ? 0xC : 0x3);
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
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Packed binary operations acquire both inputs before stores; inline FP32 constants are
// narrowed into the low half. Rejected calls commit no results or state.
inline void execute_packed_binary16(Wavefront &wave, FloatBinary32Function wave32,
                                    FloatBinary64Function wave64, uint64_t flags,
                                    uint64_t modifiers, const Operand &destination,
                                    const Operand &lhs, const Operand &rhs, bool rdna3,
                                    unsigned narrow_sources) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float16_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  PackedFloatInput a(registers, wave, lhs, wave.exec(), narrow_sources & 1, false);
  PackedFloatInput b(registers, wave, rhs, wave.exec(), narrow_sources & 2, false);
  uint32_t causes = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        uint32_t *state = &causes;
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, state)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, state);
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

// Apply the guest FP16 controls; commit results and optional causes together.
// Source transformation masks contain one bit per operand, in A/B/C order.
template <bool Accumulate, typename Wave32Function, typename Wave64Function>
inline void execute_packed_fma16(Wavefront &wave, Wave32Function wave32, Wave64Function wave64,
                                 uint64_t flags, uint64_t modifiers, const Operand &destination,
                                 const Operand &lhs, const Operand &rhs, const Operand &addend,
                                 bool rdna3, unsigned narrow_sources, unsigned replicate_sources) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!wave.exec())
    return;
  flags = float16_flags(wave, flags, rdna3);
  RegisterAccess registers(wave);
  PackedFloatInput a(registers, wave, lhs, wave.exec(), narrow_sources & 1, replicate_sources & 1);
  PackedFloatInput b(registers, wave, rhs, wave.exec(), narrow_sources & 2, replicate_sources & 2);
  PackedFloatInput c(registers, wave, addend, wave.exec(), narrow_sources & 4,
                     replicate_sources & 4);
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
            return wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, state);
          }
          return wave64(flags, write_mask, modifiers, d, &a.data, &b.data, state);
        } else {
          if (wave.wf_size() == 32) {
            return wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data,
                          state);
          }
          return wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data, state);
        }
      });
  if (success) {
    wave.set_trapsts(wave.trapsts() | causes);
    wave.raise_alu_causes(causes);
  }
}

} // namespace rocjitsu::amdgpu::arith
