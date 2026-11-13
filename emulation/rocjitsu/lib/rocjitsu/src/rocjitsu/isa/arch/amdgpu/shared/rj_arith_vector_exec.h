// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/simd.h"

#include <array>
#include <bit>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu::arith {

// A normal VGPR uses observed storage directly. Uniform, subword or delegated
// operands use scratch storage; their existing implementation owns routing.
class VectorInput32 {
public:
  VectorInput32(RegisterAccess &registers, Wavefront &wave, const Operand &operand,
                uint64_t exec_mask, uint8_t byte_mask = 0xF) {
    // A uniform fallback would otherwise read its SGPR even with no consuming lanes.
    if (!exec_mask) {
      values_.fill(0);
      data = values_.data();
      return;
    }
    if (operand.simd_capable() && byte_mask == 0xF) {
      if (auto region = registers.read_operand_region(operand, exec_mask)) {
        auto lanes = region->lanes(0);
        if (lanes.succeeded()) {
          // The span refers to CU storage (or a static zero buffer), not the view.
          data = lanes.value().data();
        } else {
          values_.fill(0);
          data = values_.data();
        }
        return;
      }
    }
    if (operand.simd_capable()) {
      auto view = registers.read_operand(operand, exec_mask, byte_mask);
      for (uint32_t lane = 0; lane < wave.wf_size(); lane += util::native_width_v<uint32_t>)
        view.load_native<uint32_t>(lane).copy_to(values_.data() + lane,
                                                 util::stdx::element_aligned);
    } else {
      values_.fill(0);
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane)
        if (exec_mask & (uint64_t{1} << lane))
          values_[lane] = registers.read_lane(operand, lane);
    }
    data = values_.data();
  }

  // The packed integer API accepts half selectors and clamp. Materialize a
  // signed source before the call so two copies of one selected half can
  // receive different signs. Other operands can still use their direct views.
  void apply_packed_integer_signs(uint32_t lanes, unsigned index, uint32_t negations,
                                  uint64_t &modifiers) {
    const uint32_t signs = (negations >> index) & 0x00010001u;
    if (!signs)
      return;
    const uint64_t low_high = RJ_ARITH_PK_LO_A_HIGH << index;
    const uint64_t high_low = RJ_ARITH_PK_HI_A_LOW << index;
    for (unsigned lane = 0; lane < lanes; ++lane)
      values_[lane] = select_packed_halves(data[lane], (modifiers & low_high) != 0,
                                           (modifiers & high_low) == 0, 0, signs);
    data = values_.data();
    modifiers &= ~(low_high | high_low);
  }

  const uint32_t *data;

private:
  std::array<uint32_t, 64> values_;
};

// rj-arith represents each 64-bit lane as low/high words in two separate VGPRs.
class VectorInput64 {
public:
  VectorInput64(RegisterAccess &registers, Wavefront &wave, const Operand &operand,
                uint64_t exec_mask, bool packed_words = false,
                ScalarPairMode scalar_mode = ScalarPairMode::Preserve) {
    if (!exec_mask) {
      values_[0].fill(0);
      values_[1].fill(0);
      data = {values_[0].data(), values_[1].data()};
      return;
    }
    region_ = registers.read_operand_region(operand, exec_mask, 2);
    if (region_) {
      auto lo = region_->lanes(0);
      auto hi = region_->lanes(1);
      if (lo.succeeded() && hi.succeeded()) {
        data = {lo.value().data(), hi.value().data()};
      } else {
        values_[0].fill(0);
        values_[1].fill(0);
        data = {values_[0].data(), values_[1].data()};
      }
      return;
    }
    if (packed_words) {
      values_[0].fill(0);
      values_[1].fill(0);
      // Packed inline constants replicate their FP32 bits, rather than widening
      // to a numeric 64-bit value. Delegated operands retain pair32 routing.
      if (operand.const_value()) {
        const auto value = registers.read_lane_pair32(operand, 0, scalar_mode);
        values_[0].fill(value.lo);
        values_[1].fill(value.hi);
      } else {
        for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
          if ((exec_mask >> lane) & 1) {
            const auto value = registers.read_lane_pair32(operand, lane, scalar_mode);
            values_[0][lane] = value.lo;
            values_[1][lane] = value.hi;
          }
        }
      }
      data = {values_[0].data(), values_[1].data()};
      return;
    }
    // A numeric 64-bit operand widens its scalar constant once; the packed
    // pair32 accessor instead repeats a 32-bit inline constant in both words.
    if (operand.const_value()) {
      const uint64_t value = registers.read_lane64(operand, 0);
      values_[0].fill(uint32_t(value));
      values_[1].fill(uint32_t(value >> 32));
      data = {values_[0].data(), values_[1].data()};
      return;
    }
    if (operand.simd_capable()) {
      // Numeric scalar sources (including SCC) are widened, not replicated pairs.
      auto view = registers.read_operand64(operand, exec_mask);
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        const uint64_t value = view.lane(lane);
        values_[0][lane] = uint32_t(value);
        values_[1][lane] = uint32_t(value >> 32);
      }
    } else {
      values_[0].fill(0);
      values_[1].fill(0);
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        if (!(exec_mask & (uint64_t{1} << lane)))
          continue;
        const uint64_t value = registers.read_lane64(operand, lane);
        values_[0][lane] = uint32_t(value);
        values_[1][lane] = uint32_t(value >> 32);
      }
    }
    data = {values_[0].data(), values_[1].data()};
  }

  std::array<const uint32_t *, 2> data;

private:
  std::optional<RegisterAccess::VgprReadRegion> region_;
  std::array<std::array<uint32_t, 64>, 2> values_;
};

// Packed inline constants arrive either as FP32 words or already narrowed
// FP16/BF16 bits.
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
        value = bf16 ? util::f32_to_bf16(std::bit_cast<float>(value))
                     : util::f32_to_f16(std::bit_cast<float>(value));
      values_[lane] = replicate ? (value & 0xffffu) * 0x10001u : value;
    }
    data = values_.data();
  }

  const uint32_t *data;

private:
  VectorInput32 source_;
  std::array<uint32_t, 64> values_;
};

// Direct stores are safe without observers: rj-arith preserves storage on rejection.
// Other destinations stage the result before acquiring an observed write view.
// Returns true on success, or reports an instruction error and returns false.
template <typename Invoke>
inline bool write_vector32_result(Wavefront &wave, RegisterAccess &registers,
                                  const Operand &destination, uint64_t exec_mask, Invoke invoke,
                                  bool allow_direct = true, uint8_t byte_mask = 0xF) {
  if (wave.instruction_execution_failed())
    return false;
  if (allow_direct && byte_mask == 0xF) {
    if (auto status = registers.try_bulk_write_operand(destination, exec_mask, invoke)) {
      if (*status != RJ_ARITH_SUCCESS)
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return *status == RJ_ARITH_SUCCESS;
    }
  }
  std::array<uint32_t, 64> result{};
  uint32_t *d = result.data();
  const int status = invoke(&d, exec_mask);
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return false;
  }
  if (destination.simd_capable()) {
    auto output = registers.write_operand(destination, exec_mask, byte_mask);
    for (uint32_t lane = 0; lane < wave.wf_size(); lane += util::native_width_v<uint32_t>)
      if (output.store_native(lane, util::load<uint32_t>(result.data() + lane), exec_mask >> lane)
              .failed())
        return false;
  } else {
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane)
      if (exec_mask & (uint64_t{1} << lane))
        registers.write_lane(destination, lane, result[lane]);
  }
  return !wave.instruction_execution_failed();
}

// Select raw words or halves using a wave-wide condition mask. Acquire both
// sources before writes, and preserve the unused destination half for B16.
template <bool Half, typename Inst, typename Wave32, typename Wave64>
inline void execute_conditional(Inst &inst, Wavefront &wave, Wave32 wave32, Wave64 wave64,
                                uint64_t flags, uint64_t modifiers, const Operand &destination,
                                const Operand &lhs, const Operand &rhs, uint64_t condition) {
  // DPP filters destinations, not ordinary source reads on active lanes.
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, lhs, exec_mask & ~condition,
                  Half ? (modifiers & RJ_ARITH_ALU_HIGH_A ? 0xc : 0x3) : 0xf);
  VectorInput32 b(registers, wave, rhs, exec_mask & condition,
                  Half ? (modifiers & RJ_ARITH_ALU_HIGH_B ? 0xc : 0x3) : 0xf);
  auto invoke = [&](uint32_t *const *d, uint64_t write_mask) {
    return wave.wf_size() == 32
               ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data,
                        uint32_t(condition))
               : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, condition);
  };
  if constexpr (requires { inst.sdwa_dst_sel_; }) {
    if (inst.inst_.src0 == SRC_SDWA) {
      uint32_t result[64]{};
      uint32_t *d = result;
      if (invoke(&d, exec_mask) != RJ_ARITH_SUCCESS) {
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
        return;
      }
      if (!wave.instruction_execution_failed()) {
        for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
          if ((exec_mask >> lane) & 1)
            sdwa::write_lane<sdwa::ResultFormat::NONE>(inst, wave, destination, lane, result[lane]);
        }
      }
      return;
    }
  }
  write_vector32_result(wave, registers, destination, exec_mask & wave.vgpr_write_mask(), invoke,
                        /*allow_direct=*/!Half,
                        Half ? (modifiers & RJ_ARITH_ALU_HIGH_D ? 0xc : 0x3) : 0xf);
}

// Consume the selected input halves and commit only the selected destination
// half. VOP1's packed operands instead supply/accept a resolved low-half value.
// SourceCount determines whether source_b and source_c must be provided.
template <unsigned SourceCount, typename Wave32, typename Wave64>
inline void execute_integer16(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                              uint64_t modifiers, const Operand &destination,
                              const Operand &source_a, const Operand *source_b = nullptr,
                              const Operand *source_c = nullptr) {
  static_assert(SourceCount >= 1 && SourceCount <= 3);
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source_a, exec_mask,
                  modifiers & RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3);
  std::optional<VectorInput32> b, c;
  if constexpr (SourceCount >= 2)
    b.emplace(registers, wave, *source_b, exec_mask, modifiers & RJ_ARITH_ALU_HIGH_B ? 0xC : 0x3);
  if constexpr (SourceCount == 3)
    c.emplace(registers, wave, *source_c, exec_mask, modifiers & RJ_ARITH_ALU_HIGH_C ? 0xC : 0x3);
  write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        if constexpr (SourceCount == 1)
          return wave.wf_size() == 32 ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data)
                                      : wave64(flags, write_mask, modifiers, d, &a.data);
        else if constexpr (SourceCount == 2)
          return wave.wf_size() == 32
                     ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b->data)
                     : wave64(flags, write_mask, modifiers, d, &a.data, &b->data);
        else
          return wave.wf_size() == 32
                     ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b->data,
                              &c->data)
                     : wave64(flags, write_mask, modifiers, d, &a.data, &b->data, &c->data);
      },
      /*allow_direct=*/false, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0x3);
}

// A two-word result uses one logical 64-bit write, preserving observation and
// invalid-pair behavior. Direct stores require a valid, unobserved VGPR pair.
template <typename Invoke>
inline bool write_vector64_result(Wavefront &wave, RegisterAccess &registers,
                                  const Operand &destination, uint64_t exec_mask, Invoke invoke,
                                  bool allow_direct = true) {
  if (wave.instruction_execution_failed())
    return false;
  if (allow_direct) {
    if (auto status = registers.try_bulk_write_operand<2>(destination, exec_mask, invoke)) {
      if (*status != RJ_ARITH_SUCCESS)
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return *status == RJ_ARITH_SUCCESS;
    }
  }
  std::array<std::array<uint32_t, 64>, 2> result{};
  uint32_t *d[] = {result[0].data(), result[1].data()};
  if (invoke(d, exec_mask) != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return false;
  }
  if (destination.simd_capable()) {
    auto output = registers.write_operand_pair32(destination, exec_mask);
    for (uint32_t lane = 0; lane < wave.wf_size(); lane += util::native_width_v<uint32_t>)
      if (output
              .store_native_pair(lane, util::load<uint32_t>(result[0].data() + lane),
                                 util::load<uint32_t>(result[1].data() + lane), exec_mask >> lane)
              .failed())
        return false;
  } else {
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane)
      if (exec_mask & (uint64_t{1} << lane))
        registers.write_lane64(destination, lane,
                               result[0][lane] | (uint64_t(result[1][lane]) << 32));
  }
  return !wave.instruction_execution_failed();
}
// Copy numeric register pairs; preserve scalar widening and delegated DPP sources.
// Packed MOV selects full words from two pairs, not replicated inline constants.
template <bool Packed>
inline void execute_pair_move(Wavefront &wave, uint64_t modifiers, const Operand &destination,
                              const Operand &first, const Operand *second) {
  if (!wave.exec()) {
    return;
  }
  RegisterAccess registers(wave);
  VectorInput64 a(registers, wave, first, wave.exec());
  std::optional<VectorInput64> b;
  if constexpr (Packed) {
    b.emplace(registers, wave, *second, wave.exec());
  }
  const uint64_t flags = wave.cu().arithmetic_flags();
  write_vector64_result(
      wave, registers, destination, wave.exec() & wave.vgpr_write_mask(),
      [&](uint32_t *const *d, uint64_t exec_mask) {
        if constexpr (Packed) {
          return wave.wf_size() == 32
                     ? rj_arith_v_pk_mov_b32_w32(flags, uint32_t(exec_mask), modifiers, d,
                                                 a.data.data(), b->data.data())
                     : rj_arith_v_pk_mov_b32_w64(flags, exec_mask, modifiers, d, a.data.data(),
                                                 b->data.data());
        } else {
          return wave.wf_size() == 32
                     ? rj_arith_v_mov_b64_w32(flags, uint32_t(exec_mask), modifiers, d,
                                              a.data.data())
                     : rj_arith_v_mov_b64_w64(flags, exec_mask, modifiers, d, a.data.data());
        }
      });
}

// Read the 32-bit count and both value words before committing an aliased pair.
template <typename Wave32, typename Wave64>
inline void execute_shift64(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                            uint64_t modifiers, const Operand &destination, const Operand &source_a,
                            const Operand &source_b) {
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source_a, exec_mask);
  VectorInput64 b(registers, wave, source_b, exec_mask);
  write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, b.data.data())
                   : wave64(flags, write_mask, modifiers, d, &a.data, b.data.data());
      });
}

// Multiply two words and add a pair; commit the scalar side result only after
// both VGPR words succeed. Scalar carry includes all active EXEC lanes.
template <typename Wave32, typename Wave64, typename Commit>
inline void execute_mad64(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                          uint64_t modifiers, const Operand &destination, const Operand &source_a,
                          const Operand &source_b, const Operand &source_c, Commit commit) {
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  if (!exec_mask) {
    uint64_t carry64 = 0;
    uint32_t carry32 = 0;
    const int status =
        wave.wf_size() == 64
            ? wave64(flags, 0, modifiers, nullptr, &carry64, nullptr, nullptr, nullptr)
            : wave32(flags, 0, modifiers, nullptr, &carry32, nullptr, nullptr, nullptr);
    if (status != RJ_ARITH_SUCCESS) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    } else {
      commit(0);
    }
    return;
  }
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source_a, exec_mask), b(registers, wave, source_b, exec_mask);
  VectorInput64 c(registers, wave, source_c, exec_mask);
  uint64_t carry = 0;
  const bool success = write_vector64_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t) {
        if (wave.wf_size() == 64)
          return wave64(flags, exec_mask, modifiers, d, &carry, &a.data, &b.data, c.data.data());
        uint32_t carry32 = 0;
        const int status = wave32(flags, uint32_t(exec_mask), modifiers, d, &carry32, &a.data,
                                  &b.data, c.data.data());
        carry = carry32;
        return status;
      },
      (exec_mask & ~wave.vgpr_write_mask()) == 0);
  if (success)
    commit(carry);
}

// Carry and vector results use the DPP destination mask; source observations use EXEC.
template <bool WithCarry, typename Wave32, typename Wave64, typename Commit>
inline void execute_carry32(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                            uint64_t modifiers, const Operand &destination, const Operand &source_a,
                            const Operand &source_b, uint64_t input_carry, Commit commit) {
  const uint64_t exec_mask =
      wave.exec() & wave.vgpr_write_mask() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source_a, wave.exec()),
      b(registers, wave, source_b, wave.exec());
  uint64_t carry = 0;
  const bool success = write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t) {
        if (wave.wf_size() == 32) {
          uint32_t carry32 = 0;
          int status;
          if constexpr (WithCarry)
            status = wave32(flags, uint32_t(exec_mask), modifiers, d, &carry32, &a.data, &b.data,
                            uint32_t(input_carry));
          else
            status = wave32(flags, uint32_t(exec_mask), modifiers, d, &carry32, &a.data, &b.data);
          carry = carry32;
          return status;
        }
        if constexpr (WithCarry)
          return wave64(flags, exec_mask, modifiers, d, &carry, &a.data, &b.data, input_carry);
        else
          return wave64(flags, exec_mask, modifiers, d, &carry, &a.data, &b.data);
      },
      (exec_mask & ~wave.vgpr_write_mask()) == 0);
  if (success)
    commit(carry);
}

using Binary32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                 const uint32_t *const *, const uint32_t *const *);
using Binary64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                 const uint32_t *const *, const uint32_t *const *);

using Unary32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                const uint32_t *const *);
using Unary64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                const uint32_t *const *);
using Ternary32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);
using Ternary64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);

// Compare four overlapping byte windows. A is a pair, B a word, and C/D
// are two packed words or four U32 words. Four-word C must be a VGPR range.
// Keep the decoded operand type when copying individual register selectors.
template <typename DecodedOperand>
inline void execute_qsad(Wavefront &wave, Ternary32Function wave32, Ternary64Function wave64,
                         uint64_t flags, uint64_t modifiers, const DecodedOperand &destination,
                         const Operand &source_a, const Operand &source_b,
                         const DecodedOperand &source_c, bool packed) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput64 a(registers, wave, source_a, exec_mask);
  VectorInput32 b(registers, wave, source_b, exec_mask);
  std::optional<VectorInput64> packed_c;
  std::array<std::optional<VectorInput32>, 4> words_c;
  std::array<const uint32_t *, 4> c{};
  if (packed) {
    packed_c.emplace(registers, wave, source_c, exec_mask);
    c[0] = packed_c->data[0];
    c[1] = packed_c->data[1];
  } else {
    auto word = source_c;
    word.size_bits_ = 32;
    for (unsigned reg = 0; reg < 4; ++reg) {
      word.encoding_value_ = source_c.encoding_value_ + reg;
      words_c[reg].emplace(registers, wave, word, exec_mask);
      c[reg] = words_c[reg]->data;
    }
  }
  if (wave.instruction_execution_failed())
    return;
  const auto invoke = [&](uint32_t *const *d, uint64_t write_mask) {
    return wave.wf_size() == 32
               ? wave32(flags, uint32_t(write_mask), modifiers, d, a.data.data(), &b.data, c.data())
               : wave64(flags, write_mask, modifiers, d, a.data.data(), &b.data, c.data());
  };
  if (packed) {
    write_vector64_result(wave, registers, destination, exec_mask, invoke);
    return;
  }
  if (auto status = registers.try_bulk_write_operand<4>(destination, exec_mask, invoke)) {
    if (*status != RJ_ARITH_SUCCESS)
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  auto word = destination;
  word.size_bits_ = 32;
  std::array<std::array<uint32_t, 64>, 4> result{};
  uint32_t *d[] = {result[0].data(), result[1].data(), result[2].data(), result[3].data()};
  if (invoke(d, exec_mask) != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  for (unsigned reg = 0; reg < 4; ++reg) {
    word.encoding_value_ = destination.encoding_value_ + reg;
    if (word.simd_capable()) {
      auto output = registers.write_operand(word, exec_mask);
      for (uint32_t lane = 0; lane < wave.wf_size(); lane += util::native_width_v<uint32_t>)
        if (output
                .store_native(lane, util::load<uint32_t>(result[reg].data() + lane),
                              exec_mask >> lane)
                .failed())
          return;
    } else {
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane)
        if ((exec_mask >> lane) & 1)
          registers.write_lane(word, lane, result[reg][lane]);
    }
  }
}

// rj-arith preserves aliased source values. Direct stores and staged write views
// both honor EXEC and the decoder's DPP write mask.
inline void execute_binary32(Wavefront &wave, Binary32Function wave32, Binary64Function wave64,
                             uint64_t flags, uint64_t modifiers, const Operand &destination,
                             const Operand &source_a, const Operand &source_b,
                             uint8_t destination_bytes = 0xF, uint32_t packed_negations = 0) {
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source_a, exec_mask), b(registers, wave, source_b, exec_mask);
  if (packed_negations) {
    a.apply_packed_integer_signs(wave.wf_size(), 0, packed_negations, modifiers);
    b.apply_packed_integer_signs(wave.wf_size(), 1, packed_negations, modifiers);
  }
  write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data);
      },
      /*allow_direct=*/destination_bytes == 0xF, destination_bytes);
}

// Read the routed source before committing any destination lanes.
inline void execute_unary32(Wavefront &wave, Unary32Function wave32, Unary64Function wave64,
                            uint64_t flags, uint64_t modifiers, const Operand &destination,
                            const Operand &source, uint8_t source_bytes = 0xF,
                            uint8_t destination_bytes = 0xF) {
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source, exec_mask, source_bytes);
  write_vector32_result(
      wave, registers, destination, exec_mask,
      [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32 ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data)
                                    : wave64(flags, write_mask, modifiers, d, &a.data);
      },
      /*allow_direct=*/destination_bytes == 0xF, destination_bytes);
}

// Pack a staged SDWA source before applying integer destination placement.
// All source lanes are captured before any potentially aliased destination write.
template <typename Inst>
inline void execute_saturating_pack_sdwa(Inst &inst, Wavefront &wave, Unary32Function wave32,
                                         Unary64Function wave64, const Operand &destination,
                                         const Operand &source) {
  if (!wave.exec()) {
    return;
  }
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source, wave.exec());
  if (wave.instruction_execution_failed()) {
    return;
  }
  uint32_t result[64]{};
  uint32_t *d = result;
  uint64_t flags = wave.cu().arithmetic_flags();
  const int error = wave.wf_size() == 32 ? wave32(flags, uint32_t(exec_mask), 0, &d, &a.data)
                                         : wave64(flags, exec_mask, 0, &d, &a.data);
  if (error != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
    if (exec_mask & (1ULL << lane)) {
      sdwa::write_lane<sdwa::ResultFormat::NONE>(inst, wave, destination, lane, result[lane]);
    }
  }
}

// Copy a routed word, retaining SDWA destination placement without FP interpretation.
template <typename Inst>
inline void execute_move32(Inst &inst, Wavefront &wave, const Operand &destination,
                           const Operand &source) {
  if constexpr (requires { inst.sdwa_dst_sel_; }) {
    if (inst.inst_.src0 == amdgpu::SRC_SDWA) {
      const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
      if (!exec_mask) {
        return;
      }
      RegisterAccess registers(wave);
      VectorInput32 input(registers, wave, source, exec_mask);
      if (wave.instruction_execution_failed()) {
        return;
      }
      std::array<uint32_t, 64> result{};
      uint32_t *d = result.data();
      const auto flags = wave.cu().arithmetic_flags();
      const int status =
          wave.wf_size() == 32
              ? rj_arith_v_mov_b32_w32(flags, uint32_t(exec_mask), 0, &d, &input.data)
              : rj_arith_v_mov_b32_w64(flags, exec_mask, 0, &d, &input.data);
      if (status != RJ_ARITH_SUCCESS) {
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
        return;
      }
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        if (exec_mask & (uint64_t{1} << lane)) {
          sdwa::write_lane<sdwa::ResultFormat::NONE>(inst, wave, destination, lane, result[lane]);
        }
      }
      return;
    }
  }
  execute_unary32(wave, rj_arith_v_mov_b32_w32, rj_arith_v_mov_b32_w64,
                  wave.cu().arithmetic_flags(), 0, destination, source);
}

// CDNA screen partition is Wave64-only. SDWA commits selected bytes through the decoder;
// ordinary operands retain the direct bulk-write path.
template <typename Inst>
inline void execute_screen_partition(Inst &inst, Wavefront &wave, const Operand &destination,
                                     const Operand &source) {
  if constexpr (requires { inst.sdwa_dst_sel_; }) {
    if (inst.inst_.src0 == SRC_SDWA) {
      uint64_t exec_mask = wave.exec();
      if (!exec_mask) {
        return;
      }
      RegisterAccess registers(wave);
      VectorInput32 a(registers, wave, source, exec_mask);
      if (wave.instruction_execution_failed()) {
        return;
      }
      uint32_t result[64] = {};
      uint32_t *d = result;
      if (rj_arith_v_screen_partition_4se_b32_w64(wave.cu().arithmetic_flags(), exec_mask, 0, &d,
                                                  &a.data) != RJ_ARITH_SUCCESS) {
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
        return;
      }
      for (uint32_t lane = 0; lane < 64; ++lane) {
        if ((exec_mask >> lane) & 1) {
          sdwa::write_lane<sdwa::ResultFormat::NONE>(inst, wave, destination, lane, result[lane]);
        }
      }
      return;
    }
  }
  execute_unary32(wave, nullptr, rj_arith_v_screen_partition_4se_b32_w64,
                  wave.cu().arithmetic_flags(), 0, destination, source);
}

// Convert selected source bytes into two full VGPR results, reading the source
// before either destination word is committed.
inline void execute_widen32_to_pair(Wavefront &wave, Unary32Function wave32, Unary64Function wave64,
                                    uint64_t flags, uint64_t modifiers, const Operand &destination,
                                    const Operand &source, uint8_t source_bytes) {
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source, exec_mask, source_bytes);
  write_vector64_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32 ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data)
                                    : wave64(flags, write_mask, modifiers, d, &a.data);
      });
}

// Read live inputs before aliased stores. Integer callers observe EXEC; DOT4
// restricts source observations to its DPP-filtered destination lanes.
inline void execute_ternary32(Wavefront &wave, Ternary32Function wave32, Ternary64Function wave64,
                              uint64_t flags, uint64_t modifiers, const Operand &destination,
                              const Operand &source_a, const Operand &source_b,
                              const Operand &source_c, uint8_t a_bytes = 0xF, uint8_t b_bytes = 0xF,
                              uint64_t source_exec_mask = ~uint64_t{0},
                              uint32_t packed_negations = 0) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  source_exec_mask &= wave.exec();
  if (!source_exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source_a, source_exec_mask, a_bytes),
      b(registers, wave, source_b, source_exec_mask, b_bytes),
      c(registers, wave, source_c, source_exec_mask);
  if (packed_negations) {
    a.apply_packed_integer_signs(wave.wf_size(), 0, packed_negations, modifiers);
    b.apply_packed_integer_signs(wave.wf_size(), 1, packed_negations, modifiers);
    c.apply_packed_integer_signs(wave.wf_size(), 2, packed_negations, modifiers);
  }
  write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data);
      });
}

} // namespace rocjitsu::amdgpu::arith
