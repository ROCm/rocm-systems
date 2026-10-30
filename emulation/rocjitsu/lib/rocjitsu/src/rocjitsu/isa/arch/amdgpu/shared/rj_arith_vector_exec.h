// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_VECTOR_EXEC_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_VECTOR_EXEC_H_

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"

#include <array>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu::arith {

// A normal VGPR retains its observed storage view. Uniform, subword or delegated
// operands use scratch storage; their existing implementation owns routing.
class VectorInput32 {
public:
  VectorInput32(RegisterAccess &registers, Wavefront &wave, const Operand &operand,
                uint64_t exec_mask, uint8_t byte_mask = 0xF) {
    if (operand.simd_capable() && byte_mask == 0xF)
      region_ = registers.read_operand_region(operand, exec_mask);
    if (region_) {
      data = region_->lanes(0).data();
      return;
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

  const uint32_t *data;

private:
  std::optional<RegisterAccess::VgprReadRegion> region_;
  std::array<uint32_t, 64> values_;
};

// rj-arith represents each 64-bit lane as low/high words in two separate VGPRs.
class VectorInput64 {
public:
  VectorInput64(RegisterAccess &registers, Wavefront &wave, const Operand &operand,
                uint64_t exec_mask) {
    region_ = registers.read_operand_region(operand, exec_mask, 2);
    if (region_) {
      data = {region_->lanes(0).data(), region_->lanes(1).data()};
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
      auto view = registers.read_operand_pair32(operand, exec_mask);
      for (uint32_t lane = 0; lane < wave.wf_size(); lane += util::native_width_v<uint32_t>) {
        view.load_lo_native<uint32_t>(lane).copy_to(values_[0].data() + lane,
                                                    util::stdx::element_aligned);
        view.load_hi_native<uint32_t>(lane).copy_to(values_[1].data() + lane,
                                                    util::stdx::element_aligned);
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

// Direct stores are safe without observers: rj-arith preserves storage on rejection.
// Other destinations stage the result before acquiring an observed write view.
// Returns true on success, or reports an instruction error and returns false.
template <typename Invoke>
inline bool write_vector32_result(Wavefront &wave, RegisterAccess &registers,
                                  const Operand &destination, uint64_t exec_mask, Invoke invoke,
                                  bool allow_direct = true, uint8_t byte_mask = 0xF) {
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
      output.store_native(lane, util::load<uint32_t>(result.data() + lane), exec_mask >> lane);
  } else {
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane)
      if (exec_mask & (uint64_t{1} << lane))
        registers.write_lane(destination, lane, result[lane]);
  }
  return true;
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
      output.store_native_pair(lane, util::load<uint32_t>(result[0].data() + lane),
                               util::load<uint32_t>(result[1].data() + lane), exec_mask >> lane);
  } else {
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane)
      if (exec_mask & (uint64_t{1} << lane))
        registers.write_lane64(destination, lane,
                               result[0][lane] | (uint64_t(result[1][lane]) << 32));
  }
  return true;
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

// Scalar carry observes every active lane, even when DPP suppresses VGPR writes.
// The decoder's result callback owns preservation of source-suppressed carry bits.
template <bool WithCarry, typename Wave32, typename Wave64, typename Commit>
inline void execute_carry32(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                            uint64_t modifiers, const Operand &destination, const Operand &source_a,
                            const Operand &source_b, uint64_t input_carry, Commit commit) {
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source_a, exec_mask), b(registers, wave, source_b, exec_mask);
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
        output.store_native(lane, util::load<uint32_t>(result[reg].data() + lane),
                            exec_mask >> lane);
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
                             uint8_t destination_bytes = 0xF) {
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source_a, exec_mask), b(registers, wave, source_b, exec_mask);
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

// All three routed inputs remain live through the call, including aliased VGPRs.
inline void execute_ternary32(Wavefront &wave, Ternary32Function wave32, Ternary64Function wave64,
                              uint64_t flags, uint64_t modifiers, const Operand &destination,
                              const Operand &source_a, const Operand &source_b,
                              const Operand &source_c, uint8_t source_a_bytes = 0xF,
                              uint8_t source_b_bytes = 0xF) {
  const uint64_t exec_mask = wave.exec() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source_a, exec_mask, source_a_bytes),
      b(registers, wave, source_b, exec_mask, source_b_bytes),
      c(registers, wave, source_c, exec_mask);
  write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), modifiers, d, &a.data, &b.data, &c.data)
                   : wave64(flags, write_mask, modifiers, d, &a.data, &b.data, &c.data);
      });
}

} // namespace rocjitsu::amdgpu::arith

#endif // ROCJITSU_ISA_AMDGPU_SHARED_RJ_ARITH_VECTOR_EXEC_H_
