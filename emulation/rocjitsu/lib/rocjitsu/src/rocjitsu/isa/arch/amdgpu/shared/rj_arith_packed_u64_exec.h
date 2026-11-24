// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1251 packed U64 carriers, bounded shift admission and transactional rj-arith calls.
#pragma once
#include "rj_arith/detail/rj_arith_integer.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <cassert>
#include <optional>
namespace rocjitsu::amdgpu::arith {
// Borrow an entire packed VGPR tuple, or materialize the scalar carrier's replicated elements.
// Four-word inputs contain two U64s; two-word count inputs contain two independent U32s.
template <int Words, typename DecodedOperand> class PackedU64Input {
public:
  PackedU64Input(RegisterAccess &registers, const DecodedOperand &operand, uint32_t exec_mask) {
    region_ = registers.read_operand_region(operand, exec_mask, Words);
    if (region_) {
      for (int reg = 0; reg < Words; ++reg) {
        auto lanes = region_->lanes(reg);
        if (lanes.failed()) {
          data.fill(nullptr);
          return;
        }
        data[reg] = lanes.value().data();
      }
      return;
    }
    for (auto &word : values_)
      word.fill(0);
    const auto reg = operand.to_register_ref();
    const bool vgpr = reg && reg->cls == RegClass::VGPR;
    for (int lane = 0; lane < 32; ++lane) {
      if (!(exec_mask >> lane & 1))
        continue;
      if constexpr (Words == 2) {
        // Scalar count operands broadcast one word, rather than reading a scalar pair.
        const auto value = registers.read_lane_pair32(operand, lane, ScalarPairMode::Replicate32);
        values_[0][lane] = value.lo;
        values_[1][lane] = value.hi;
      } else {
        uint64_t lo = registers.read_lane64(operand, lane), hi = lo;
        if (vgpr) {
          auto upper = operand;
          upper.encoding_value_ += 2;
          upper.size_bits_ = 64;
          hi = registers.read_lane64(upper, lane);
        }
        values_[0][lane] = uint32_t(lo);
        values_[1][lane] = uint32_t(lo >> 32);
        values_[2][lane] = uint32_t(hi);
        values_[3][lane] = uint32_t(hi >> 32);
      }
    }
    for (int word = 0; word < Words; ++word)
      data[word] = values_[word].data();
  }
  std::array<const uint32_t *, Words> data;

private:
  std::optional<RegisterAccess::VgprReadRegion> region_;
  std::array<std::array<uint32_t, 32>, Words> values_;
};
// Execute a packed U64 instruction after all source reads; observers see stores only on success.
// ShiftAdd keeps the existing active-count admission in VM, outside the arithmetic API domain.
template <bool ShiftAdd, typename Function, typename Destination, typename A, typename B,
          typename C = A>
inline void execute_packed_u64(Wavefront &wave, Function function, uint64_t modifiers,
                               const Destination &destination, const A &lhs, const B &rhs,
                               const C *third = nullptr) {
  assert(wave.wf_size() == 32);
  const uint32_t exec_mask = uint32_t(wave.exec());
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  PackedU64Input<4, A> a(registers, lhs, exec_mask);
  PackedU64Input<ShiftAdd ? 2 : 4, B> b(registers, rhs, exec_mask);
  std::optional<PackedU64Input<4, C>> c;
  if constexpr (ShiftAdd)
    c.emplace(registers, *third, exec_mask);
  if (wave.instruction_execution_failed())
    return;
  if constexpr (ShiftAdd) {
    for (int lane = 0; lane < 32; ++lane) {
      if ((exec_mask >> lane & 1) && (b.data[0][lane] > 4 || b.data[1][lane] > 4)) {
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
        return;
      }
    }
  }
  auto invoke = [&](uint32_t *const *d, uint64_t write_mask) {
    if constexpr (ShiftAdd)
      return function(wave.cu().arithmetic_flags(), uint32_t(write_mask), modifiers, d,
                      a.data.data(), b.data.data(), c->data.data());
    else
      return function(wave.cu().arithmetic_flags(), uint32_t(write_mask), modifiers, d,
                      a.data.data(), b.data.data());
  };
  if (auto status = registers.try_bulk_write_operand<4>(destination, exec_mask, invoke)) {
    if (*status != RJ_ARITH_SUCCESS)
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  // Stage observed destinations before asking for any write view.
  uint32_t result[4][32];
  uint32_t *d[] = {result[0], result[1], result[2], result[3]};
  const uint32_t write_mask = exec_mask & uint32_t(wave.vgpr_write_mask());
  if (invoke(d, write_mask) != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  auto word = destination;
  word.size_bits_ = 32;
  for (int reg = 0; reg < 4; ++reg) {
    word.encoding_value_ = destination.encoding_value_ + reg;
    for (int lane = 0; lane < 32; ++lane)
      if (write_mask >> lane & 1)
        registers.write_lane(word, lane, result[reg][lane]);
  }
}
} // namespace rocjitsu::amdgpu::arith
