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
using Ternary32Function = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);
using Ternary64Function = int (*)(uint64_t, uint64_t, uint64_t, uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *);

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
