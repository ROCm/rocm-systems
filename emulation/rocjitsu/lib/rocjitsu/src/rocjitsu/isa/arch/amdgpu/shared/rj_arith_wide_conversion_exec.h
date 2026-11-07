// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wide scaled conversion bridges validate register ranges, borrow observed
// inputs, and publish only successfully computed active-lane results.
#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include <array>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace rocjitsu::amdgpu::arith {
// Read the scale before data: exponent-255 lanes do not depend on data or seeds.
// Narrowing preserves those lanes; widening still publishes canonical NaNs there.
// PairA describes FP4 stochastic pairs; PairD describes packed FP32 widening.
template <bool Narrow, bool Ternary, bool PairA = false, bool PairD = false, typename Wave32,
          typename Wave64>
inline void execute_scaled_conversion(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                                      uint64_t modifiers, const Operand &destination,
                                      const Operand &source_a, const Operand *source_b,
                                      const Operand &scale_operand, uint8_t source_bytes,
                                      uint8_t destination_bytes) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask) {
    return;
  }
  RegisterAccess registers(wave);
  VectorInput32 scale(registers, wave, scale_operand, exec_mask);
  uint64_t data_mask = exec_mask;
  for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
    if (((scale.data[lane] >> 23) & 255) == 255) {
      data_mask &= ~(1ULL << lane);
    }
  }
  using Input = std::conditional_t<PairA, VectorInput64, VectorInput32>;
  // Paired input uses all bytes; the single-register form selects its encoded bytes.
  auto read_a = [&]() {
    if constexpr (PairA) {
      return Input(registers, wave, source_a, data_mask);
    } else {
      return Input(registers, wave, source_a, data_mask, source_bytes);
    }
  };
  Input a = read_a();
  std::optional<VectorInput32> b;
  if constexpr (Ternary) {
    b.emplace(registers, wave, *source_b, data_mask);
  }
  const uint32_t *const *a_words;
  if constexpr (PairA) {
    a_words = a.data.data();
  } else {
    a_words = &a.data;
  }
  auto invoke = [&](uint32_t *const *d, uint64_t lanes) {
    if constexpr (Ternary) {
      return wave.wf_size() == 32
                 ? wave32(flags, uint32_t(lanes), modifiers, d, a_words, &b->data, &scale.data)
                 : wave64(flags, lanes, modifiers, d, a_words, &b->data, &scale.data);
    } else {
      return wave.wf_size() == 32
                 ? wave32(flags, uint32_t(lanes), modifiers, d, a_words, &scale.data)
                 : wave64(flags, lanes, modifiers, d, a_words, &scale.data);
    }
  };
  const uint64_t write_mask = Narrow ? data_mask : exec_mask;
  if (!write_mask) {
    if (!wave.instruction_execution_failed() && invoke(nullptr, 0) != RJ_ARITH_SUCCESS) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    }
  } else if constexpr (PairD) {
    write_vector64_result(wave, registers, destination, write_mask, invoke);
  } else {
    write_vector32_result(wave, registers, destination, write_mask, invoke,
                          /*allow_direct=*/destination_bytes == 0xf || Narrow, destination_bytes,
                          /*preserve_destination=*/Narrow);
  }
}

// Widen packed narrow registers using pre-resolved physical A/D bases.
// ByteScale selects gfx1250 PK8/PK16 and E8M0; gfx950 PK32 special scales perform
// no source-A reads or destination writes. mode selects the E8M0 byte.
// Decoders provide at most six input registers and 32 output registers.
template <bool ByteScale = false>
inline void execute_wide_scaled_widen(Wavefront &wave, Binary32Function wave32,
                                      Binary64Function wave64, uint64_t flags, uint32_t destination,
                                      uint32_t source, const Operand &scale, uint32_t output_words,
                                      uint64_t mode = 0, uint32_t input_words = ByteScale ? 3 : 6) {
  uint64_t exec_mask =
      wave.exec() & wave.vgpr_write_mask() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  RegisterAccess registers(wave);
  if (!exec_mask || wave.instruction_execution_failed() ||
      !registers.owns_vgpr_range(source, input_words) ||
      !registers.owns_vgpr_range(destination, output_words)) {
    return;
  }
  VectorInput32 b(registers, wave, scale, exec_mask);
  for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
    if (!ByteScale && ((b.data[lane] >> 23) & 255) == 255) {
      exec_mask &= ~(1ULL << lane);
    }
  }
  if (wave.instruction_execution_failed()) {
    return;
  }
  if (!exec_mask) {
    // Special scales suppress data accesses, not semantics validation.
    const int status = wave.wf_size() == 32 ? wave32(flags, 0, mode, nullptr, nullptr, nullptr)
                                            : wave64(flags, 0, mode, nullptr, nullptr, nullptr);
    if (status != RJ_ARITH_SUCCESS) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    }
    return;
  }
  auto input = registers.read_vgpr_region(source, input_words, exec_mask);
  std::array<const uint32_t *, 6> a;
  for (uint32_t reg = 0; reg < input_words; ++reg) {
    auto lanes = input.lanes(reg);
    if (lanes.failed()) {
      return;
    }
    a[reg] = lanes.value().data();
  }
  auto invoke = [&](uint32_t *const *d) {
    return wave.wf_size() == 32 ? wave32(flags, uint32_t(exec_mask), mode, d, a.data(), &b.data)
                                : wave64(flags, exec_mask, mode, d, a.data(), &b.data);
  };
  if (auto status = registers.try_bulk_write_vgprs<32>(destination, output_words, invoke)) {
    if (*status != RJ_ARITH_SUCCESS) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    }
    return;
  }
  // Observed writes are transactional. Only lanes in exec_mask are initialized
  // by the API and subsequently published; other lanes need no scratch copy.
  std::array<std::array<uint32_t, 64>, 32> result;
  std::array<uint32_t *, 32> d;
  for (uint32_t reg = 0; reg < output_words; ++reg) {
    d[reg] = result[reg].data();
  }
  if (invoke(d.data()) != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  auto output = registers.write_vgpr_region(destination, output_words, exec_mask);
  for (uint32_t reg = 0; reg < output_words; ++reg) {
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
      if ((exec_mask >> lane) & 1) {
        output.set_lane(reg, lane, result[reg][lane]);
      }
    }
  }
}
// Narrow 32 elements into six registers. Interleaved FP32 RNE consumes A16/B16;
// other gfx950 forms consume A16 or A32 and optionally a stochastic seed.
// gfx950 scale-exponent255 lanes perform no data reads or destination writes.
// FullScale selects gfx1250: eight or 16 elements, one to three outputs, full
// FP32 division scales and no special-scale suppression.
template <bool Ternary, bool FullScale = false, int OutputWords = FullScale ? 3 : 6,
          typename Wave32, typename Wave64>
void execute_wide_scaled_narrow(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                                uint32_t destination, uint32_t source_a, uint32_t source_b,
                                uint32_t input_words, bool interleaved, const Operand *seed_operand,
                                const Operand &scale_operand) {
  constexpr uint32_t output_words = OutputWords;
  uint64_t exec_mask =
      wave.exec() & wave.vgpr_write_mask() & (wave.wf_size() == 32 ? 0xffffffffULL : ~0ULL);
  RegisterAccess registers(wave);
  if (!exec_mask || wave.instruction_execution_failed() ||
      !registers.owns_vgpr_range(source_a, input_words) ||
      (interleaved && !registers.owns_vgpr_range(source_b, 16)) ||
      !registers.owns_vgpr_range(destination, output_words)) {
    return;
  }
  VectorInput32 scale(registers, wave, scale_operand, exec_mask);
  for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
    if (!FullScale && ((scale.data[lane] >> 23) & 255) == 255) {
      exec_mask &= ~(1ULL << lane);
    }
  }
  if (wave.instruction_execution_failed()) {
    return;
  }
  if constexpr (!FullScale) {
    if (!exec_mask) {
      // Validate even when every special-scale lane suppresses its result. The
      // zero EXEC API contract permits null data pointers and performs no reads.
      int status;
      if constexpr (Ternary) {
        status = wave.wf_size() == 32 ? wave32(flags, 0, 0, nullptr, nullptr, nullptr, nullptr)
                                      : wave64(flags, 0, 0, nullptr, nullptr, nullptr, nullptr);
      } else {
        status = wave.wf_size() == 32 ? wave32(flags, 0, 0, nullptr, nullptr, nullptr)
                                      : wave64(flags, 0, 0, nullptr, nullptr, nullptr);
      }
      if (status != RJ_ARITH_SUCCESS) {
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      }
      return;
    }
  }
  auto input = registers.read_vgpr_region(source_a, input_words, exec_mask);
  std::array<const uint32_t *, 32> a;
  for (uint32_t reg = 0; reg < input_words; ++reg) {
    auto lanes = input.lanes(reg);
    if (lanes.failed()) {
      return;
    }
    a[reg] = lanes.value().data();
  }
  std::array<const uint32_t *, 16> b;
  std::optional<VectorInput32> seed;
  if (interleaved) {
    auto second = registers.read_vgpr_region(source_b, 16, exec_mask);
    for (uint32_t reg = 0; reg < 16; ++reg) {
      auto lanes = second.lanes(reg);
      if (lanes.failed()) {
        return;
      }
      b[reg] = lanes.value().data();
    }
  } else if (seed_operand) {
    seed.emplace(registers, wave, *seed_operand, exec_mask);
    b[0] = seed->data;
  }
  if (wave.instruction_execution_failed()) {
    return;
  }
  auto invoke = [&](uint32_t *const *d) {
    if constexpr (FullScale) {
      if constexpr (Ternary) {
        return wave32(flags, uint32_t(exec_mask), 0, d, a.data(), b.data(), &scale.data);
      } else {
        return wave32(flags, uint32_t(exec_mask), 0, d, a.data(), &scale.data);
      }
    } else if constexpr (Ternary) {
      return wave.wf_size() == 32
                 ? wave32(flags, uint32_t(exec_mask), 0, d, a.data(), b.data(), &scale.data)
                 : wave64(flags, exec_mask, 0, d, a.data(), b.data(), &scale.data);
    } else {
      return wave.wf_size() == 32 ? wave32(flags, uint32_t(exec_mask), 0, d, a.data(), &scale.data)
                                  : wave64(flags, exec_mask, 0, d, a.data(), &scale.data);
    }
  };
  if (auto status =
          registers.try_bulk_write_vgprs<output_words>(destination, output_words, invoke)) {
    if (*status != RJ_ARITH_SUCCESS) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    }
    return;
  }
  // Observers receive writes only after successful arithmetic, with no reads
  // of inactive destination storage. gfx950 special-scale lanes remain excluded.
  std::array<std::array<uint32_t, 64>, output_words> result;
  std::array<uint32_t *, output_words> d;
  for (uint32_t reg = 0; reg < output_words; ++reg) {
    d[reg] = result[reg].data();
  }
  if (invoke(d.data()) != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  auto output = registers.write_vgpr_region(destination, output_words, exec_mask);
  for (uint32_t reg = 0; reg < output_words; ++reg) {
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
      if ((exec_mask >> lane) & 1) {
        output.set_lane(reg, lane, result[reg][lane]);
      }
    }
  }
}
} // namespace rocjitsu::amdgpu::arith
