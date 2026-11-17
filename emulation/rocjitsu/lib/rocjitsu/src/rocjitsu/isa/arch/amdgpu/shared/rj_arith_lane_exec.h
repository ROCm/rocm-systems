// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rj_arith_vector_exec.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu::arith {

// Observe only the selected source lane. READFIRSTLANE selects lane zero when
// EXEC is zero; explicit READLANE indices wrap independently of EXEC.
template <bool First>
inline void execute_read_lane(Wavefront &wave, uint64_t flags, const Operand &destination,
                              const Operand &source, uint32_t lane = 0) {
  const uint64_t exec_mask = wave.exec();
  lane = First ? (exec_mask ? std::countr_zero(exec_mask) : 0) : lane & (wave.wf_size() - 1);
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source, 1ULL << lane);
  if (wave.instruction_execution_failed())
    return;
  uint32_t result = 0;
  int status;
  if constexpr (First)
    status = wave.wf_size() == 32
                 ? rj_arith_v_readfirstlane_b32_w32(flags, uint32_t(exec_mask), 0, &result, &a.data)
                 : rj_arith_v_readfirstlane_b32_w64(flags, exec_mask, 0, &result, &a.data);
  else
    status = wave.wf_size() == 32 ? rj_arith_v_readlane_b32_w32(flags, 0, &result, &a.data, lane)
                                  : rj_arith_v_readlane_b32_w64(flags, 0, &result, &a.data, lane);
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  registers.write_scalar(destination, result);
}

// WRITELANE ignores EXEC and vector destination masks. Stage its single result
// and use the scalar-lane writer, which also observes that independent write.
inline void execute_write_lane(Wavefront &wave, uint64_t flags, const Operand &destination,
                               const Operand &source, const Operand &selector) {
  RegisterAccess registers(wave);
  const uint32_t value = registers.read_scalar(source);
  const uint32_t lane = registers.read_scalar(selector) & (wave.wf_size() - 1);
  if (wave.instruction_execution_failed())
    return;
  std::array<uint32_t, 64> result;
  uint32_t *d = result.data();
  const int status = wave.wf_size() == 32 ? rj_arith_v_writelane_b32_w32(flags, 0, &d, value, lane)
                                          : rj_arith_v_writelane_b32_w64(flags, 0, &d, value, lane);
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  registers.write_scalar_selected_lane(destination, lane, result[lane]);
}

// Snapshot both read/write operands before either commit. VOP1 half operands
// already resolve their encoded halves; commit only their logical low halves.
template <bool Half>
inline void execute_swap(Wavefront &wave, uint64_t flags, const Operand &destination,
                         const Operand &source) {
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  constexpr uint8_t bytes = Half ? 0x3 : 0xf;
  VectorInput32 old_d(registers, wave, destination, exec_mask, bytes);
  VectorInput32 old_a(registers, wave, source, exec_mask, bytes);
  if (wave.instruction_execution_failed())
    return;
  std::array<uint32_t, 64> d_values, a_values;
  std::copy_n(old_d.data, wave.wf_size(), d_values.data());
  std::copy_n(old_a.data, wave.wf_size(), a_values.data());
  uint32_t *d = d_values.data(), *a = a_values.data();
  int status;
  if constexpr (Half)
    status = wave.wf_size() == 32 ? rj_arith_v_swap_b16_w32(flags, uint32_t(exec_mask), 0, &d, &a)
                                  : rj_arith_v_swap_b16_w64(flags, exec_mask, 0, &d, &a);
  else
    status = wave.wf_size() == 32 ? rj_arith_v_swap_b32_w32(flags, uint32_t(exec_mask), 0, &d, &a)
                                  : rj_arith_v_swap_b32_w64(flags, exec_mask, 0, &d, &a);
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  // Packed half operands expose physical storage, but their logical writes also
  // apply the encoded half selector. Raw storage views must not bypass it.
  if (!destination.simd_capable() || !source.simd_capable()) {
    for (unsigned lane = 0; lane < wave.wf_size(); ++lane) {
      if ((exec_mask >> lane) & 1) {
        registers.write_lane(destination, lane, d_values[lane]);
        registers.write_lane(source, lane, a_values[lane]);
      }
    }
    return;
  }
  auto output_d = registers.write_operand(destination, exec_mask, bytes);
  auto output_a = registers.write_operand(source, exec_mask, bytes);
  for (unsigned lane = 0; lane < wave.wf_size(); lane += util::native_width_v<uint32_t>) {
    if (output_d.store_native(lane, util::load<uint32_t>(d_values.data() + lane), exec_mask >> lane)
            .failed())
      return;
    if (output_a.store_native(lane, util::load<uint32_t>(a_values.data() + lane), exec_mask >> lane)
            .failed())
      return;
  }
}

// Exchange adjacent rows without requiring the source lane to be active.
// Read both operands before publishing either output; observation masks cover
// the opposite row consumed by each active destination.
template <int Stride>
inline void execute_permlane_swap(Wavefront &wave, uint64_t flags, const Operand &destination,
                                  const Operand &source) {
  if (wave.wf_size() <= Stride) {
    return;
  }
  uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  uint64_t low_rows = Stride == 16 ? 0x0000ffff0000ffffULL : 0x00000000ffffffffULL;
  uint64_t a_write = exec_mask & low_rows, d_write = exec_mask & ~low_rows;
  if (!exec_mask) {
    return;
  }
  RegisterAccess registers(wave);
  VectorInput32 old_a(registers, wave, source, d_write >> Stride);
  VectorInput32 old_d(registers, wave, destination, a_write << Stride);
  if (wave.instruction_execution_failed()) {
    return;
  }
  std::array<uint32_t, 64> a_values, d_values;
  std::copy_n(old_a.data, wave.wf_size(), a_values.data());
  std::copy_n(old_d.data, wave.wf_size(), d_values.data());
  uint32_t *a = a_values.data(), *d = d_values.data();
  int status;
  if constexpr (Stride == 16) {
    status = wave.wf_size() == 32
                 ? rj_arith_v_permlane16_swap_b32_w32(flags, uint32_t(exec_mask), 0, &d, &a)
                 : rj_arith_v_permlane16_swap_b32_w64(flags, exec_mask, 0, &d, &a);
  } else {
    status = rj_arith_v_permlane32_swap_b32_w64(flags, exec_mask, 0, &d, &a);
  }
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  for (unsigned lane = 0; lane < wave.wf_size(); ++lane) {
    if ((a_write >> lane) & 1) {
      registers.write_lane(source, lane, a_values[lane]);
    }
    if ((d_write >> lane) & 1) {
      registers.write_lane(destination, lane, d_values[lane]);
    }
  }
}

// Retain scalar group_width access and active-lane source observation for index generation.
inline void execute_permlane_index(Wavefront &wave, uint64_t flags, const Operand &destination,
                                   const Operand &source, const Operand &width_operand) {
  RegisterAccess registers(wave);
  uint32_t group_width = registers.read_scalar(width_operand);
  uint64_t exec_mask = wave.exec();
  if (!exec_mask || wave.instruction_execution_failed()) {
    return;
  }
  if (!width_operand.is_vgpr() && (!group_width || (group_width & (group_width - 1)))) {
    wave.report_undefined_behavior("permlane group width is not a power of two");
  }
  VectorInput32 a(registers, wave, source, exec_mask);
  write_vector32_result(
      wave, registers, destination, exec_mask, [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32 ? rj_arith_v_permlane_idx_gen_b32_w32(
                                          flags, uint32_t(write_mask), 0, d, &a.data, group_width)
                                    : rj_arith_v_permlane_idx_gen_b32_w64(flags, write_mask, 0, d,
                                                                          &a.data, group_width);
      });
}

// Wave32 is a no-op without operand access. Wave64 reads the opposite half,
// including inactive source lanes, and masks destination stores normally.
inline void execute_permlane64(Wavefront &wave, uint64_t flags, const Operand &destination,
                               const Operand &source) {
  if (wave.wf_size() == 32) {
    if (rj_arith_v_permlane64_b32_w32(flags, 0, nullptr, nullptr) != RJ_ARITH_SUCCESS)
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  const uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  VectorInput32 a(registers, wave, source, (exec_mask << 32) | (exec_mask >> 32));
  write_vector32_result(wave, registers, destination, exec_mask,
                        [&](uint32_t *const *d, uint64_t write_mask) {
                          return rj_arith_v_permlane64_b32_w64(flags, write_mask, 0, d, &a.data);
                        });
}

// Permutation arithmetic stays in rj-arith. The observed/staged path calculates the
// destination write mask separately, since an unreadable source without FI or
// BOUND_CTRL preserves D and must not produce a write notification.
template <bool Cross, bool Variable, typename Wave32, typename Wave64>
inline void execute_permlane16(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                               uint64_t modifiers, const Operand &destination,
                               const Operand &source, const Operand &indices_or_lo,
                               const Operand *high = nullptr) {
  const uint64_t exec_mask = wave.exec();
  if (!exec_mask)
    return;
  RegisterAccess registers(wave);
  std::optional<VectorInput32> indices;
  uint32_t lo = 0, hi = 0;
  if constexpr (Variable)
    indices.emplace(registers, wave, indices_or_lo, exec_mask);
  else {
    if (exec_mask & 0x00ff00ff00ff00ffULL)
      lo = registers.read_scalar(indices_or_lo);
    if (exec_mask & 0xff00ff00ff00ff00ULL)
      hi = registers.read_scalar(*high);
  }
  if (wave.instruction_execution_failed())
    return;
  const auto permutation = valu_permutation_lanes(
      Cross ? ValuPermutation::X16 : ValuPermutation::Perm16, exec_mask, wave.wf_size(), 16, true,
      modifiers & RJ_ARITH_PERMLANE_FI, modifiers & RJ_ARITH_PERMLANE_BOUND_CTRL,
      [&](unsigned lane) {
        return Variable ? indices->data[lane] : ((lane & 8 ? hi : lo) >> ((lane & 7) * 4));
      });
  VectorInput32 a(registers, wave, source, permutation.read);
  const auto invoke = [&](uint32_t *const *d) {
    if constexpr (Variable)
      return wave.wf_size() == 32
                 ? wave32(flags, uint32_t(exec_mask), modifiers, d, &a.data, &indices->data)
                 : wave64(flags, exec_mask, modifiers, d, &a.data, &indices->data);
    else
      return wave.wf_size() == 32
                 ? wave32(flags, uint32_t(exec_mask), modifiers, d, &a.data, lo, hi)
                 : wave64(flags, exec_mask, modifiers, d, &a.data, lo, hi);
  };
  // Unobserved ordinary destinations keep rj-arith's direct SIMD masked stores.
  if ((exec_mask & wave.vgpr_write_mask()) == exec_mask) {
    auto status = registers.try_bulk_write_operand(
        destination, exec_mask, [&](uint32_t *const *d, uint64_t) { return invoke(d); });
    if (status) {
      if (*status != RJ_ARITH_SUCCESS)
        wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
  }
  const uint64_t write_mask = permutation.write & wave.vgpr_write_mask();
  if (!write_mask)
    return;
  write_vector32_result(
      wave, registers, destination, write_mask,
      [&](uint32_t *const *d, uint64_t) { return invoke(d); },
      /*allow_direct=*/false);
}

// Observe the selected source lanes, including inactive sources, before allowing
// direct masked output. Decoder operands retain scalar and subword interpretation.
template <ValuPermutation Kind, typename Wave32, typename Wave64>
inline void execute_group_permutation(Wavefront &wave, Wave32 wave32, Wave64 wave64, uint64_t flags,
                                      const Operand &destination, const Operand &source,
                                      uint32_t selector, uint32_t group_width) {
  uint64_t exec_mask = wave.exec();
  if (!exec_mask) {
    return;
  }
  uint64_t read_mask = 0;
  for (unsigned lane = 0; lane < wave.wf_size(); ++lane) {
    if ((exec_mask >> lane) & 1) {
      unsigned selected =
          valu_permutation_source(Kind, lane, selector, group_width, wave.wf_size());
      if (selected < wave.wf_size()) {
        read_mask |= uint64_t{1} << selected;
      }
    }
  }
  RegisterAccess registers(wave);
  VectorInput32 input(registers, wave, source, read_mask);
  write_vector32_result(
      wave, registers, destination, exec_mask & wave.vgpr_write_mask(),
      [&](uint32_t *const *d, uint64_t write_mask) {
        return wave.wf_size() == 32
                   ? wave32(flags, uint32_t(write_mask), 0, d, &input.data, selector, group_width)
                   : wave64(flags, write_mask, 0, d, &input.data, selector, group_width);
      });
}

// Resolve numeric operand widths before invoking the packed permutation. Stage
// all result words so aliases and observed multi-register writes remain ordered.
template <int Bits, typename Wave32, typename Wave64, typename Destination>
inline void execute_packed_permutation(Wavefront &wave, Wave32 wave32, Wave64 wave64,
                                       uint64_t flags, const Destination &destination,
                                       const Operand &source_a, const Operand &source_b,
                                       const Operand &source_c) {
  uint64_t exec_mask = wave.exec() & wave.vgpr_write_mask();
  if (!exec_mask) {
    return;
  }
  RegisterAccess registers(wave);
  std::optional<VectorInput32> a32, b32;
  std::optional<VectorInput64> a64, b64;
  const uint32_t *a[2], *b[2];
  if constexpr (Bits == 8) {
    a64.emplace(registers, wave, source_a, exec_mask);
    a[0] = a64->data[0];
    a[1] = a64->data[1];
  } else {
    a32.emplace(registers, wave, source_a, exec_mask);
    a[0] = a32->data;
  }
  if constexpr (Bits == 4) {
    b32.emplace(registers, wave, source_b, exec_mask);
    b[0] = b32->data;
  } else {
    b64.emplace(registers, wave, source_b, exec_mask);
    b[0] = b64->data[0];
    b[1] = b64->data[1];
  }
  VectorInput64 c(registers, wave, source_c, exec_mask);
  if (wave.instruction_execution_failed()) {
    return;
  }
  std::array<uint32_t, 64> result[Bits / 2]{};
  uint32_t *d[Bits / 2];
  for (int word = 0; word < Bits / 2; ++word) {
    d[word] = result[word].data();
  }
  int status = wave.wf_size() == 32 ? wave32(flags, uint32_t(exec_mask), 0, d, a, b, c.data.data())
                                    : wave64(flags, exec_mask, 0, d, a, b, c.data.data());
  if (status != RJ_ARITH_SUCCESS) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  // Preserve the concrete ISA operand when adjusting its register index.
  Destination output = destination;
  output.size_bits_ = 32;
  for (int word = 0; word < Bits / 2; ++word) {
    output.encoding_value_ = destination.encoding_value_ + word;
    auto view = registers.write_operand(output, exec_mask);
    for (unsigned lane = 0; lane < wave.wf_size(); lane += util::native_width_v<uint32_t>) {
      if (view.store_native(lane, util::load<uint32_t>(d[word] + lane), exec_mask >> lane)
              .failed()) {
        return;
      }
    }
  }
}

} // namespace rocjitsu::amdgpu::arith
