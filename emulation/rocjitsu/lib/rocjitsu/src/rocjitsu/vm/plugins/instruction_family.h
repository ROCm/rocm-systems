// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file instruction_family.h
/// @brief The exclusive instruction families shared by the reporting plugins.
///
/// The throughput and instruction-mix plugins both bucket executed
/// instructions into these families and both put the bucket names in their
/// JSONL. Their reports are meant to join on mnemonic, which only works while
/// the two agree on every classification -- so they classify through this one
/// header rather than through per-plugin copies that have to be kept in step.
///
/// Header-only on purpose: each plugin is built into a self-contained
/// `librocjitsu_plugin_<name>.so` that links only its own object library, so a
/// shared translation unit would need every plugin's link line to grow a
/// common target for no benefit.

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rocjitsu::plugins {

/// Exclusive instruction families. The order is part of both plugins' JSONL
/// schemas -- they present the per-family breakdown in enum order -- so
/// entries may be appended before `Count` but never reordered or removed.
enum class InstructionFamily : size_t {
  Scalar,
  Vector,
  Matrix,
  Lds,
  Global,
  Control,
  Other,
  Count,
};

inline constexpr size_t kInstructionFamilyCount = static_cast<size_t>(InstructionFamily::Count);

/// @brief Bucket @p inst into its family.
///
/// Classification prefers the instruction's own metadata (the MFMA flag, the
/// memory-op flag plus the execution pipeline's address space, the control
/// flags) and falls back to the mnemonic prefix, which keeps synthetic and
/// model-only instructions useful even when they carry no pipeline state.
inline InstructionFamily classify_instruction(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  const auto has_prefix = [mnemonic](std::string_view prefix) {
    return mnemonic.starts_with(prefix);
  };

  if (inst.is_mfma() || has_prefix("v_mfma_") || has_prefix("v_smfmac_") || has_prefix("v_wmma_") ||
      has_prefix("v_swmmac_"))
    return InstructionFamily::Matrix;

  if (inst.is_memory_op()) {
    if (const auto *state = inst.data()) {
      if (state->tag() == amdgpu::LOCAL_MEM)
        return InstructionFamily::Lds;
      if (state->tag() == amdgpu::GLOBAL_MEM || state->tag() == amdgpu::SCALAR_MEM)
        return InstructionFamily::Global;
    }

    // These fallbacks keep synthetic/model-only instructions useful even when
    // they do not carry execution-pipeline state.
    if (has_prefix("ds_"))
      return InstructionFamily::Lds;
    return InstructionFamily::Global;
  }

  constexpr uint64_t control_flags = BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL |
                                     PROGRAM_TERMINATOR | WAITCNT | BARRIER;
  if ((inst.flags() & control_flags) != 0 || has_prefix("s_nop") || has_prefix("s_sleep") ||
      has_prefix("s_delay"))
    return InstructionFamily::Control;
  if (has_prefix("s_"))
    return InstructionFamily::Scalar;
  if (has_prefix("v_"))
    return InstructionFamily::Vector;
  return InstructionFamily::Other;
}

/// @brief The JSONL key for @p family. Out-of-range values report as "other".
inline std::string_view instruction_family_name(InstructionFamily family) {
  constexpr std::array<std::string_view, kInstructionFamilyCount> names = {
      "scalar", "vector", "matrix", "lds", "global", "control", "other"};
  const size_t index = static_cast<size_t>(family);
  return index < names.size() ? names[index] : "other";
}

} // namespace rocjitsu::plugins
