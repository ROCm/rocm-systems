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

#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
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

/// @brief Bucket an instruction using decoded facts and the actual memory route.
/// @details Matrix and GWS classification precedes routing. Control instructions
/// precede the decoded-memory fallback. This ordering is shared by both reporting
/// plugins and preserves their exclusive JSONL family contract.
inline InstructionFamily classify_instruction(const Instruction &inst) {
  if (inst.is_mfma() || inst.is_matrix())
    return InstructionFamily::Matrix;

  switch (inst.amdgpu_operation()) {
  case amdgpu::Operation::DS_GWS_BARRIER:
  case amdgpu::Operation::DS_GWS_INIT:
  case amdgpu::Operation::DS_GWS_SEMA_V:
  case amdgpu::Operation::DS_GWS_SEMA_P:
  case amdgpu::Operation::DS_GWS_SEMA_BR:
  case amdgpu::Operation::DS_GWS_SEMA_RELEASE_ALL:
    return InstructionFamily::Control;
  default:
    break;
  }

  const auto decoded_memory_family = [&inst]() -> InstructionFamily {
    switch (inst.amdgpu_memory_addressing()) {
    case amdgpu::MemoryAddressing::Lds:
    case amdgpu::MemoryAddressing::Gds:
    case amdgpu::MemoryAddressing::DirectLds:
      return InstructionFamily::Lds;
    case amdgpu::MemoryAddressing::Scalar:
    case amdgpu::MemoryAddressing::Flat:
    case amdgpu::MemoryAddressing::Global:
    case amdgpu::MemoryAddressing::Scratch:
    case amdgpu::MemoryAddressing::Buffer:
    case amdgpu::MemoryAddressing::TypedBuffer:
    case amdgpu::MemoryAddressing::Image:
    case amdgpu::MemoryAddressing::Tensor:
    case amdgpu::MemoryAddressing::Cluster:
      return InstructionFamily::Global;
    case amdgpu::MemoryAddressing::None:
      break;
    }
    if ((inst.flags() & LDS) != 0)
      return InstructionFamily::Lds;
    if ((inst.flags() & (VMEM | SMEM)) != 0)
      return InstructionFamily::Global;
    const amdgpu::MemoryIssueInfo *issue = inst.amdgpu_memory_issue_info();
    if (issue == nullptr)
      return InstructionFamily::Other;
    for (const amdgpu::MemoryCounterObligation obligation : issue->counter_obligations()) {
      if (obligation.completion_class() == amdgpu::MemoryCompletionClass::LDS)
        return InstructionFamily::Lds;
    }
    return InstructionFamily::Global;
  };

  if (inst.is_memory_op()) {
    if (const DynamicInstState *state = inst.data()) {
      if (state->tag() == amdgpu::LOCAL_MEM)
        return InstructionFamily::Lds;
      if (state->tag() == amdgpu::GLOBAL_MEM || state->tag() == amdgpu::SCALAR_MEM)
        return InstructionFamily::Global;
    }
    const InstructionFamily family = decoded_memory_family();
    return family != InstructionFamily::Other ? family : InstructionFamily::Global;
  }

  constexpr uint64_t kControlFlags = BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL |
                                     PROGRAM_TERMINATOR | WAITCNT | BARRIER;
  if ((inst.flags() & kControlFlags) != 0)
    return InstructionFamily::Control;
  switch (inst.amdgpu_operation()) {
  case amdgpu::Operation::S_NOP:
  case amdgpu::Operation::S_SLEEP:
  case amdgpu::Operation::S_SLEEP_VAR:
  case amdgpu::Operation::S_DELAY_ALU:
  case amdgpu::Operation::S_BARRIER:
  case amdgpu::Operation::S_BARRIER_LEAVE:
  case amdgpu::Operation::S_BARRIER_INIT:
  case amdgpu::Operation::S_BARRIER_JOIN:
  case amdgpu::Operation::S_BARRIER_SIGNAL:
  case amdgpu::Operation::S_BARRIER_SIGNAL_ISFIRST:
  case amdgpu::Operation::S_BARRIER_WAIT:
  case amdgpu::Operation::DS_NOP:
    return InstructionFamily::Control;
  case amdgpu::Operation::S_MEMTIME:
  case amdgpu::Operation::S_MEMREALTIME:
    // Timestamp reads share SMEM completion but report scalar work, not traffic.
    return InstructionFamily::Scalar;
  default:
    break;
  }

  if (const InstructionFamily family = decoded_memory_family(); family != InstructionFamily::Other)
    return family;
  if ((inst.flags() & SALU) != 0)
    return InstructionFamily::Scalar;
  if ((inst.flags() & VALU) != 0)
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
