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

  // Memory traffic that the generated encodings describe only by name. Whole
  // prefix families never set MEMORY_OP -- RDNA4 `ds_direct_load` and
  // `tbuffer_load_format_x`, RDNA3.5 `lds_direct_load`, the image encodings
  // (generated/rdna4/vimage.cpp mentions MEMORY_OP nowhere), the CDNA5 tensor
  // transfers, and every scalar-memory encoding: generated/*/smem.cpp stopped
  // setting the flag entirely when the decoded memory-issue metadata landed,
  // so `s_load_*` and friends now arrive unflagged. Several other mnemonics
  // carry it on one architecture and not another. Classifying by prefix as
  // well as by flag keeps a mnemonic in the same family whichever
  // architecture executed it.
  //
  // The tensor pair moves between global memory and LDS; it is bucketed as
  // global for the same reason a non-LDS memory op is -- the global side is
  // the access being described. Cache maintenance (`buffer_inv`, `buffer_wb*`)
  // acts on the global hierarchy and is counted there too.
  const auto memory_family = [&has_prefix]() -> InstructionFamily {
    if (has_prefix("ds_") || has_prefix("lds_"))
      return InstructionFamily::Lds;
    if (has_prefix("buffer_") || has_prefix("tbuffer_") || has_prefix("global_") ||
        has_prefix("scratch_") || has_prefix("flat_") || has_prefix("image_") ||
        has_prefix("tensor_"))
      return InstructionFamily::Global;
    // Scalar memory. `s_load`/`s_store`/`s_atomic`/`s_scratch` are data
    // access; `s_dcache_*`, `s_prefetch_*`, `s_atc_probe*` and `s_gl1_inv` are
    // cache maintenance against the same hierarchy, grouped with it for the
    // reason the vector-side `buffer_inv`/`buffer_wb*` are. All of them would
    // otherwise be counted as ALU work by the bare `s_` prefix below.
    if (has_prefix("s_load") || has_prefix("s_store") || has_prefix("s_atomic") ||
        has_prefix("s_buffer") || has_prefix("s_scratch") || has_prefix("s_dcache") ||
        has_prefix("s_prefetch") || has_prefix("s_atc_probe") || has_prefix("s_gl1_inv"))
      return InstructionFamily::Global;
    return InstructionFamily::Other;
  };

  // Last resort for memory traffic no prefix knows about. Develop's decoded
  // memory-issue metadata names the completion domain directly, which catches
  // encodings whose mnemonic follows no family convention -- the CDNA5
  // `cluster_load_*` set is the live case, and it carries no MEMORY_OP either.
  // Consulted after the prefix table so that generic FLAT, which reports both
  // LDS and VMEM obligations, keeps the documented global bucket.
  const auto issued_memory_family = [&inst]() -> InstructionFamily {
    const auto *issue = inst.amdgpu_memory_issue_info();
    if (issue == nullptr)
      return InstructionFamily::Other;
    for (const auto obligation : issue->counter_obligations()) {
      if (obligation.completion_class() == amdgpu::MemoryCompletionClass::LDS)
        return InstructionFamily::Lds;
    }
    return InstructionFamily::Global;
  };

  if (inst.is_memory_op()) {
    if (const auto *state = inst.data()) {
      if (state->tag() == amdgpu::LOCAL_MEM)
        return InstructionFamily::Lds;
      if (state->tag() == amdgpu::GLOBAL_MEM || state->tag() == amdgpu::SCALAR_MEM)
        return InstructionFamily::Global;
    }

    // No execution-pipeline state: synthetic and model-only instructions reach
    // here. Name them instead, defaulting to global as the flag already said
    // this is a memory op.
    if (const InstructionFamily named = memory_family(); named != InstructionFamily::Other)
      return named;
    if (const InstructionFamily issued = issued_memory_family(); issued != InstructionFamily::Other)
      return issued;
    return InstructionFamily::Global;
  }

  constexpr uint64_t control_flags = BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL |
                                     PROGRAM_TERMINATOR | WAITCNT | BARRIER;
  // Checked before the memory fallback below so that LDS-pipe synchronisation
  // and no-ops are not counted as LDS traffic. `s_barrier` catches the split
  // barrier family: on CDNA5 only `s_barrier_wait` carries the BARRIER flag,
  // so init/join/leave/signal/signal_isfirst would otherwise land in `scalar`
  // next to ordinary ALU work. `ds_gws_` is the GWS barrier/semaphore set,
  // which carries no flag on any architecture.
  if ((inst.flags() & control_flags) != 0 || has_prefix("s_nop") || has_prefix("s_sleep") ||
      has_prefix("s_delay") || has_prefix("s_barrier") || has_prefix("ds_nop") ||
      has_prefix("ds_gws_"))
    return InstructionFamily::Control;

  if (const InstructionFamily named = memory_family(); named != InstructionFamily::Other)
    return named;
  if (const InstructionFamily issued = issued_memory_family(); issued != InstructionFamily::Other)
    return issued;

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
