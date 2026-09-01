// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_virtualization.h
/// @brief DBT helpers for translating oversized hardware LDS usage to a backing buffer.

#pragma once

#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {

class Instruction;

/// @brief SGPR reservation chosen for virtual-LDS address generation.
///
/// @details The binary translator only needs the resource contract. The
/// architecture-specific prologue and DS lowering details live under semantic/;
/// lds_virtualization.cpp dispatches to them so the main translation loop does
/// not grow target-pair or target-encoding knowledge.
struct VirtualLdsBaseSgprReservation {
  uint16_t base = 0;
  uint16_t prologue_temp = 0;
  bool spill_per_use = false;
};

/// @brief Return true when this translation pair can emit virtual-LDS sidecars.
///
/// @details BinaryTranslator uses this as a capability query so its main
/// relocation flow does not need target-pair-specific branches. The actual
/// target instruction encoders and resource limits remain in pair-specific
/// semantic implementations.
[[nodiscard]] bool supports_virtual_lds_sidecars(rj_code_arch_t guest_arch,
                                                 rj_code_arch_t host_arch);

/// @brief Return true when @p inst requires a virtual-LDS body variant.
///
/// @details The current implementation supports CDNA4 guest LDS storage
/// operations when translating to CDNA3. Other architecture pairs return false
/// until they get an explicit lowering path.
[[nodiscard]] bool source_instruction_uses_virtualizable_lds(const Instruction &inst,
                                                             rj_code_arch_t guest_arch,
                                                             rj_code_arch_t host_arch);

/// @brief Reserve scalar state used by virtual-LDS address lowering.
///
/// @details The selected pair is either permanently descriptor-backed, or it is
/// borrowed around each lowered LDS operation with save/restore code emitted by
/// the architecture-specific lowering.
[[nodiscard]] std::optional<VirtualLdsBaseSgprReservation>
reserve_virtual_lds_base_sgpr_pair(TranslationContext &context, KernelBlockScope blocks,
                                   const KdTranslation &translation, rj_code_arch_t guest_arch,
                                   rj_code_arch_t host_arch);

/// @brief Append the target entry prologue that initializes the virtual-LDS base.
///
/// @returns false when the descriptor ABI state cannot encode the prologue for
/// the selected host architecture.
[[nodiscard]] bool append_virtual_lds_entry_prologue(KdTranslation &translation,
                                                     rj_code_arch_t guest_arch,
                                                     rj_code_arch_t host_arch);

/// @brief Try to lower one instruction into virtual-LDS backing-buffer traffic.
///
/// @details Unsupported architecture pairs return ExpandStatus::NotHandled.
/// Supported pairs return a failure if the instruction definitely accesses LDS
/// storage but this lowering cannot preserve it safely.
[[nodiscard]] ExpandResult lower_virtual_lds_instruction(const Instruction &inst,
                                                         TranslationContext &context,
                                                         rj_code_arch_t guest_arch,
                                                         rj_code_arch_t host_arch);

} // namespace rocjitsu
