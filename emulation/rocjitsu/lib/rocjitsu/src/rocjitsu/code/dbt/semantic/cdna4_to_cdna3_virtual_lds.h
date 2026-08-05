// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna4_to_cdna3_virtual_lds.h
/// @brief Pair-specific virtual-LDS lowering entry points.

#pragma once

#include "rocjitsu/code/dbt/lds_virtualization.h"

namespace rocjitsu {

/// @brief Return true when a CDNA4 instruction needs the CDNA3 virtual-LDS sidecar.
[[nodiscard]] bool
cdna4_to_cdna3_source_instruction_uses_virtualizable_lds(const Instruction &inst);

/// @brief Reserve the CDNA3 scalar state used by virtual-LDS lowering.
[[nodiscard]] std::optional<VirtualLdsBaseSgprReservation>
reserve_cdna4_to_cdna3_virtual_lds_base_sgpr_pair(TranslationContext &context,
                                                  KernelBlockScope blocks,
                                                  const KdTranslation &translation);

/// @brief Append the CDNA3 prologue that initializes the virtual-LDS pointer.
[[nodiscard]] bool append_cdna4_to_cdna3_virtual_lds_entry_prologue(KdTranslation &translation);

/// @brief Lower one CDNA4 LDS access to CDNA3 backing-buffer traffic.
[[nodiscard]] ExpandResult
lower_cdna4_to_cdna3_virtual_lds_instruction(const Instruction &inst, TranslationContext &context);

} // namespace rocjitsu
