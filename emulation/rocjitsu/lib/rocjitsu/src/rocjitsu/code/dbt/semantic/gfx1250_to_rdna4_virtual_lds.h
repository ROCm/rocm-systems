// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/dbt/lds_virtualization.h"

namespace rocjitsu {

[[nodiscard]] bool gfx1250_to_rdna4_supports_virtual_lds();
[[nodiscard]] bool gfx1250_source_instruction_uses_virtualizable_lds(const Instruction &inst);
[[nodiscard]] std::optional<VirtualLdsBaseSgprReservation>
reserve_rdna4_virtual_lds_base_sgpr_pair(TranslationContext &context,
                                         const KdTranslation &translation);
[[nodiscard]] bool append_rdna4_virtual_lds_entry_prologue(KdTranslation &translation);
[[nodiscard]] ExpandResult lower_gfx1250_to_rdna4_virtual_lds_instruction(
    const Instruction &inst, const LivenessAnalysis &liveness, TranslationContext &context);

} // namespace rocjitsu
