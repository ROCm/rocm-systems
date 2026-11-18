// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/rules.h
/// @brief Declarations for ISA-pair semantic expansion rule tables.

#pragma once

#include "rocjitsu/code/dbt/translation_rule.h"
#include "rocjitsu/code/rj_code.h"

#include <span>

namespace rocjitsu {

/// @brief CDNA4 source rules for the RDNA4 target.
[[nodiscard]] std::span<const TranslationRule> semantic_expand_rules_cdna4_to_rdna4();

/// @brief CDNA4 VOP2 reverse shifts with different encoding or DPP routing on RDNA.
[[nodiscard]] std::span<const RegisteredInstructionRewrite>
cdna4_reverse_shift_rules_for_rdna(rj_code_arch_t host_arch);

/// @brief CDNA4 source rules for the CDNA3 target.
[[nodiscard]] std::span<const TranslationRule> semantic_expand_rules_cdna4_to_cdna3();

/// @brief gfx1250 B0-to-A0 semantic expansions.
[[nodiscard]] std::span<const TranslationRule> semantic_expand_rules_gfx1250_b0_to_a0();

/// @brief Complete gfx1250 B0-to-A0 handwritten rewrite registry.
[[nodiscard]] RewriteRegistry rewrite_registry_gfx1250_b0_to_a0();

/// @brief CDNA4 source rules for the RDNA3 target.
[[nodiscard]] std::span<const TranslationRule> semantic_expand_rules_cdna4_to_rdna3();

} // namespace rocjitsu
