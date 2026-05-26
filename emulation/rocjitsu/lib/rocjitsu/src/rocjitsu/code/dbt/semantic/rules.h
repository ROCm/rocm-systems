// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/rules.h
/// @brief Declarations for ISA-pair semantic expansion rule tables.

#pragma once

#include "rocjitsu/code/dbt/translation_rule.h"

#include <span>

namespace rocjitsu {

/// @brief CDNA4 source rules for the RDNA4 target.
[[nodiscard]] std::span<const TranslationRule> semantic_expand_rules_cdna4_to_rdna4();

/// @brief CDNA4 source rules for the CDNA3 target.
[[nodiscard]] std::span<const TranslationRule> semantic_expand_rules_cdna4_to_cdna3();

/// @brief CDNA4 source rules for the RDNA3 target.
[[nodiscard]] std::span<const TranslationRule> semantic_expand_rules_cdna4_to_rdna3();

} // namespace rocjitsu
