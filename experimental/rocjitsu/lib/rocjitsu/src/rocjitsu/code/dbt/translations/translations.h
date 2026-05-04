// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translations.h
/// @brief Per-architecture semantic translation rule tables.

#pragma once

#include "rocjitsu/code/dbt/translation_rule.h"

#include <span>

namespace rocjitsu {

[[nodiscard]] std::span<const TranslationRule> cdna4_to_cdna3_expand_rules();

} // namespace rocjitsu
