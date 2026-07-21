// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_translation_profile.cpp
/// @brief Self-contained registration for the gfx1250 errata HotSwap path.

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/code/dbt/gfx1250_b0_to_a0.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <memory>

namespace rocjitsu {

namespace {

[[nodiscard]] std::unique_ptr<Decoder> create_gfx1250_decoder(rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250)
    return nullptr;
  return std::make_unique<IsaDecoder<gfx1250::Isa>>();
}

} // namespace

TranslationProfile gfx1250_b0_to_a0_translation_profile() {
  return {.encoding_translate = nullptr,
          .legalization_lookup = gfx1250_b0_to_a0_legalization,
          .semantic_rules = semantic_expand_rules_gfx1250_b0_to_a0(),
          .create_decoder = create_gfx1250_decoder};
}

} // namespace rocjitsu
