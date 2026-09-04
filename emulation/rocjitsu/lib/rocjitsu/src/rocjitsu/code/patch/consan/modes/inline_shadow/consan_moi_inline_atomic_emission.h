// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_inline_atomic_emission.h
/// @brief InlineShadow atomic-ordering native emission contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_sync_emission.h"
#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_shadow_emission.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] bool
inline_atomic_scalar_spill_aliases_guest_address(const ConSanMoiAtomicAddressPlan &address_plan,
                                                 uint16_t scalar_base, uint16_t scalar_count);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_atomic_ordering_cave_words(
    std::span<const uint8_t> bytes, const consan_detail::MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiInlineAtomicEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint64_t cave_text_offset, uint64_t return_text_offset, uint32_t &guest_instruction_offset,
    std::vector<std::string> &errors, std::span<const uint32_t> trailing_guest_words = {},
    bool fallthrough = false);

} // namespace rocjitsu::consan_moi_impl
