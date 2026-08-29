// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_supercollider_support.h
/// @brief Private SuperCollider scratch and guest-replay planning helpers.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <array>
#include <cstdint>
#include <optional>

namespace rocjitsu {

class Instruction;
class LivenessAnalysis;

[[nodiscard]] bool is_instrumentable_group_flat_hint(ConSanFlatAddressSpaceHint hint,
                                                     ConSanFlatProvenanceMode mode);
[[nodiscard]] std::optional<uint16_t> flat_dword_count(const ConSanAccessInventorySite &access);
[[nodiscard]] bool flat_scratch_tuple_base_is_valid(const ConSanAccessInventorySite &access,
                                                    uint16_t candidate);
[[nodiscard]] uint16_t flat_scratch_search_start(const ConSanAccessInventorySite &access);
[[nodiscard]] std::optional<uint16_t>
choose_flat_scratch_vgpr(const ConSanAccessInventorySite &access, const ConSanOptions &options,
                         const Instruction *inst, const LivenessAnalysis *liveness,
                         std::optional<uint16_t> min_auto_scratch_vgpr,
                         std::optional<uint16_t> max_auto_scratch_vgpr, uint16_t required_vgprs);
[[nodiscard]] std::optional<uint16_t>
choose_flat_spill_scratch_vgpr(const ConSanAccessInventorySite &access, uint16_t allocation_count,
                               uint16_t required_vgprs);
[[nodiscard]] std::optional<std::array<uint32_t, 3>>
retarget_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst, rj_code_arch_t arch);
[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_flat_load_from_flat_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst,
                                rj_code_arch_t arch);
[[nodiscard]] std::optional<uint16_t>
flat_check_trap_compare_vgpr(const ConSanAccessInventorySite &access);
[[nodiscard]] std::optional<uint16_t>
check_trap_compare_vgpr(const ConSanAccessInventorySite &access, uint16_t chunk_index,
                        rj_code_arch_t arch, uint16_t gfx1250_vgpr_msb_mode);

} // namespace rocjitsu
