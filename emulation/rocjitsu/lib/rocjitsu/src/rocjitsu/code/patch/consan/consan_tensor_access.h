// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/rj_code.h"
#include <cstdint>
#include <vector>

namespace rocjitsu::consan {
struct ProgramSite;
namespace detail {

/// Wrap a fixed-stack VGPR spill/fill so both preserve all wave lanes, even
/// when the guest EXEC is empty. Each bracket restores its incoming EXEC.
/// The caller owns a dead ordinary SGPR pair, disjoint from descriptor operands
/// and scalar spill state; this is not a bootstrap for borrowed scalar scratch.
[[nodiscard]] std::optional<VgprSpillSequence>
tensor_full_wave_spill(const VgprSpillSequence &spill, uint16_t exec_save_sgpr,
                       rj_code_arch_t arch);

/// Select an element of a valid, LDS-fitting CDNA5 tensor-load descriptor.
/// Two caller-provided 32-bit hashes select a position within the tile and an
/// iteration independently. Returns the linear LDS element index (including
/// iteration increment) and the tile's element count; count zero suppresses
/// observation. Global bounds do not suppress load observations: masked loads
/// still write zeros to LDS. All outputs and five consecutive scratch VGPRs
/// must be disjoint from the hash inputs. Preserves hashes, SGPRs, EXEC, SCC,
/// and inactive lanes; clobbers VCC. Caller selects the low VGPR banks.
[[nodiscard]] bool append_select_tensor_load_element(std::vector<uint32_t> &words,
                                                     const ProgramSite &site,
                                                     uint16_t element_hash_vgpr,
                                                     uint16_t iteration_hash_vgpr,
                                                     uint16_t element_vgpr, uint16_t count_vgpr,
                                                     uint16_t scratch_vgpr, rj_code_arch_t arch);

} // namespace detail
} // namespace rocjitsu::consan
