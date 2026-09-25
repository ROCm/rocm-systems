// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/rj_code.h"
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu::consan {
struct ProgramSite;
struct ProgramContainerId;
class ProgramInventory;
struct DispatchIdentity;
struct WorkgroupSources;
struct PrivateStateLayout;
struct OperatingPoint;
namespace detail {
struct PlannedProbeResources;

/// Owners which may execute wave-wide tensor accesses, including shared helpers.
[[nodiscard]] std::vector<ProgramContainerId>
tensor_execution_owner_kernels(const ProgramInventory &inventory);
[[nodiscard]] bool site_has_tensor_owner(const ProgramInventory &inventory, const ProgramSite &site,
                                         std::span<const ProgramContainerId> tensor_owners);
[[nodiscard]] bool
tensor_identity_sources_are_wave_uniform(const DispatchIdentity &dispatch,
                                         const WorkgroupSources &workgroup,
                                         bool full_wave_private_initialized = false);

/// Wrap a fixed-stack VGPR spill/fill so both preserve all wave lanes, even
/// when the guest EXEC is empty. Each bracket restores its incoming EXEC.
/// The caller owns a dead ordinary SGPR pair, disjoint from descriptor operands
/// and scalar spill state; this is not a bootstrap for borrowed scalar scratch.
[[nodiscard]] std::optional<VgprSpillSequence>
tensor_full_wave_spill(const VgprSpillSequence &spill, uint16_t exec_save_sgpr,
                       rj_code_arch_t arch);

/// Preserve a borrowed ordinary scalar window through fixed private memory,
/// independently of guest EXEC. The auxiliary SGPR pair must be dead and
/// disjoint from the borrowed window; the transfer VGPR is preserved by caller.
[[nodiscard]] std::optional<SgprSpillSequence>
tensor_full_wave_scalar_spill(const SgprSpillSequence &spill, uint16_t auxiliary_sgpr,
                              rj_code_arch_t arch);
/// Resolve the bootstrap pair and wrap both spill classes for full-wave use.
[[nodiscard]] bool prepare_tensor_full_wave_resources(PlannedProbeResources &probe,
                                                      const OperatingPoint &point,
                                                      rj_code_arch_t arch);

/// Replicate initialized lane-zero identity to every lane at kernel entry.
/// The SGPR pair is borrowed by the entry prologue; the caller preserves any
/// guest ABI value in it. EXEC/VCC/SCC and both scratch VGPRs are preserved.
/// Spill slots must be disjoint from the prologue's existing live spill frame.
[[nodiscard]] bool append_full_wave_private_identity(std::vector<uint32_t> &words,
                                                     const PrivateStateLayout &layout,
                                                     uint16_t scratch_vgpr, uint16_t exec_save_sgpr,
                                                     const VgprSpillSequence &spill,
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
