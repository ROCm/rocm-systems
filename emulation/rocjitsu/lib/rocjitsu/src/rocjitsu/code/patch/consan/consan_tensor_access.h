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

/// Exact per-lane unsigned division for decoding tensor tile coordinates.
/// Division by zero yields UINT32_MAX and the original dividend. Inputs,
/// outputs, and three consecutive scratch VGPRs must be disjoint. Preserves
/// inputs, SGPRs, EXEC, SCC, and inactive lanes; clobbers VCC. No memory access.
[[nodiscard]] bool append_tensor_divmod_u32(std::vector<uint32_t> &words, uint16_t dividend_vgpr,
                                            uint16_t divisor_vgpr, uint16_t quotient_vgpr,
                                            uint16_t remainder_vgpr, uint16_t scratch_vgpr,
                                            rj_code_arch_t arch);

/// The 64-bit counterpart used to invert tensor iteration offsets with 48-bit
/// strides. Each input/output names a consecutive VGPR pair; four scratch
/// VGPRs are required. Same preservation/alias contract as the 32-bit helper;
/// division by zero yields UINT64_MAX and the original dividend.
[[nodiscard]] bool append_tensor_divmod_u64(std::vector<uint32_t> &words, uint16_t dividend_vgpr,
                                            uint16_t divisor_vgpr, uint16_t quotient_vgpr,
                                            uint16_t remainder_vgpr, uint16_t scratch_vgpr,
                                            rj_code_arch_t arch);

/// Invert an iteration's linear global element offset for an admitted,
/// nonempty, advancing dense tensor-load descriptor (rank two or three).
/// Outputs three consecutive 64-bit logical coordinates, including any
/// padding remainder in dimension zero. Descriptor strides may reorder axes.
/// Input pair, six output VGPRs and 18 scratch VGPRs must be disjoint.
/// Preserves inputs, SGPRs, EXEC, SCC and inactive lanes; clobbers VCC.
/// The caller handles empty/disabled descriptors without reading global memory.
[[nodiscard]] bool append_tensor_iteration_origin(std::vector<uint32_t> &words,
                                                  const ProgramSite &site, uint16_t offset_vgpr,
                                                  uint16_t origin_vgpr, uint16_t scratch_vgpr,
                                                  rj_code_arch_t arch);

/// Recover the global source of a selected unpadded LDS element. The inputs
/// are the element/count produced by append_select_tensor_load_element for an
/// admitted LDS-fitting descriptor. Repeated overlapping tiles select the last
/// writer. Dense/gather bounds produce a per-lane 0/1 predicate; a zero predicate
/// means the expected LDS value is zero and MUST NOT cause a global read.
/// Outputs are a global-address VGPR pair and the bounds predicate. Inputs,
/// outputs, and 30 scratch VGPRs must be disjoint. Preserves input registers,
/// SGPRs, EXEC, SCC, and inactive lanes; clobbers VCC. Caller selects low VGPR banks.
[[nodiscard]] bool
append_materialize_tensor_load_source(std::vector<uint32_t> &words, const ProgramSite &site,
                                      uint16_t element_vgpr, uint16_t count_vgpr,
                                      uint16_t address_vgpr, uint16_t in_bounds_vgpr,
                                      uint16_t scratch_vgpr, rj_code_arch_t arch);

inline constexpr uint16_t kTensorLoadCompareScratchVgprs = 46;
inline constexpr uint16_t kTensorLoadCompareWorkspaceOffset = 16;

/// Four flag-save SGPRs, plus eight descriptor-copy SGPRs when D1[0] aliases
/// another descriptor group. Zero means descriptor operands are unavailable.
[[nodiscard]] uint16_t tensor_load_compare_state_sgprs(const ProgramSite &site);

/// Emit a wave-wide tensor-load value check. Samples source values before the
/// original DMA, executes it once, then compares sampled LDS values. Atomic
/// completion is deferred until after comparison and performed exactly once
/// through native LDS barrier-arrive. The caller preserves all 46 scratch VGPRs
/// across all lanes and selects low VGPR banks; four dead, even-aligned ordinary
/// SGPRs hold incoming VCC/EXEC and must not overlap descriptor groups.
/// If D1[0] aliases another descriptor group, eight additional dead SGPRs
/// hold a D1 copy so disabling completion cannot change a second operand.
/// Delay/action words must preserve SGPRs, EXEC and SCC; an action may clobber
/// VCC and the workspace, while delay must also preserve the saved payload and
/// LDS address. Returns the word offset of the relocated original instruction.
[[nodiscard]] bool append_tensor_load_compare(std::vector<uint32_t> &words, const ProgramSite &site,
                                              std::span<const uint32_t> original,
                                              uint16_t scratch_vgpr, uint16_t state_sgpr,
                                              std::span<const uint32_t> delay_words,
                                              std::span<const uint32_t> mismatch_words,
                                              uint32_t &guest_word_offset, rj_code_arch_t arch);

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
