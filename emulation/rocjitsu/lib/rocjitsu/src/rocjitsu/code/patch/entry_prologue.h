// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file entry_prologue.h
/// @brief The DBI kernel-entry prologue: reserved SGPR storage and its words.

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief SGPRs the entry prologue reserves: a persistent pair plus an
///        entry-only temporary pair, allocated as one aligned run.
inline constexpr uint8_t kDbiEntryStorageRegisters = 4;

/// @brief SGPR pairs the DBI entry prologue owns for the life of the kernel.
struct DbiEntryStorage {
  uint16_t persistent_base = 0; ///< s[base:base+1], written at entry, read at every site.
  uint16_t entry_temp_base = 0; ///< s[base+2:base+3], live only inside the prologue.
};

/// @brief Lowest SGPR index above everything the kernel's own code and ABI name:
///        @ref explicit_ordinary_sgpr_bound folded with the descriptor's
///        user-SGPR block and its enabled system SGPRs (workgroup IDs and
///        workgroup info).
///
/// @details Deliberately not COMPUTE_PGM_RSRC1's SGPR count. That is an
/// allocation total whose tail can hold VCC, flat scratch, XNACK and granularity
/// padding, so starting new storage there can land in special-SGPR territory.
[[nodiscard]] uint32_t dbi_entry_storage_floor(KernelBlockScope blocks,
                                               const rocr::llvm::amdhsa::kernel_descriptor_t &desc,
                                               rj_code_arch_t arch);

/// @brief Choose the SGPR run the entry prologue owns, or fail closed.
///
/// @param blocks Every block reachable from the kernel entry. A partial scope
///        understates the floor and yields a run the kernel is already using.
/// @param desc The kernel's descriptor, read for its user-SGPR ABI.
/// @param arch ISA used to decode @p desc's USER_SGPR_COUNT layout.
/// @param kernel_sgpr_count The kernel's SGPR allocation. The run must fit
///        inside it; the allocation is not grown.
/// @param reserved Registers the caller has already spoken for, such as the
///        probe-call return-link pair. Skipped rather than treated as a floor,
///        so a kernel can place storage below them.
/// @param error_out Optional; filled with the reason on failure.
/// @returns The chosen storage, or nullopt when the allocation leaves no
///          aligned run of @ref kDbiEntryStorageRegisters above the floor.
[[nodiscard]] std::optional<DbiEntryStorage>
plan_dbi_entry_storage(KernelBlockScope blocks, const rocr::llvm::amdhsa::kernel_descriptor_t &desc,
                       rj_code_arch_t arch, uint32_t kernel_sgpr_count, const RegisterSet &reserved,
                       std::string *error_out = nullptr);

/// @brief The one payload DBI appends to a kernel's kernarg wrapper: a 64-bit
///        pointer the prologue loads into its persistent pair.
inline constexpr KernargExtensionPayloadLayout kDbiEntryPayloadLayout{.size = 8, .alignment = 8};

/// @brief Variant name DBI declares its kernarg extension under.
inline constexpr std::string_view kDbiKernargVariantName = "dbi";

/// @brief Payload name within that variant's record.
inline constexpr std::string_view kDbiEntryPayloadName = "dbi-log-buffer";

/// @brief Prologue words plus the wrapper offsets they encode.
///
/// @details The offsets are returned rather than recomputed by consumers because
/// they are baked into @ref words as immediates, so whoever writes the matching
/// `.rocjitsu.kernarg` record has something to check against.
struct DbiEntryPrologue {
  std::vector<uint32_t> words;
  uint32_t payload_byte_offset = 0;             ///< Wrapper offset of the DBI payload.
  uint32_t original_kernarg_pointer_offset = 0; ///< Wrapper offset of the guest's pointer.
};

/// @brief Build the words that run before the kernel's first original instruction.
///
/// @details Loads the DBI payload pointer into the persistent pair and restores
/// the guest's original kernarg segment pointer, which the CP replaced with a
/// pointer to the rocjitsu wrapper. The emitted sequence is laid out in the
/// implementation.
///
/// @param desc The kernel's descriptor. Its kernarg_size sizes the wrapper, and
///        it must enable ENABLE_SGPR_KERNARG_SEGMENT_PTR.
/// @param arch ISA to encode for, which sets the SMEM immediate range.
/// @param storage The run from @ref plan_dbi_entry_storage. Must be one
///        contiguous aligned run and must not overlap the kernarg segment pair.
/// @param error_out Optional; filled with the reason on failure.
/// @returns nullopt on a descriptor with no kernarg pointer, storage that is
///          misaligned, out of range, not one run, or overlapping, or a wrapper
///          whose offsets do not fit the target's SMEM immediate.
///
/// @throws util::UnimplementedInst for a non-AMDGPU architecture, matching the
///         builders it emits through. Every other rejection is fail-closed.
[[nodiscard]] std::optional<DbiEntryPrologue>
build_dbi_entry_prologue(const rocr::llvm::amdhsa::kernel_descriptor_t &desc, rj_code_arch_t arch,
                         DbiEntryStorage storage, std::string *error_out = nullptr);

/// @brief Everything a caller needs to splice a prologue into one kernel: the
///        registers it owns and the words that fill them.
struct DbiEntryProloguePlan {
  DbiEntryStorage storage;
  DbiEntryPrologue prologue;
};

/// @brief Decide whether @p desc's kernel can carry an entry prologue, and plan
///        one if it can.
///
/// @details Composes @ref plan_dbi_entry_storage and @ref build_dbi_entry_prologue
/// behind the one gate neither of them owns: kernarg preloading. Decides nothing
/// about placement, which needs the text rather than the descriptor.
///
/// @param blocks Every block reachable from the kernel entry.
/// @param desc The kernel's descriptor.
/// @param arch ISA to plan and encode for.
/// @param kernel_sgpr_count The kernel's SGPR allocation, which bounds the run.
/// @param reserved Registers the storage must avoid, such as the probe-call link
///        pairs and the probe bodies' own clobbers.
/// @param error_out Optional; filled with the reason on failure.
/// @returns nullopt when the kernel preloads kernargs, or for any reason the two
///          composed steps report.
///
/// @throws util::UnimplementedInst for a non-AMDGPU architecture, from the
///         builders @ref build_dbi_entry_prologue emits through.
[[nodiscard]] std::optional<DbiEntryProloguePlan>
plan_dbi_entry_prologue(KernelBlockScope blocks,
                        const rocr::llvm::amdhsa::kernel_descriptor_t &desc, rj_code_arch_t arch,
                        uint32_t kernel_sgpr_count, const RegisterSet &reserved,
                        std::string *error_out = nullptr);

} // namespace rocjitsu
