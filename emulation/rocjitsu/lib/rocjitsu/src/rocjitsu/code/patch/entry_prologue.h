// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file entry_prologue.h
/// @brief Framework-reserved SGPR storage for the DBI kernel-entry prologue.

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cstdint>
#include <optional>
#include <string>

namespace rocjitsu {

/// @brief SGPRs the entry prologue reserves: a persistent pair plus an
///        entry-only temporary pair, allocated as one aligned run.
///
/// @details Sizing the run at four rather than two is what keeps the guest
/// kernarg restore removable without re-opening storage selection.
inline constexpr uint8_t kDbiEntryStorageRegisters = 4;

/// @brief SGPR pairs the DBI entry prologue owns for the life of the kernel.
struct DbiEntryStorage {
  uint16_t persistent_base = 0; ///< s[base:base+1], written at entry, read at every site.
  uint16_t entry_temp_base = 0; ///< s[base+2:base+3], live only inside the prologue.
};

/// @brief Lowest SGPR index above everything the kernel's own code and ABI name:
///        @ref explicit_ordinary_sgpr_bound folded with the descriptor's
///        user-SGPR block and its enabled workgroup-ID system SGPRs.
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

} // namespace rocjitsu
