// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna3_lds.h
/// @brief CDNA3 emission helpers for virtualizing LDS accesses through GLOBAL memory.

#pragma once

#include "rocjitsu/code/dbt/translation_rule.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu {

/// @brief One CDNA3 GLOBAL access to a virtual-LDS backing buffer.
///
/// @details Instruction-specific lowerings own operand staging and provide an
/// even-based 64-bit VGPR address pair. This descriptor contains only the final
/// memory operation; @ref append_cdna3_virtual_lds_access owns the target-specific
/// backing-pointer protocol and completion wait.
struct Cdna3VirtualLdsAccess {
  bool is_load = false;     ///< Load from virtual LDS when true; store otherwise.
  uint8_t op = 0;           ///< CDNA3 FLAT/GLOBAL opcode.
  uint8_t data_vgpr = 0;    ///< Load destination or store source VGPR/AccVGPR.
  uint8_t address_vgpr = 0; ///< Even low VGPR of the 64-bit GLOBAL address pair.
  uint16_t byte_offset = 0; ///< Signed 13-bit FLAT/GLOBAL immediate payload.
  bool acc = false;         ///< Access data through the AccVGPR namespace.
};

/// @brief Caller-owned temporaries used while borrowing the virtual-LDS SGPR pair.
///
/// @details Descriptor-full kernels save the real backing pointer in private
/// scratch and borrow a guest SGPR pair only while forming one GLOBAL address.
/// The caller provides two VGPRs whose values are dead or already preserved and
/// a private-scratch slot that does not overlap any other live spill in the
/// surrounding replacement sequence.
struct Cdna3VirtualLdsBorrowScratch {
  uint8_t pointer_vgpr_lo = 0;
  uint8_t pointer_vgpr_hi = 0;
  uint32_t saved_sgpr_private_offset = 0;
};

/// @brief Append one complete CDNA3 virtual-LDS backing-buffer access.
///
/// @details With a persistent descriptor-backed base, the emitter zero-extends
/// the LDS byte offset in the VGPR pair and emits GLOBAL with that scalar base.
/// In spill-per-use mode it reloads the backing pointer, forms a full 64-bit VGPR
/// address, restores the borrowed guest SGPR pair, and emits GLOBAL with
/// SADDR=null. Both paths wait for the memory operation before returning.
///
/// @returns false when the address, scalar base, or borrow scratch is not encodable.
[[nodiscard]] bool append_cdna3_virtual_lds_access(
    std::vector<uint32_t> &words, const TranslationContext &context,
    const Cdna3VirtualLdsAccess &access,
    std::optional<Cdna3VirtualLdsBorrowScratch> borrow_scratch = std::nullopt);

} // namespace rocjitsu
