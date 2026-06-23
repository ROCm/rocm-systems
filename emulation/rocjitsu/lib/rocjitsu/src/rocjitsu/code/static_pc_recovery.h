// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file static_pc_recovery.h
/// @brief Static indirect branch PC recovery helpers for block construction and text relocation.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {

class Instruction;
struct KernelTextLayout;

/// @brief Recovered indirect PC-relative branch through a statically-built PC register.
///
/// @details BasicBlock construction turns the recovered destination into an
/// ordinary CFG successor from the block that contains the setpc/swappc
/// consumer. DBT keeps this metadata so relocation can rewrite the original
/// getpc-relative address-builder range in place after the block's final target
/// offsets are known.
struct IndirectCallFixup {
  uint64_t source_getpc_offset = 0;          ///< Source offset of the s_getpc_b64 producer.
  uint64_t source_recovery_begin_offset = 0; ///< First source byte of replaceable builder code.
  uint64_t source_recovery_end_offset = 0;   ///< One-past-end source byte of builder code.
  uint64_t source_call_offset = 0;           ///< Source offset of the setpc/swappc consumer.
  uint64_t source_target_offset = 0;         ///< Recovered source branch target offset.
  uint16_t source_call_sreg = 0;             ///< Low SGPR of the recovered PC pair.
  uint64_t target_getpc_offset = 0;          ///< Relocated offset of the s_getpc_b64 producer.
  uint64_t target_recovery_begin_offset = 0; ///< Relocated first byte of replaceable builder code.
  uint64_t target_recovery_end_offset = 0;   ///< Relocated one-past-end byte of builder code.
};

/// @brief Recover statically-materialized indirect branch targets from decoded text.
///
/// @details This scan is intentionally structural: it recognizes known
/// s_getpc-relative PC builder sequences and records the later setpc/swappc
/// consumers that use the built PC register pair. BasicBlock owns CFG edge
/// placement; DBT owns later relocation of the recorded builder range.
[[nodiscard]] std::vector<IndirectCallFixup>
recover_static_indirect_call_targets(std::span<const Instruction *const> insts,
                                     std::span<const uint64_t> offsets,
                                     std::span<const uint8_t> text, rj_code_arch_t arch);

/// @brief Reuse the latest recovered target for a later branch through a carried PC register.
[[nodiscard]] std::optional<IndirectCallFixup>
recover_carried_indirect_call_target(const KernelTextLayout &layout, const Instruction &inst,
                                     uint32_t word, uint64_t source_call_offset,
                                     rj_code_arch_t arch);

/// @brief Return whether a setpc instruction is a known direct-call return.
[[nodiscard]] bool recovered_direct_call_return(const KernelTextLayout &layout,
                                                const Instruction &inst, uint32_t word,
                                                uint64_t source_offset, rj_code_arch_t arch);

/// @brief Return whether @p inst is a setpc from @p ssrc0 on @p arch.
[[nodiscard]] bool s_setpc_from_sreg(rj_code_arch_t arch, const Instruction &inst, uint32_t word,
                                     uint16_t ssrc0);

/// @brief Match an indirect scalar call and return its destination SGPR pair.
///
/// @details The generated ISA model classifies indirect calls and reports
/// whether the instruction carries a PC-relative branch offset. This helper
/// uses those structural properties plus the encoded SDST field, keeping DBT
/// recovery independent from instruction names.
[[nodiscard]] std::optional<uint16_t> s_call_sdst(const Instruction &inst, uint32_t word);

} // namespace rocjitsu
