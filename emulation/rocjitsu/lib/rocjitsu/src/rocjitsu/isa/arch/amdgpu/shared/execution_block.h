// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_block.h
/// @brief The AMDGPU block an instruction issues to.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_EXECUTION_BLOCK_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_EXECUTION_BLOCK_H_

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief The AMDGPU block an instruction issues to, as returned by
/// Instruction::amdgpu_execution_block().
/// @details A static, decode-time classification for consumers that model the
/// issue path, such as a timing model. It is derived from the encoding-class
/// flags (SALU, VALU, SMEM, VMEM, LDS in instruction.h) with one policy
/// override: every instruction that may write the PC (Instruction::writes_pc())
/// reports BRANCH instead of SALU. That covers direct, conditional and indirect
/// branches and calls, s_endpgm, trap entry and return (s_trap, s_rfe*) and the
/// debug conditional branches (s_cbranch_cdbg*). The SALU encoding flag remains
/// in Instruction::flags() when the override is not wanted.
///
/// It does not see address routing, so generic FLAT is VMEM even when its
/// address resolves to LDS, and it does not separate matrix from vector work.
/// The reporting plugins' plugins::classify_instruction() answers a different
/// question -- which family a report buckets an executed instruction into --
/// and consults execution state and mnemonics to keep its JSONL stable across
/// architectures; use that for reports and this for the issue path.
enum class ExecutionBlock : uint8_t {
  UNKNOWN, ///< In no block below, e.g. exports and non-AMDGPU instructions.
  SALU,    ///< Scalar ALU, including waits, barriers, messages and no-ops.
  VALU,    ///< Vector ALU, including matrix, transcendental and VOPD operations.
  BRANCH,  ///< Scalar-encoded instructions that may write the PC; see above.
  SMEM,    ///< Scalar memory and scalar cache maintenance.
  VMEM,    ///< Vector memory: buffer, global, scratch, flat, image and tensor.
  LDS,     ///< LDS, including LDS-direct, GWS and the ds_ no-op.
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_EXECUTION_BLOCK_H_
