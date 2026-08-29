// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_barrier_move_proof.h
/// @brief Target-neutral structured-CFG proof for exact barrier relocation.

#pragma once

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class Decoder;

/// The exact control-flow witnesses retained by a structured barrier move.
struct StructuredExecDiamondProof {
  uint32_t guard_block_index = 0;
  uint32_t destination_block_index = 0;
  uint32_t source_block_index = 0;
  uint64_t guard_offset = 0;
  uint64_t destination_offset = 0;
  uint64_t source_offset = 0;
};

/// Build the pristine executable CFG used to prove or rederive a barrier move.
[[nodiscard]] std::vector<std::unique_ptr<BasicBlock>>
build_original_proof_basic_blocks(const AmdGpuCodeObject &code_object, Decoder &decoder,
                                  rj_code_arch_t arch);

/// Prove a destructive structured EXEC diamond whose restore begins the
/// source block immediately before the barrier pair.
[[nodiscard]] std::optional<StructuredExecDiamondProof>
prove_structured_exec_diamond(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                              uint32_t source_index, uint32_t destination_index,
                              uint64_t barrier_begin);

/// Prove a completing diamond whose destination guard dominates the two arms
/// that reconverge at the barrier's source block.
[[nodiscard]] std::optional<StructuredExecDiamondProof>
prove_completing_structured_diamond(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                    uint32_t source_index, uint32_t destination_index,
                                    uint64_t barrier_begin, uint64_t destination_offset);

} // namespace rocjitsu
