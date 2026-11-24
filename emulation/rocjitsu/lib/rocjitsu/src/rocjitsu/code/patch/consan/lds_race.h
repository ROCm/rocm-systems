// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/register_set.h"

#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {
class BasicBlock;
class Instruction;
} // namespace rocjitsu

namespace rocjitsu::consan {
struct InputError;
struct InputProgram;

/// @brief A decoder-qualified physical access; addresses remain runtime operands.
struct NativeLdsAccess {
  uint32_t site = 0;
  uint64_t original_offset = 0;
  AccessKind kind = AccessKind::Read;
  uint16_t address_vgpr = 0;
  uint16_t immediate_bytes = 0;
  // Byte count set by decoding; zero is unspecified and cannot qualify a site.
  uint32_t width = 0;
  LdsRangeRecipe lds_ranges{};
  AccessOrigin origin = AccessOrigin::Native;
};

/// @brief Qualify LDS scope and select sites without owning binary/CFG machinery.
[[nodiscard]] Result qualify_lds_race_entry(const InputProgram &program, InputError &error);
[[nodiscard]] Result qualify_lds_race_block(const BasicBlock &block, InputError &error);

/// @brief Qualify original workgroup barrier groups and populate program.barriers.
/// @details Require an original LDS drain immediately before signal and a matching
/// wait in the same block, with no intervening memory or synchronization. Acyclic
/// groups must dominate exits; the qualified single-block loop uses repeated epochs.
[[nodiscard]] Result qualify_lds_race_barrier(InputProgram &program,
                                              std::span<BasicBlock *const> blocks,
                                              InputError &error);

/// @brief Order the reachable CFG, deferring conditional self-edges to barrier qualification.
/// @details Reject all other cycles. The input collector uses this order to preserve
/// increasing native LDS access PCs along paths with backward trampoline edges.
/// A deferred self-edge is admitted only if qualify_lds_race_barrier() later succeeds.
[[nodiscard]] FailureOr<std::vector<const BasicBlock *>>
qualify_lds_race_cfg(std::span<const std::unique_ptr<BasicBlock>> blocks, InputError &error);
[[nodiscard]] Result
classify_lds_race_instruction(const Instruction &instruction, rocjitsu::IsaFamily arch,
                              uint32_t allocated_vgprs, std::vector<NativeLdsAccess> &accesses,
                              InputError &error, const RegisterSet *shared_high = nullptr);
[[nodiscard]] FailureOr<uint32_t> lds_race_row_stride(size_t sites, InputError &error,
                                                      bool has_barriers);

/// @brief Check decoded LDS multiplicity and original bytes independently of emission.
[[nodiscard]] Result validate_lds_race_accesses(const InputProgram &program,
                                                const Manifest &manifest,
                                                std::span<BasicBlock *const> blocks,
                                                std::span<const uint8_t> prepared_text,
                                                InputError &error);

} // namespace rocjitsu::consan
