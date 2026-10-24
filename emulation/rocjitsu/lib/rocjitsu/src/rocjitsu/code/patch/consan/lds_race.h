// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/rj_code.h"

#include <span>
#include <vector>

namespace rocjitsu {
class BasicBlock;
class Instruction;
} // namespace rocjitsu

namespace rocjitsu::consan {
struct InputError;
struct InputProgram;
struct PreparedArtifact;

/// @brief A decoder-qualified physical access; addresses remain runtime operands.
struct NativeLdsAccess {
  uint32_t site = 0;
  uint64_t original_offset = 0;
  AccessKind kind = AccessKind::Read;
  uint16_t address_vgpr = 0;
  uint16_t immediate_bytes = 0;
};

/// @brief Qualify LDS scope and select sites without owning binary/CFG machinery.
[[nodiscard]] Result qualify_lds_race_entry(const InputProgram &program, InputError &error);
[[nodiscard]] Result qualify_lds_race_block(const BasicBlock &block, InputError &error);
[[nodiscard]] Result classify_lds_race_instruction(const Instruction &instruction,
                                                   rj_code_arch_t arch, uint32_t allocated_vgprs,
                                                   std::vector<NativeLdsAccess> &accesses,
                                                   InputError &error);
[[nodiscard]] FailureOr<uint32_t> lds_race_row_stride(size_t sites, InputError &error,
                                                      bool barriers = false);

[[nodiscard]] Result qualify_lds_race_barrier(InputProgram &program,
                                              std::span<BasicBlock *const> blocks,
                                              InputError &error);

/// @brief Check decoded LDS multiplicity and original bytes independently of emission.
[[nodiscard]] Result validate_lds_race_accesses(const InputProgram &program,
                                                const PreparedArtifact &artifact,
                                                std::span<BasicBlock *const> blocks,
                                                std::span<const uint8_t> prepared_text,
                                                InputError &error);

} // namespace rocjitsu::consan
