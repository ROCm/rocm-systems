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

/// @brief Ordinary-register and per-row reservations for native LDS observations.
struct LdsRaceProbePlan {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  uint32_t wave_size = 0;
  uint32_t scalar_base = 0;
  uint32_t vector_base = 0;
  uint32_t sgprs = 0;
  uint32_t vgprs = 0;
  uint32_t row_stride = 0;
};

/// @brief Qualify LDS scope and select sites without owning binary/CFG machinery.
[[nodiscard]] Result qualify_lds_race_entry(const InputProgram &program, InputError &error);
[[nodiscard]] Result qualify_lds_race_block(const BasicBlock &block, InputError &error);
[[nodiscard]] Result classify_lds_race_instruction(const Instruction &instruction,
                                                   rj_code_arch_t arch, uint32_t allocated_vgprs,
                                                   std::vector<NativeLdsAccess> &accesses,
                                                   InputError &error);
[[nodiscard]] FailureOr<LdsRaceProbePlan> plan_lds_race_probes(const InputProgram &program,
                                                               InputError &error);

// The coordinator places the shared DBI context bootstrap once. These emit only
// this analysis's row entry/access/exit observations and preserve guest state.
[[nodiscard]] Result emit_lds_race_entry(std::vector<uint8_t> &text, const LdsRaceProbePlan &plan,
                                         InputError &error);
[[nodiscard]] FailureOr<SiteFacts> emit_lds_race_access(std::vector<uint8_t> &text,
                                                        const LdsRaceProbePlan &plan,
                                                        const NativeLdsAccess &access,
                                                        InputError &error);
[[nodiscard]] Result emit_lds_race_exit(std::vector<uint8_t> &text, const LdsRaceProbePlan &plan,
                                        InputError &error);

/// @brief Check decoded LDS multiplicity and original bytes independently of emission.
[[nodiscard]] Result validate_lds_race_accesses(const InputProgram &program,
                                                const PreparedArtifact &artifact,
                                                std::span<BasicBlock *const> blocks,
                                                std::span<const uint8_t> prepared_text,
                                                InputError &error);

} // namespace rocjitsu::consan
