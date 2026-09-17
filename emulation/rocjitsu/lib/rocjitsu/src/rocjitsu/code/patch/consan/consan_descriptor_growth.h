// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_descriptor_growth.h
/// @brief Application of owner-keyed descriptor mutation transactions.

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
class AmdGpuCodeObject;
class CodeObjectPatcher;
} // namespace rocjitsu

namespace rocjitsu::consan {

class ProgramInventory;

using DescriptorRegisterRequirements = std::unordered_map<uint64_t, uint16_t>;
using DescriptorMemoryRequirements = std::unordered_map<uint64_t, uint32_t>;

/// Non-owning description of every descriptor field one lowering transaction
/// may grow. Keys are immutable ProgramInventory descriptor identities; the
/// mutation owner resolves their current offsets after any earlier image move.
struct DescriptorMutationBatch {
  const DescriptorRegisterRequirements &vgprs;
  const DescriptorRegisterRequirements &sgprs;
  const DescriptorMemoryRequirements &private_segment_bytes;
  const DescriptorMemoryRequirements &group_segment_bytes;
};

/// Caller-specific constraints that are not descriptor mutation mechanics.
/// ConSan and SuperCollider can therefore share the owner and transaction while
/// retaining their distinct operand limits and accumulator-bank proof.
struct DescriptorMutationPolicy {
  uint32_t maximum_ordinary_vgpr_count = 0;
  bool inventory_proves_empty_accumulator_bank = false;
  std::optional<uint32_t> maximum_group_segment_bytes;
};

[[nodiscard]] bool
apply_descriptor_mutations_to_patcher(CodeObjectPatcher &patcher, const ProgramInventory &inventory,
                                      const AmdGpuCodeObject &active_code_object,
                                      const DescriptorMutationBatch &batch,
                                      const DescriptorMutationPolicy &policy, rj_code_arch_t arch,
                                      std::string_view subject, std::vector<std::string> &errors);

[[nodiscard]] bool
apply_descriptor_mutations_to_bytes(std::span<uint8_t> image, const ProgramInventory &inventory,
                                    const DescriptorMutationBatch &batch,
                                    const DescriptorMutationPolicy &policy, rj_code_arch_t arch,
                                    std::string_view subject, std::vector<std::string> &errors);

} // namespace rocjitsu::consan
