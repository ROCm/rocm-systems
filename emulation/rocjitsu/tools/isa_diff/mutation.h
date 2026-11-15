// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mutation.h
/// @brief Synthetic error injection: a validator self-test, never an LLVM finding.
///
/// @details A mutation XORs a mask into one recorded instruction word on one side of a
/// region; typical uses are a wrong opcode, a wrong immediate or a wrong operand. The
/// mutated word must still decode to the same size and to a P0 instruction. That side's
/// declared writes are re-derived from the decode. The entry and observable sets are
/// kept from the real analysis, so the kernel context is unchanged. Every resulting
/// region, result and counterexample is labelled synthetic.
#pragma once

#include "region.h"

#include <cstdint>
#include <optional>
#include <string>

namespace rocjitsu::isa_diff {

struct MutationSpec {
  bool new_side = true;
  size_t inst = 0;
  size_t word = 0;
  uint32_t xor_mask = 0;
  std::string label = "mutation";
};

[[nodiscard]] std::optional<RegionPair>
make_synthetic_mutation(const RegionPair &base, const MutationSpec &spec, std::string &error);

} // namespace rocjitsu::isa_diff
