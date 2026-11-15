// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file roles.h
/// @brief How each entry location is used by the region's instructions.
///
/// @details The boundary sweep and the random policy use this to aim error-prone
/// values at the operands that are sensitive to them:
///   - shift amounts and bit indexes get 31/32/33/63/64 and values whose low bits are 0;
///   - S_BFE control words get offset/width edges;
///   - sign-extension inputs get 0x7f/0x80/0x7fff/0x8000;
///   - two registers feeding one add/sub/compare/min/max/mul/select/bitwise
///     instruction get carry, borrow, overflow, equal and complementary pairs;
///   - a register compared with or added to a constant gets c, c+-1, -c, ~c and the
///     values that cross the unsigned and signed wrap points.
/// Roles come from the decoded operands only and never change the contract.
#pragma once

#include "location.h"
#include "region.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

enum RoleBit : uint32_t {
  kRoleShiftAmount = 1u << 0,
  kRoleBitIndex = 1u << 1,
  kRoleBfeControl = 1u << 2,
  kRoleSextInput = 1u << 3,
  kRoleSignedEdge = 1u << 4,
  kRoleArith = 1u << 5,
};

/// Two register sources of one instruction (low locations for 64-bit operands).
struct OperandPair {
  unsigned a = 0;
  unsigned b = 0;
  bool wide = false;
  std::string mnemonic;
  bool operator==(const OperandPair &o) const { return a == o.a && b == o.b && wide == o.wide; }
};

/// A register source combined with a constant source in one instruction.
struct ConstRelation {
  unsigned loc = 0;
  uint32_t constant = 0;
  std::string mnemonic;
};

/// Opcode/effect summary of one region instruction (logged for later policy learning).
struct InstSummary {
  uint64_t pc = 0;
  std::string mnemonic;
  std::string text;
  LocSet reads, writes;
};

struct RegionRoles {
  std::vector<InstSummary> old_summary, new_summary;
  std::array<uint32_t, kNumLocs> roles{};
  std::vector<OperandPair> pairs;
  std::vector<ConstRelation> const_relations;
  /// Locations read by any region instruction on either side (explicit and fieldless).
  LocSet read;
  bool decoded = false; ///< False if some instruction did not decode (roles then partial).
};

/// @brief Infer roles from both sides' recorded instruction words.
/// @param new_to_old Optional location translation for the new side (inverse of a contract
/// register map), so roles, pairs and reads are keyed by the old (sample) naming.
[[nodiscard]] RegionRoles infer_roles(const std::vector<RegionInst> &old_insts,
                                      const std::vector<RegionInst> &new_insts,
                                      const std::array<unsigned, kNumLocs> *new_to_old = nullptr);

/// @brief Human-readable role names, e.g. "shift_amount,arith".
[[nodiscard]] std::string role_names(uint32_t roles);

} // namespace rocjitsu::isa_diff
