// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_set.h
/// @brief Register references and register sets for ISA-independent dataflow.
///
/// @details This is the foundation for the liveness design described in
/// docs/dbt_dbi_plan.md. It models all register classes that matter for DBT/DBI
/// dataflow: scalar registers, vector registers, AccVGPRs, and special
/// architectural registers such as EXEC/VCC/SCC/M0/TTMP. Keeping these in
/// disjoint register-file namespaces prevents unrelated registers with the same
/// numeric index from colliding during analysis.

#pragma once

#include "rocjitsu/isa/isa_traits.h"

#include <bitset>
#include <cstdint>
#include <optional>

namespace rocjitsu {

/// @brief ISA-independent register-file class.
///
/// @details Each class has its own namespace. For example SGPR 4 and VGPR 4 are
/// different registers, so they must not collide in the same flat bitset. The
/// enum is deliberately small and hardware-oriented; operands that are literals,
/// labels, waitcnt immediates, message IDs, and other non-register values should
/// not produce a RegisterRef.
enum class RegClass : uint8_t {
  SGPR,         ///< Scalar general-purpose register, indexed as sN.
  VGPR,         ///< Vector general-purpose register, indexed as vN.
  ACC_VGPR,     ///< CDNA accumulator VGPR, indexed as accN.
  EXEC,         ///< EXEC mask, width depends on wave size.
  VCC,          ///< VCC condition mask, width depends on wave size.
  SCC,          ///< Scalar condition code bit.
  M0,           ///< M0 special scalar register.
  FLAT_SCRATCH, ///< Flat-scratch base pair.
  TTMP,         ///< Trap-temporary registers, indexed as ttmpN.
  PC,           ///< Program counter/control-flow dependency.
};

/// @brief A contiguous register reference within one register file.
///
/// @details `index` is relative to `cls`, not a raw operand encoding value.
/// `width` is measured in 32-bit register lanes. A 64-bit SGPR pair is
/// `{RegClass::SGPR, base, 2}`; EXEC in Wave64 is `{RegClass::EXEC, 0, 2}`;
/// EXEC in Wave32 is `{RegClass::EXEC, 0, 1}`.
struct RegisterRef {
  RegClass cls;
  uint16_t index;
  uint8_t width = 1;

  constexpr bool operator==(const RegisterRef &) const = default;
};

/// @brief Per-class register set used for def/use and liveness dataflow.
///
/// @details A RegisterSet can represent an instruction's use set, def set,
/// basic-block live-in/live-out set, or live-before/live-after set. Set
/// operations are member-wise across register classes, so SGPR/VGPR/special
/// registers stay disjoint. This matters for workgroup-id rewriting: an ABI
/// value initially in a CDNA SGPR should be tracked as an SGPR value, while the
/// replacement source on RDNA4 is a TTMP register.
class RegisterSet {
public:
  /// @brief Add every 32-bit register lane covered by `ref`.
  void expand(RegisterRef ref);

  /// @brief Remove every 32-bit register lane covered by `ref`.
  void erase(RegisterRef ref);

  /// @brief Return true if every lane covered by `ref` is present.
  [[nodiscard]] bool contains(RegisterRef ref) const;

  /// @brief Return true when no register class contains any live bits.
  [[nodiscard]] bool none() const;

  /// @brief Return true if any register lane is present in both sets.
  [[nodiscard]] bool intersects(const RegisterSet &rhs) const;

  RegisterSet &operator|=(const RegisterSet &rhs);
  RegisterSet &operator&=(const RegisterSet &rhs);
  RegisterSet &operator-=(const RegisterSet &rhs);

  friend RegisterSet operator|(RegisterSet lhs, const RegisterSet &rhs) {
    lhs |= rhs;
    return lhs;
  }
  friend RegisterSet operator&(RegisterSet lhs, const RegisterSet &rhs) {
    lhs &= rhs;
    return lhs;
  }
  friend RegisterSet operator-(RegisterSet lhs, const RegisterSet &rhs) {
    lhs -= rhs;
    return lhs;
  }

  friend bool operator==(const RegisterSet &, const RegisterSet &) = default;

private:
  // Special registers are packed into one small side bitset because they have
  // different natural widths and sparse numeric encodings. The chosen ranges are
  // stable internal analysis slots, not hardware encodings:
  //   0..1   EXEC_LO/EXEC_HI
  //   2..3   VCC_LO/VCC_HI
  //   4      SCC
  //   5      M0
  //   6..7   FLAT_SCRATCH_LO/HI
  //   8..23  TTMP0..TTMP15
  //   24     PC/control dependency
  static constexpr size_t kSpecialExecBase = 0;
  static constexpr size_t kSpecialVccBase = 2;
  static constexpr size_t kSpecialSccBase = 4;
  static constexpr size_t kSpecialM0Base = 5;
  static constexpr size_t kSpecialFlatScratchBase = 6;
  static constexpr size_t kSpecialTtmpBase = 8;
  static constexpr size_t kSpecialPcBase = 24;

  [[nodiscard]] static std::optional<size_t> special_offset(RegisterRef ref);

  std::bitset<ISA_MAX_SGPRS> sgprs_;
  std::bitset<ISA_MAX_VGPRS> vgprs_;
  std::bitset<ISA_MAX_ACC_VGPRS> acc_vgprs_;
  std::bitset<ISA_MAX_SPECIAL> special_;
};

} // namespace rocjitsu
