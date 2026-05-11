// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_set.h
/// @brief Register references and register sets for ISA-level register files.
///
/// @details Tracks scalar registers, vector registers, and AccVGPRs. Other
/// architectural register classes may appear in decoded metadata, but they are
/// ignored by RegisterSet until there is a concrete consumer for
/// special-register liveness.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>

namespace rocjitsu {

/// @brief RegisterSet storage capacities, derived from AMDGPU family traits.
///
/// @details These are storage bounds for an ISA-independent analysis set, not
/// per-kernel allocation limits. Wave32 vs. Wave64 changes lane count, not the
/// number of SGPR/VGPR indices addressable within a wavefront register file.
inline constexpr size_t REGISTER_SET_MAX_SGPRS =
    std::max<size_t>(amdgpu::CdnaIsaBase::MAX_SGPRS_PER_WF, amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF);
inline constexpr size_t REGISTER_SET_MAX_VGPRS =
    std::max<size_t>(amdgpu::CdnaIsaBase::MAX_VGPRS_PER_WF, amdgpu::RdnaIsaBase::MAX_VGPRS_PER_WF);
inline constexpr size_t REGISTER_SET_MAX_ACC_VGPRS = REGISTER_SET_MAX_VGPRS;

/// @brief Normal SGPRs safe for scratch allocation across supported families.
///
/// @details CDNA exposes 102 ordinary SGPRs per wavefront while RDNA exposes
/// 106. Liveness itself tracks the union, but generic scratch selection must be
/// conservative unless it is made target-ISA-specific.
inline constexpr size_t REGISTER_SET_ALLOCATABLE_SGPRS =
    std::min<size_t>(amdgpu::CdnaIsaBase::MAX_SGPRS_PER_WF, amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF);

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
  EXEC,         ///< EXEC mask. Not currently tracked by RegisterSet.
  VCC,          ///< VCC condition mask. Not currently tracked by RegisterSet.
  SCC,          ///< Scalar condition code bit. Not currently tracked by RegisterSet.
  M0,           ///< M0 special scalar register. Not currently tracked by RegisterSet.
  FLAT_SCRATCH, ///< Flat-scratch base pair. Not currently tracked by RegisterSet.
  TTMP,         ///< Trap-temporary registers. Not currently tracked by RegisterSet.
  PC,           ///< Program counter/control-flow dependency. Not currently tracked by RegisterSet.
};

/// @brief A contiguous register reference within one register file.
///
/// @details `index` is relative to `cls`, not a raw operand encoding value.
/// `width` is measured in 32-bit register lanes. A 64-bit SGPR pair is
/// `{RegClass::SGPR, base, 2}`. The current MR ISA max tracked operand width
/// is 32 lanes (1024-bit MFMA accumulator operands), so uint8_t has ample room.
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
/// operations are member-wise across tracked register classes, so SGPR, VGPR,
/// and AccVGPR lanes stay disjoint.
class RegisterSet {
public:
  /// @brief Add every 32-bit register lane covered by `ref`.
  void expand(RegisterRef ref);

  /// @brief Remove every 32-bit register lane covered by `ref`.
  void erase(RegisterRef ref);

  /// @brief Remove all tracked registers in one register class.
  void clear_class(RegClass cls);

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
  std::bitset<REGISTER_SET_MAX_SGPRS> sgprs_;
  std::bitset<REGISTER_SET_MAX_VGPRS> vgprs_;
  std::bitset<REGISTER_SET_MAX_ACC_VGPRS> acc_vgprs_;
};

} // namespace rocjitsu
