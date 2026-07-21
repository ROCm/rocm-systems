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

#include "rocjitsu/isa/arch/amdgpu/isa_properties.h"
#include "rocjitsu/isa/arch/amdgpu/shared/cdna_isa_base.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"

#include <algorithm>
#include <bit>
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
inline constexpr size_t REGISTER_SET_MAX_VGPRS = MAX_SUPPORTED_ADDRESSABLE_VGPRS_PER_WF;
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
/// @details The first three classes (SGPR, VGPR, ACC_VGPR) are ordinary
/// indexed register files tracked by `RegisterSet`. The remainder are
/// architectural special registers: they are singletons (there is one EXEC,
/// one SCC, ...), are not used for scratch allocation, and are represented by
/// `SpecialRegisterSet` rather than `RegisterSet`. `is_special_reg_class()`
/// distinguishes the two groups; keep the ordinary classes first so that
/// predicate stays a simple partition.
enum class RegClass : uint8_t {
  SGPR,         ///< Scalar general-purpose register, indexed as sN. Tracked by RegisterSet.
  VGPR,         ///< Vector general-purpose register, indexed as vN. Tracked by RegisterSet.
  ACC_VGPR,     ///< CDNA accumulator VGPR, indexed as accN. Tracked by RegisterSet.
  EXEC,         ///< EXEC mask. Special register; tracked by SpecialRegisterSet, not RegisterSet.
  VCC,          ///< VCC condition mask. Special register; tracked by SpecialRegisterSet.
  SCC,          ///< Scalar condition code bit. Special register; tracked by SpecialRegisterSet.
  M0,           ///< M0 special scalar register. Special register; tracked by SpecialRegisterSet.
  FLAT_SCRATCH, ///< Flat-scratch base pair. Special register; tracked by SpecialRegisterSet.
  TTMP,         ///< Trap-temporary registers. Special register; tracked by SpecialRegisterSet.
  PC, ///< Program counter/control-flow dep. Special register; tracked by SpecialRegisterSet.
};

/// @brief True if @p cls is an architectural special register (EXEC, VCC, SCC,
/// M0, FLAT_SCRATCH, TTMP, PC) rather than an ordinary indexed register file
/// (SGPR, VGPR, ACC_VGPR). Special registers live in `SpecialRegisterSet`;
/// ordinary ones live in `RegisterSet`.
[[nodiscard]] constexpr bool is_special_reg_class(RegClass cls) {
  switch (cls) {
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::TTMP:
  case RegClass::PC:
    return true;
  case RegClass::SGPR:
  case RegClass::VGPR:
  case RegClass::ACC_VGPR:
    return false;
  }
  return false;
}

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

  /// @brief Return the total number of single-lane registers tracked across
  ///        all classes (SGPR + VGPR + AccVGPR).
  [[nodiscard]] size_t size() const;

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

  /// @brief Invoke @p f with each tracked single-lane register in the set.
  ///
  /// @details Visits SGPRs, then VGPRs, then AccVGPRs in ascending index
  /// order. Each yielded RegisterRef has @c width=1 — multi-lane refs that
  /// were inserted via @c expand are visited as their constituent lanes.
  template <typename F> void for_each(F &&f) const {
    for (size_t i = 0; i < sgprs_.size(); ++i) {
      if (sgprs_.test(i))
        f(RegisterRef{RegClass::SGPR, static_cast<uint16_t>(i), 1});
    }
    for (size_t i = 0; i < vgprs_.size(); ++i) {
      if (vgprs_.test(i))
        f(RegisterRef{RegClass::VGPR, static_cast<uint16_t>(i), 1});
    }
    for (size_t i = 0; i < acc_vgprs_.size(); ++i) {
      if (acc_vgprs_.test(i))
        f(RegisterRef{RegClass::ACC_VGPR, static_cast<uint16_t>(i), 1});
    }
  }

private:
  std::bitset<REGISTER_SET_MAX_SGPRS> sgprs_;
  std::bitset<REGISTER_SET_MAX_VGPRS> vgprs_;
  std::bitset<REGISTER_SET_MAX_ACC_VGPRS> acc_vgprs_;
};

/// @brief A set of architectural special registers (EXEC, VCC, SCC, M0, PC,
/// FLAT_SCRATCH, TTMP).
///
/// @details Special registers are singletons — there is exactly one EXEC, one
/// SCC, and so on, with no meaningful index or width — so membership is the
/// only state, and this set stores just a class-membership bitmask rather than
/// per-index bitsets. It is deliberately a separate type from `RegisterSet`
/// so DBT and DBI can handle them accordingly.
class SpecialRegisterSet {
public:
  /// @brief Add a special-register class. Ordinary register classes
  /// (SGPR/VGPR/ACC_VGPR) are ignored — those belong in `RegisterSet`, and
  /// silently dropping them mirrors how `RegisterSet` ignores special classes.
  void insert(RegClass cls) {
    if (is_special_reg_class(cls))
      mask_ |= bit(cls);
  }

  /// @brief True if @p cls is present. Only meaningful for special classes;
  /// ordinary classes are never members and always return false.
  [[nodiscard]] bool contains(RegClass cls) const { return (mask_ & bit(cls)) != 0; }

  /// @brief True when no special register is present.
  [[nodiscard]] bool empty() const { return mask_ == 0; }

  /// @brief Number of distinct special registers present.
  [[nodiscard]] size_t size() const { return static_cast<size_t>(std::popcount(mask_)); }

  /// @brief True if any special register is present in both sets.
  [[nodiscard]] bool intersects(const SpecialRegisterSet &rhs) const {
    return (mask_ & rhs.mask_) != 0;
  }

  SpecialRegisterSet &operator|=(const SpecialRegisterSet &rhs) {
    mask_ |= rhs.mask_;
    return *this;
  }

  friend SpecialRegisterSet operator|(SpecialRegisterSet lhs, const SpecialRegisterSet &rhs) {
    lhs |= rhs;
    return lhs;
  }

  friend bool operator==(const SpecialRegisterSet &, const SpecialRegisterSet &) = default;

  /// @brief Invoke @p f with each present special register class, in ascending
  /// `RegClass` value order.
  template <typename F> void for_each(F &&f) const {
    uint16_t bits = mask_;
    while (bits != 0) {
      const auto i = static_cast<uint8_t>(std::countr_zero(bits));
      f(static_cast<RegClass>(i));
      bits &= static_cast<uint16_t>(bits - 1);
    }
  }

private:
  static constexpr uint16_t bit(RegClass cls) {
    return static_cast<uint16_t>(1u << static_cast<uint8_t>(cls));
  }

  // The bitmask indexes bits by raw RegClass value, so every class must fit in
  // `mask_`. If RegClass ever grows past the mask width, widen `mask_` (and the
  // shift base in `bit`) rather than silently truncating membership.
  static_assert(static_cast<uint8_t>(RegClass::PC) < 16,
                "SpecialRegisterSet mask_ must hold a bit for every RegClass value");

  /// @brief Bit `static_cast<uint8_t>(cls)` set iff special class `cls`
  /// present. Only special-class bits are ever set (see `insert`).
  uint16_t mask_ = 0;
};

} // namespace rocjitsu
