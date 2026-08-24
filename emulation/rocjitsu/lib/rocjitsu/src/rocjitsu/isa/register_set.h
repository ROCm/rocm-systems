// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_set.h
/// @brief Register references and register sets for ISA-level register files.
///
/// @details Tracks the ordinary indexed register files (SGPR, VGPR, AccVGPR) as
/// per-index bitsets and the architectural special registers (EXEC, VCC, SCC,
/// M0, FLAT_SCRATCH, PC) as a compact singleton mask in the same set.
/// Consumers that only reason about allocatable/indexed registers (scratch
/// liveness, spilling) project the special members out with `ordinary_only()`.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
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
inline constexpr size_t REGISTER_SET_MAX_TTMPS = 16;

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
/// indexed register files with a per-index bitset. The remainder are
/// architectural special registers: they are singletons (there is one EXEC,
/// one SCC, ...), are not used for scratch allocation, and are stored in a
/// compact membership mask. `is_special_reg_class()` distinguishes the two
/// groups; keep the ordinary classes first so that predicate stays a simple
/// partition.
enum class RegClass : uint8_t {
  SGPR,         ///< Scalar general-purpose register, indexed as sN. Ordinary, per-index.
  VGPR,         ///< Vector general-purpose register, indexed as vN. Ordinary, per-index.
  ACC_VGPR,     ///< CDNA accumulator VGPR, indexed as accN. Ordinary, per-index.
  EXEC,         ///< EXEC mask. Special singleton register.
  VCC,          ///< VCC condition mask. Special singleton register.
  SCC,          ///< Scalar condition code bit. Special singleton register.
  M0,           ///< M0 special scalar register. Special singleton register.
  FLAT_SCRATCH, ///< Flat-scratch base pair. Special singleton register.
  PC,           ///< Program counter/control-flow dep. Special singleton register.
};

/// @brief True if @p cls is an architectural special register (EXEC, VCC, SCC,
/// M0, FLAT_SCRATCH, PC) rather than an ordinary indexed register file
/// (SGPR, VGPR, ACC_VGPR). Special registers are singletons held in the set's
/// special mask; ordinary ones occupy per-index bitsets.
[[nodiscard]] constexpr bool is_special_reg_class(RegClass cls) {
  switch (cls) {
  case RegClass::EXEC:
  case RegClass::VCC:
  case RegClass::SCC:
  case RegClass::M0:
  case RegClass::FLAT_SCRATCH:
  case RegClass::PC:
    return true;
  case RegClass::SGPR:
  case RegClass::VGPR:
  case RegClass::ACC_VGPR:
    return false;
  }
  return false;
}

/// @brief Widest special-class value that can set a bit in a special mask.
/// @details Derived from `is_special_reg_class` rather than a named enumerator,
/// so appending a RegClass keeps the bound correct without updating callers.
[[nodiscard]] constexpr uint8_t widest_special_reg_class() {
  uint8_t widest = 0;
  for (unsigned v = 0; v <= 0xFF; ++v) {
    if (is_special_reg_class(static_cast<RegClass>(v)))
      widest = static_cast<uint8_t>(v);
  }
  return widest;
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
/// basic-block live-in/live-out set, or live-before/live-after set. It stores
/// the ordinary indexed classes (SGPR, VGPR, AccVGPR) as per-index bitsets and
/// the architectural special registers (EXEC, VCC, ...) as a singleton
/// membership mask. Set operations are member-wise, so the ordinary classes
/// and the special mask all stay disjoint.
///
/// @details Special classes are singleton resources: `expand`, `erase`, and
/// `contains` ignore a special `RegisterRef`'s index/width and act on the
/// class as a whole, and full iteration emits a canonical `{cls, 0, 1}` ref
/// per present special class. The general APIs (`size`, `none`, `for_each`)
/// describe the full set; consumers that reason only about allocatable/indexed
/// registers use the ordinary views (`ordinary_size`, `for_each_ordinary`,
/// `ordinary_only`) so singleton special state never looks spillable or indexed.
class RegisterSet {
public:
  /// @brief Add `ref`. For ordinary classes this marks every 32-bit lane it
  /// covers; for special classes it marks the singleton (index/width ignored).
  void expand(RegisterRef ref);

  /// @brief Remove `ref`. For ordinary classes this clears every lane it
  /// covers; for special classes it clears the singleton (index/width ignored).
  void erase(RegisterRef ref);

  /// @brief Remove every register in one class (ordinary bitset or special bit).
  void clear_class(RegClass cls);

  /// @brief Return true if `ref` is present. For ordinary classes every covered
  /// lane must be present; for special classes only membership is checked.
  [[nodiscard]] bool contains(RegisterRef ref) const;

  /// @brief Return true if any lane covered by `ref` is present.
  ///
  /// @details The any-of counterpart to contains(). Register allocation asks
  /// this to reject a candidate tuple: a run is usable only when none of its
  /// lanes is in the unavailable set. Classes RegisterSet does not track answer
  /// false, matching contains().
  [[nodiscard]] bool intersects(RegisterRef ref) const;

  /// @brief Return true when neither the ordinary bitsets nor the special mask
  /// hold any member.
  [[nodiscard]] bool none() const;

  /// @brief Total number of members: ordinary single-lane registers plus
  /// distinct special singletons.
  [[nodiscard]] size_t size() const;

  /// @brief Number of ordinary single-lane registers only (SGPR + VGPR +
  /// AccVGPR), excluding special singletons. Use this for scratch/slot sizing.
  [[nodiscard]] size_t ordinary_size() const;

  /// @brief True if any special singleton is present.
  [[nodiscard]] bool has_specials() const { return special_regs_ != 0; }

  /// @brief Return true if any member is present in both sets (ordinary or special).
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

  /// @brief Return a copy holding only the ordinary members; special singletons
  /// are dropped. This is the projection scratch/liveness/spill consumers use.
  [[nodiscard]] RegisterSet ordinary_only() const {
    RegisterSet copy = *this;
    copy.special_regs_ = 0;
    return copy;
  }

  /// @brief Invoke @p f with each member of the full set.
  ///
  /// @details Visits ordinary SGPRs, VGPRs, then AccVGPRs in ascending index
  /// order (each as a @c width=1 RegisterRef), then each present special class
  /// in ascending `RegClass` order as a canonical `{cls, 0, 1}` ref.
  template <typename F> void for_each(F &&f) const {
    for_each_ordinary(f);
    for_each_special([&](RegClass cls) { f(RegisterRef{cls, 0, 1}); });
  }

  /// @brief Invoke @p f with each ordinary single-lane register (SGPR, VGPR,
  /// AccVGPR) in ascending index order. Special singletons are not visited.
  template <typename F> void for_each_ordinary(F &&f) const {
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

  /// @brief Invoke @p f with each present special class, in ascending
  /// `RegClass` value order.
  template <typename F> void for_each_special(F &&f) const {
    uint16_t bits = special_regs_;
    while (bits != 0) {
      const auto i = static_cast<uint8_t>(std::countr_zero(bits));
      f(static_cast<RegClass>(i));
      bits &= static_cast<uint16_t>(bits - 1);
    }
  }

private:
  /// @brief Mask bit for a special class. Only valid for special classes.
  static constexpr uint16_t special_bit(RegClass cls) {
    return static_cast<uint16_t>(1u << static_cast<uint8_t>(cls));
  }

  // The special mask indexes bits by raw RegClass value, so every special class
  // must fit in `special_regs_`. If a special class ever grows past the mask
  // width, widen it (and the shift base in `special_bit`) rather than
  // truncating membership.
  static_assert(widest_special_reg_class() < 16,
                "special_regs_ must hold a bit for every special RegClass");

  std::bitset<REGISTER_SET_MAX_SGPRS> sgprs_;
  std::bitset<REGISTER_SET_MAX_VGPRS> vgprs_;
  std::bitset<REGISTER_SET_MAX_ACC_VGPRS> acc_vgprs_;

  /// @brief Bit `static_cast<uint8_t>(cls)` set iff special class `cls` is
  /// present. Only special-class bits are ever set (see `expand`).
  uint16_t special_regs_ = 0;
};

} // namespace rocjitsu
