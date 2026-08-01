// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_capability_contract.h
/// @brief Typed high-level ConSan semantic capability projection.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

/// Stable engine identities used by the public capability contract. MOI keeps
/// its implementation enum separate because SuperCollider is not an MOI
/// engine.
enum class ConSanCapabilityEngine : uint8_t {
  SuperCollider,
  RecordReplay,
  Sampled,
  InlineShadow,
  Count,
};

enum class ConSanCapabilityDomain : uint8_t {
  Access,
  Barrier,
  Atomic,
  Fence,
  Count,
};

/// Semantic forms are intentionally coarser than ISA mnemonics. Target-native
/// spellings enter the decoded inventory before this projection applies.
enum class ConSanCapabilityForm : uint8_t {
  NativeLdsAccess,
  GroupFlatAccess,
  WorkgroupBarrier,
  ClusterBarrier,
  OrderedFlatAtomic,
  OrderedVglobalAtomic,
  OrderedLdsAtomic,
  RelaxedLdsAtomicAccess,
  AddressedOrdinaryFence,
  Count,
};

/// Supported means causal evidence for MOI engines or redundant access
/// observation for SuperCollider. The other dispositions make intentionally
/// weaker engine semantics explicit instead of presenting them as omissions.
enum class ConSanCapabilityDisposition : uint8_t {
  OutOfContract,
  NotApplicable,
  Unsupported,
  Supported,
  MutationOnly,
  AccessOnly,
  AssociatedOnly,
};

struct ConSanCapabilityTarget {
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
};

/// The production target-admission map and the documentation iteration order
/// share this table. Adding support for a target therefore cannot leave the
/// public matrix on a separate target list.
inline constexpr std::array<ConSanCapabilityTarget, 5> kConSanCapabilityTargets = {{
    {ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_ARCH_CDNA3},
    {ROCJITSU_CODE_TARGET_GFX950, ROCJITSU_CODE_ARCH_CDNA4},
    {ROCJITSU_CODE_TARGET_GFX1100, ROCJITSU_CODE_ARCH_RDNA3},
    {ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_ARCH_RDNA4},
    {ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_ARCH_GFX1250},
}};

inline constexpr std::array<ConSanCapabilityEngine, 4> kConSanCapabilityEngines = {
    ConSanCapabilityEngine::SuperCollider,
    ConSanCapabilityEngine::RecordReplay,
    ConSanCapabilityEngine::Sampled,
    ConSanCapabilityEngine::InlineShadow,
};

inline constexpr std::array<ConSanCapabilityDomain, 4> kConSanCapabilityDomains = {
    ConSanCapabilityDomain::Access,
    ConSanCapabilityDomain::Barrier,
    ConSanCapabilityDomain::Atomic,
    ConSanCapabilityDomain::Fence,
};

inline constexpr std::array<ConSanCapabilityForm, 9> kConSanCapabilityForms = {
    ConSanCapabilityForm::NativeLdsAccess,        ConSanCapabilityForm::GroupFlatAccess,
    ConSanCapabilityForm::WorkgroupBarrier,       ConSanCapabilityForm::ClusterBarrier,
    ConSanCapabilityForm::OrderedFlatAtomic,      ConSanCapabilityForm::OrderedVglobalAtomic,
    ConSanCapabilityForm::OrderedLdsAtomic,       ConSanCapabilityForm::RelaxedLdsAtomicAccess,
    ConSanCapabilityForm::AddressedOrdinaryFence,
};

template <typename Enum, std::size_t N>
[[nodiscard]] constexpr bool consan_capability_enum_is_complete(const std::array<Enum, N> &values) {
  constexpr std::size_t count = static_cast<std::size_t>(Enum::Count);
  if (N != count)
    return false;
  std::array<bool, count> seen{};
  for (Enum value : values) {
    const std::size_t index = static_cast<std::size_t>(value);
    if (index >= count || seen[index])
      return false;
    seen[index] = true;
  }
  return true;
}

[[nodiscard]] constexpr bool consan_capability_targets_are_unique() {
  for (std::size_t lhs = 0; lhs < kConSanCapabilityTargets.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < kConSanCapabilityTargets.size(); ++rhs) {
      if (kConSanCapabilityTargets[lhs].target == kConSanCapabilityTargets[rhs].target)
        return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr rj_code_arch_t consan_arch_for_target(rj_code_target_id_t target) {
  for (const ConSanCapabilityTarget &supported : kConSanCapabilityTargets) {
    if (target == supported.target)
      return supported.arch;
  }
  return ROCJITSU_CODE_ARCH_INVALID;
}

[[nodiscard]] constexpr bool consan_is_capability_target(rj_code_target_id_t target) {
  return consan_arch_for_target(target) != ROCJITSU_CODE_ARCH_INVALID;
}

/// RDNA ConSan probes use the code-object dispatch identity literal instead
/// of reserving a guest SGPR pair. This is an instrumentation policy, not an
/// instruction-encoding property.
[[nodiscard]] constexpr bool consan_arch_uses_literal_dispatch_identity(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA4 ||
         arch == ROCJITSU_CODE_ARCH_GFX1250;
}

[[nodiscard]] constexpr ConSanCapabilityDomain consan_capability_domain(ConSanCapabilityForm form) {
  switch (form) {
  case ConSanCapabilityForm::NativeLdsAccess:
  case ConSanCapabilityForm::GroupFlatAccess:
    return ConSanCapabilityDomain::Access;
  case ConSanCapabilityForm::WorkgroupBarrier:
  case ConSanCapabilityForm::ClusterBarrier:
    return ConSanCapabilityDomain::Barrier;
  case ConSanCapabilityForm::OrderedFlatAtomic:
  case ConSanCapabilityForm::OrderedVglobalAtomic:
  case ConSanCapabilityForm::OrderedLdsAtomic:
  case ConSanCapabilityForm::RelaxedLdsAtomicAccess:
    return ConSanCapabilityDomain::Atomic;
  case ConSanCapabilityForm::AddressedOrdinaryFence:
    return ConSanCapabilityDomain::Fence;
  case ConSanCapabilityForm::Count:
    return ConSanCapabilityDomain::Count;
  }
  return ConSanCapabilityDomain::Count;
}

/// Target-family availability shared with production admission. Exact
/// mnemonic, encoding, scope, and operand checks remain in the decoded
/// inventory and lowerers.
[[nodiscard]] constexpr bool consan_arch_supports_capability_form(rj_code_arch_t arch,
                                                                  ConSanCapabilityForm form) {
  const bool supported_arch =
      arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
      arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA4 ||
      arch == ROCJITSU_CODE_ARCH_GFX1250;
  if (!supported_arch)
    return false;

  switch (form) {
  case ConSanCapabilityForm::NativeLdsAccess:
  case ConSanCapabilityForm::GroupFlatAccess:
  case ConSanCapabilityForm::WorkgroupBarrier:
  case ConSanCapabilityForm::OrderedFlatAtomic:
  case ConSanCapabilityForm::OrderedVglobalAtomic:
  case ConSanCapabilityForm::AddressedOrdinaryFence:
    return true;
  case ConSanCapabilityForm::ClusterBarrier:
  case ConSanCapabilityForm::OrderedLdsAtomic:
    return arch == ROCJITSU_CODE_ARCH_GFX1250;
  case ConSanCapabilityForm::RelaxedLdsAtomicAccess:
    return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
           arch == ROCJITSU_CODE_ARCH_GFX1250;
  case ConSanCapabilityForm::Count:
    return false;
  }
  return false;
}

/// Return the documented high-level disposition. Exact instruction admission
/// still comes from the decoded semantic inventory and engine lowerers; this
/// function projects their stable target/engine contract without duplicating
/// target-native mnemonic lists.
[[nodiscard]] constexpr ConSanCapabilityDisposition
consan_capability_disposition(rj_code_target_id_t target, ConSanCapabilityEngine engine,
                              ConSanCapabilityForm form) {
  const rj_code_arch_t arch = consan_arch_for_target(target);
  if (arch == ROCJITSU_CODE_ARCH_INVALID || engine == ConSanCapabilityEngine::Count ||
      form == ConSanCapabilityForm::Count)
    return ConSanCapabilityDisposition::OutOfContract;
  if (!consan_arch_supports_capability_form(arch, form))
    return ConSanCapabilityDisposition::NotApplicable;

  const bool supercollider = engine == ConSanCapabilityEngine::SuperCollider;

  switch (form) {
  case ConSanCapabilityForm::NativeLdsAccess:
  case ConSanCapabilityForm::GroupFlatAccess:
    return ConSanCapabilityDisposition::Supported;
  case ConSanCapabilityForm::WorkgroupBarrier:
    return supercollider ? ConSanCapabilityDisposition::MutationOnly
                         : ConSanCapabilityDisposition::Supported;
  case ConSanCapabilityForm::ClusterBarrier:
    return supercollider ? ConSanCapabilityDisposition::MutationOnly
                         : ConSanCapabilityDisposition::Supported;
  case ConSanCapabilityForm::OrderedFlatAtomic:
    return supercollider ? ConSanCapabilityDisposition::MutationOnly
                         : ConSanCapabilityDisposition::Supported;
  case ConSanCapabilityForm::OrderedVglobalAtomic:
    return supercollider ? ConSanCapabilityDisposition::MutationOnly
                         : ConSanCapabilityDisposition::Supported;
  case ConSanCapabilityForm::OrderedLdsAtomic:
    return supercollider ? ConSanCapabilityDisposition::MutationOnly
                         : ConSanCapabilityDisposition::Supported;
  case ConSanCapabilityForm::RelaxedLdsAtomicAccess:
    return supercollider ? ConSanCapabilityDisposition::NotApplicable
                         : ConSanCapabilityDisposition::AccessOnly;
  case ConSanCapabilityForm::AddressedOrdinaryFence:
    if (supercollider)
      return ConSanCapabilityDisposition::MutationOnly;
    return engine == ConSanCapabilityEngine::RecordReplay
               ? ConSanCapabilityDisposition::Supported
               : ConSanCapabilityDisposition::AssociatedOnly;
  case ConSanCapabilityForm::Count:
    return ConSanCapabilityDisposition::OutOfContract;
  }
  return ConSanCapabilityDisposition::OutOfContract;
}

[[nodiscard]] constexpr std::string_view
consan_capability_engine_name(ConSanCapabilityEngine engine) {
  switch (engine) {
  case ConSanCapabilityEngine::SuperCollider:
    return "SuperCollider";
  case ConSanCapabilityEngine::RecordReplay:
    return "Record/Replay";
  case ConSanCapabilityEngine::Sampled:
    return "Sampled";
  case ConSanCapabilityEngine::InlineShadow:
    return "Inline Shadow";
  case ConSanCapabilityEngine::Count:
    return "unknown";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view consan_capability_form_name(ConSanCapabilityForm form) {
  switch (form) {
  case ConSanCapabilityForm::NativeLdsAccess:
    return "native LDS";
  case ConSanCapabilityForm::GroupFlatAccess:
    return "group FLAT";
  case ConSanCapabilityForm::WorkgroupBarrier:
    return "workgroup";
  case ConSanCapabilityForm::ClusterBarrier:
    return "cluster";
  case ConSanCapabilityForm::OrderedFlatAtomic:
    return "ordered FLAT";
  case ConSanCapabilityForm::OrderedVglobalAtomic:
    return "ordered VGLOBAL";
  case ConSanCapabilityForm::OrderedLdsAtomic:
    return "ordered LDS";
  case ConSanCapabilityForm::RelaxedLdsAtomicAccess:
    return "relaxed LDS RMW";
  case ConSanCapabilityForm::AddressedOrdinaryFence:
    return "addressed ordinary";
  case ConSanCapabilityForm::Count:
    return "unknown";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view
consan_capability_disposition_name(ConSanCapabilityDisposition disposition) {
  switch (disposition) {
  case ConSanCapabilityDisposition::OutOfContract:
    return "out of contract";
  case ConSanCapabilityDisposition::NotApplicable:
    return "not applicable";
  case ConSanCapabilityDisposition::Unsupported:
    return "unsupported";
  case ConSanCapabilityDisposition::Supported:
    return "supported";
  case ConSanCapabilityDisposition::MutationOnly:
    return "mutation only";
  case ConSanCapabilityDisposition::AccessOnly:
    return "access only";
  case ConSanCapabilityDisposition::AssociatedOnly:
    return "associated only";
  }
  return "unknown";
}

static_assert(consan_capability_targets_are_unique());
static_assert(consan_capability_enum_is_complete(kConSanCapabilityEngines));
static_assert(consan_capability_enum_is_complete(kConSanCapabilityDomains));
static_assert(consan_capability_enum_is_complete(kConSanCapabilityForms));

} // namespace rocjitsu
