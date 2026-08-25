// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_capability_contract.h
/// @brief Typed high-level ConSan semantic capability projection.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

/// Names the four user-visible ConSan engines represented by the capability
/// matrix.
///
/// This is a documentation and capability-contract identity, not the runtime
/// engine-selection type. In particular, the MOI implementation keeps its own
/// enum because SuperCollider is a ConSan engine but is not an MOI engine.
/// Keeping this projection separate lets the public matrix describe all four
/// engines without making an implementation layer depend on an engine it does
/// not own. `Count` is an iteration sentinel and never denotes an engine.
enum class ConSanCapabilityEngine : uint8_t {
  SuperCollider,
  RecordReplay,
  Sampled,
  InlineShadow,
  Count,
};

/// Groups normalized capability forms by the kind of program behavior they
/// describe.
///
/// A domain is deliberately broader than either an ISA instruction class or a
/// GPU address space. For example, native LDS and group-FLAT operations are
/// different instruction forms but both belong to the access domain. Domains
/// exist so generated documentation and contract checks can organize forms;
/// they do not decide whether an individual instruction is supported and must
/// not be used as a substitute for decoded-instruction analysis. `Count` is an
/// iteration sentinel and is not a semantic domain.
enum class ConSanCapabilityDomain : uint8_t {
  Access,
  Barrier,
  Atomic,
  Fence,
  Count,
};

/// Identifies a target-independent GPU operation shape whose ConSan behavior
/// is described by the public capability contract.
///
/// Forms are intentionally coarser than ISA mnemonics: target-native
/// instruction spellings and operand encodings first enter the exact decoded
/// inventory, and only then project onto one of these forms. A form therefore
/// records the behavioral question being asked (for example, an ordered LDS
/// atomic), while the decoder and lowerer retain responsibility for exact
/// instruction admission. `Count` is an iteration sentinel and never denotes
/// an operation.
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

/// Identifies the architectural product lineage to which a supported target
/// belongs.
///
/// Product lineage captures architectural intent such as CDNA versus RDNA; it
/// does not imply that two members use the same machine-instruction encoding.
/// Conversely, targets from different product lineages may share encodings:
/// gfx1250 is CDNA even though it shares the gfx12 encoding family with RDNA4.
/// Callers that need to choose an instruction layout must use
/// `ConSanEncodingFamily`, not this type.
enum class ConSanArchitectureFamily : uint8_t {
  Cdna,
  Rdna,
};

/// Selects the machine-instruction encoding rules used by ConSan decoders and
/// lowerers.
///
/// An encoding family is a statement about compatible binary layouts and
/// instruction-building facilities, not product branding or the complete set
/// of semantic capabilities. The two gfx9 CDNA generations remain distinct
/// because some exact encodings differ, while RDNA4 and CDNA5 intentionally
/// share `Gfx12`. Exact target exceptions may still be handled after this
/// coarse routing decision.
enum class ConSanEncodingFamily : uint8_t {
  Gfx9Cdna3,
  Gfx9Cdna4,
  Gfx11,
  Gfx12,
};

/// Describes how accumulator registers participate in a kernel's physical
/// vector-register allocation.
///
/// `None` means ConSan does not need a separate accumulator allocation model.
/// `DescriptorPartitioned` means a descriptor field divides one allocation
/// into ordinary VGPR and AccVGPR regions. `SelectableVgprBank` means the ISA
/// selects among physical VGPR banks rather than exposing the gfx9 descriptor
/// partition. Resource planning uses this type to interpret allocation facts;
/// it does not by itself select an instrumentation strategy.
enum class ConSanAccumulatorModel : uint8_t {
  None,
  DescriptorPartitioned,
  SelectableVgprBank,
};

/// Identifies the target-provided facility from which instrumentation can
/// obtain a stable dispatch identity.
///
/// `PreloadedSgprPair` requires the descriptor/prologue path to make a pair of
/// scalar registers available. `CodeObjectLiteral` means the instrumenter can
/// materialize the identity from immutable code-object data instead. This enum
/// records availability, not policy: an engine may choose not to consume the
/// facility even when the target provides it.
enum class ConSanDispatchIdentitySource : uint8_t {
  PreloadedSgprPair,
  CodeObjectLiteral,
};

/// Identifies the architectural source of the workgroup coordinates consumed
/// by ConSan probes.
///
/// Some targets expose coordinates through descriptor-enabled system SGPRs;
/// others make them available through command-processor temporary registers.
/// The distinction controls prologue and resource planning and must not be
/// inferred from CDNA/RDNA product lineage. This type describes the source
/// facility only, not the higher-level owner key assembled from it.
enum class ConSanWorkgroupIdentitySource : uint8_t {
  DescriptorSystemSgprs,
  CommandProcessorTtmps,
};

/// Selects the target generation's wait-counter encoding and counter model.
///
/// ConSan uses this type when it must order injected memory operations with
/// guest operations. The values name encoding families rather than individual
/// wait instructions: exact counter choice and immediate construction remain
/// the responsibility of the relevant lowerer. Keeping the distinction typed
/// prevents product-family tests from silently standing in for wait semantics.
enum class ConSanWaitCounterFamily : uint8_t {
  Gfx9,
  Gfx11,
  Gfx12,
};

/// Names the direct scalar call form available to injected ConSan code.
///
/// `SCallB64` and `SCallI64` have different encodings and return-address
/// behavior, so routing and relay construction must ask this typed target fact
/// rather than infer it from a broad architecture family. The enum describes
/// architectural availability; each engine still decides whether a direct
/// call is suitable at a particular patch site.
enum class ConSanDirectCallForm : uint8_t {
  SCallB64,
  SCallI64,
};

/// Describes the strength of the stable contract for one
/// target/engine/capability-form combination.
///
/// `Supported` means causal evidence for an MOI engine or redundant access
/// observation for SuperCollider. `MutationOnly` means the form can participate
/// in fault injection but is not observed as evidence. `AccessOnly` means the
/// access aspect is covered without claiming synchronization semantics.
/// `AssociatedOnly` means the form can be associated with another observed
/// event but is not independently covered. `NotApplicable` is a deliberate
/// target/engine exclusion, whereas `OutOfContract` means the query itself used
/// an unknown target, sentinel value, or otherwise invalid combination. These
/// distinctions keep weaker behavior visible instead of presenting every
/// non-`Supported` result as an accidental omission.
enum class ConSanCapabilityDisposition : uint8_t {
  OutOfContract,
  NotApplicable,
  Supported,
  MutationOnly,
  AccessOnly,
  AssociatedOnly,
};

/// The immutable, target-wide architectural contract used by ConSan.
///
/// There is exactly one profile for each admitted target. It contains facts
/// that remain constant across every code object and kernel for that target:
/// product and encoding families, available identity/call/wait facilities,
/// register-allocation rules, address and memory limits, and normalized
/// semantic-form availability. It deliberately contains no engine choice,
/// decoded-instruction state, resource-allocation decision, or mutable analysis
/// result.
///
/// A profile is selected once from the code-object target and thereafter
/// passed or queried as read-only data. Descriptor-selected facts such as wave
/// size are validated and projected into `ConSanKernelTargetProfile`; they are
/// never written back here. This separation makes architecture support a small
/// auditable table and prevents local lowerers from growing their own competing
/// definitions of the same hardware facts.
struct ConSanTargetProfile {
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  ConSanArchitectureFamily architecture_family = ConSanArchitectureFamily::Cdna;
  ConSanEncodingFamily encoding_family = ConSanEncodingFamily::Gfx9Cdna3;
  ConSanAccumulatorModel accumulator_model = ConSanAccumulatorModel::None;
  ConSanDispatchIdentitySource dispatch_identity = ConSanDispatchIdentitySource::PreloadedSgprPair;
  ConSanWorkgroupIdentitySource workgroup_identity =
      ConSanWorkgroupIdentitySource::DescriptorSystemSgprs;
  ConSanWaitCounterFamily wait_counter_family = ConSanWaitCounterFamily::Gfx9;
  ConSanDirectCallForm direct_call_form = ConSanDirectCallForm::SCallB64;

  bool supports_wave32 = false;
  bool supports_wave64 = true;
  uint8_t exec_register_width_bits = 64;
  uint8_t global_address_width_bits = 64;
  uint8_t lds_address_width_bits = 32;
  uint8_t vgpr_allocation_granularity_wave32 = 0;
  uint8_t vgpr_allocation_granularity_wave64 = 0;
  uint8_t sgpr_allocation_granularity = 8;
  uint8_t accumulator_offset_granularity = 0;
  uint16_t ordinary_sgpr_limit = 0;
  uint16_t reserved_ordinary_sgpr_base = 0;
  uint16_t reserved_ordinary_sgpr_count = 0;
  uint16_t user_sgpr_initialization_limit = 0;

  uint32_t address_free_private_limit_bytes = 0;
  uint32_t private_allocation_granularity_bytes = 1;
  uint32_t max_group_segment_bytes = 0;
  int32_t direct_branch_min_displacement_bytes = 0;
  int32_t direct_branch_max_displacement_bytes = 0;

  bool supports_kernarg_preload_overflow_recovery = false;
  bool has_cluster_facilities = false;
  bool has_selectable_vgpr_bank = false;
  bool requires_even_vgpr_tuples = false;
  uint16_t semantic_form_mask = 0;
};

/// The validated, immutable target view for one kernel descriptor.
///
/// This record combines a pointer to the code object's `ConSanTargetProfile`
/// with descriptor-selected wave size and the allocation consequences of that
/// choice. Construction fails when the descriptor requests a wave size the
/// target does not support, so downstream resource planning can consume this
/// type without repeating wave-size admission logic. The `target` pointer
/// refers to the static profile table and therefore outlives every derived
/// kernel view.
///
/// The record remains intentionally narrow: decoded sites, liveness, selected
/// scratch registers, and engine policy belong to later per-kernel stages and
/// must not be added as mutable fields here.
struct ConSanKernelTargetProfile {
  const ConSanTargetProfile *target = nullptr;
  uint8_t wave_size = 0;
  uint8_t active_exec_mask_width_bits = 0;
  uint8_t vgpr_allocation_granularity = 0;
};

[[nodiscard]] constexpr uint16_t consan_capability_form_bit(ConSanCapabilityForm form) {
  return form == ConSanCapabilityForm::Count
             ? 0u
             : static_cast<uint16_t>(1u << static_cast<uint8_t>(form));
}

inline constexpr uint16_t kConSanCommonSemanticFormMask =
    consan_capability_form_bit(ConSanCapabilityForm::NativeLdsAccess) |
    consan_capability_form_bit(ConSanCapabilityForm::GroupFlatAccess) |
    consan_capability_form_bit(ConSanCapabilityForm::WorkgroupBarrier) |
    consan_capability_form_bit(ConSanCapabilityForm::OrderedFlatAtomic) |
    consan_capability_form_bit(ConSanCapabilityForm::OrderedVglobalAtomic) |
    consan_capability_form_bit(ConSanCapabilityForm::AddressedOrdinaryFence);

inline constexpr uint16_t kConSanCdnaSemanticFormMask =
    kConSanCommonSemanticFormMask |
    consan_capability_form_bit(ConSanCapabilityForm::RelaxedLdsAtomicAccess);

inline constexpr uint16_t kConSanGfx1250SemanticFormMask =
    kConSanCdnaSemanticFormMask | consan_capability_form_bit(ConSanCapabilityForm::ClusterBarrier) |
    consan_capability_form_bit(ConSanCapabilityForm::OrderedLdsAtomic);

/// The production target-admission map, architectural facts, semantic-form
/// support, and documentation iteration order share this table. Adding a
/// target therefore cannot update one of those contracts without the others.
inline constexpr std::array<ConSanTargetProfile, 5> kConSanTargetProfiles = {{
    {
        .target = ROCJITSU_CODE_TARGET_GFX942,
        .arch = ROCJITSU_CODE_ARCH_CDNA3,
        .architecture_family = ConSanArchitectureFamily::Cdna,
        .encoding_family = ConSanEncodingFamily::Gfx9Cdna3,
        .accumulator_model = ConSanAccumulatorModel::DescriptorPartitioned,
        .dispatch_identity = ConSanDispatchIdentitySource::PreloadedSgprPair,
        .workgroup_identity = ConSanWorkgroupIdentitySource::DescriptorSystemSgprs,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx9,
        .direct_call_form = ConSanDirectCallForm::SCallB64,
        .supports_wave32 = false,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 0,
        .vgpr_allocation_granularity_wave64 = 8,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 4,
        .ordinary_sgpr_limit = 102,
        .reserved_ordinary_sgpr_base = 0,
        .reserved_ordinary_sgpr_count = 0,
        .user_sgpr_initialization_limit = 16,
        .address_free_private_limit_bytes = 0x1000u,
        .private_allocation_granularity_bytes = 16,
        .max_group_segment_bytes = 64u * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = true,
        .has_cluster_facilities = false,
        .has_selectable_vgpr_bank = false,
        .requires_even_vgpr_tuples = true,
        .semantic_form_mask = kConSanCdnaSemanticFormMask,
    },
    {
        .target = ROCJITSU_CODE_TARGET_GFX950,
        .arch = ROCJITSU_CODE_ARCH_CDNA4,
        .architecture_family = ConSanArchitectureFamily::Cdna,
        .encoding_family = ConSanEncodingFamily::Gfx9Cdna4,
        .accumulator_model = ConSanAccumulatorModel::DescriptorPartitioned,
        .dispatch_identity = ConSanDispatchIdentitySource::PreloadedSgprPair,
        .workgroup_identity = ConSanWorkgroupIdentitySource::DescriptorSystemSgprs,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx9,
        .direct_call_form = ConSanDirectCallForm::SCallB64,
        .supports_wave32 = false,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 0,
        .vgpr_allocation_granularity_wave64 = 8,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 4,
        .ordinary_sgpr_limit = 102,
        .reserved_ordinary_sgpr_base = 0,
        .reserved_ordinary_sgpr_count = 0,
        .user_sgpr_initialization_limit = 16,
        .address_free_private_limit_bytes = 0x1000u,
        .private_allocation_granularity_bytes = 16,
        .max_group_segment_bytes = 64u * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = true,
        .has_cluster_facilities = false,
        .has_selectable_vgpr_bank = false,
        .requires_even_vgpr_tuples = true,
        .semantic_form_mask = kConSanCdnaSemanticFormMask,
    },
    {
        .target = ROCJITSU_CODE_TARGET_GFX1100,
        .arch = ROCJITSU_CODE_ARCH_RDNA3,
        .architecture_family = ConSanArchitectureFamily::Rdna,
        .encoding_family = ConSanEncodingFamily::Gfx11,
        .accumulator_model = ConSanAccumulatorModel::None,
        .dispatch_identity = ConSanDispatchIdentitySource::CodeObjectLiteral,
        .workgroup_identity = ConSanWorkgroupIdentitySource::DescriptorSystemSgprs,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx11,
        .direct_call_form = ConSanDirectCallForm::SCallB64,
        .supports_wave32 = true,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 8,
        .vgpr_allocation_granularity_wave64 = 4,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 0,
        .ordinary_sgpr_limit = 106,
        .reserved_ordinary_sgpr_base = 0,
        .reserved_ordinary_sgpr_count = 0,
        .user_sgpr_initialization_limit = 16,
        .address_free_private_limit_bytes = 0x1000u,
        .private_allocation_granularity_bytes = 16,
        .max_group_segment_bytes = 64u * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = true,
        .has_cluster_facilities = false,
        .has_selectable_vgpr_bank = false,
        .requires_even_vgpr_tuples = false,
        .semantic_form_mask = kConSanCommonSemanticFormMask,
    },
    {
        .target = ROCJITSU_CODE_TARGET_GFX1201,
        .arch = ROCJITSU_CODE_ARCH_RDNA4,
        .architecture_family = ConSanArchitectureFamily::Rdna,
        .encoding_family = ConSanEncodingFamily::Gfx12,
        .accumulator_model = ConSanAccumulatorModel::None,
        .dispatch_identity = ConSanDispatchIdentitySource::CodeObjectLiteral,
        .workgroup_identity = ConSanWorkgroupIdentitySource::CommandProcessorTtmps,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx12,
        .direct_call_form = ConSanDirectCallForm::SCallB64,
        .supports_wave32 = true,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 8,
        .vgpr_allocation_granularity_wave64 = 4,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 0,
        .ordinary_sgpr_limit = 106,
        .reserved_ordinary_sgpr_base = 0,
        .reserved_ordinary_sgpr_count = 0,
        .user_sgpr_initialization_limit = 16,
        .address_free_private_limit_bytes = 0x800000u,
        .private_allocation_granularity_bytes = 1,
        .max_group_segment_bytes = 64u * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = false,
        .has_cluster_facilities = false,
        .has_selectable_vgpr_bank = false,
        .requires_even_vgpr_tuples = false,
        .semantic_form_mask = kConSanCommonSemanticFormMask,
    },
    {
        .target = ROCJITSU_CODE_TARGET_GFX1250,
        .arch = ROCJITSU_CODE_ARCH_CDNA5,
        .architecture_family = ConSanArchitectureFamily::Cdna,
        .encoding_family = ConSanEncodingFamily::Gfx12,
        .accumulator_model = ConSanAccumulatorModel::SelectableVgprBank,
        .dispatch_identity = ConSanDispatchIdentitySource::CodeObjectLiteral,
        .workgroup_identity = ConSanWorkgroupIdentitySource::CommandProcessorTtmps,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx12,
        .direct_call_form = ConSanDirectCallForm::SCallI64,
        .supports_wave32 = true,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 16,
        .vgpr_allocation_granularity_wave64 = 8,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 0,
        .ordinary_sgpr_limit = 106,
        .reserved_ordinary_sgpr_base = 102,
        .reserved_ordinary_sgpr_count = 4,
        .user_sgpr_initialization_limit = 32,
        .address_free_private_limit_bytes = 0x800000u,
        .private_allocation_granularity_bytes = 1,
        .max_group_segment_bytes = static_cast<uint32_t>(ROCJITSU_GFX1250_LDS_SIZE_KB) * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = false,
        .has_cluster_facilities = true,
        .has_selectable_vgpr_bank = true,
        .requires_even_vgpr_tuples = true,
        .semantic_form_mask = kConSanGfx1250SemanticFormMask,
    },
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

[[nodiscard]] constexpr const ConSanTargetProfile *
consan_target_profile(rj_code_target_id_t target) {
  for (const ConSanTargetProfile &profile : kConSanTargetProfiles) {
    if (target == profile.target)
      return &profile;
  }
  return nullptr;
}

[[nodiscard]] constexpr const ConSanTargetProfile *consan_target_profile(rj_code_arch_t arch) {
  for (const ConSanTargetProfile &profile : kConSanTargetProfiles) {
    if (arch == profile.arch)
      return &profile;
  }
  return nullptr;
}

[[nodiscard]] constexpr bool consan_target_profiles_are_valid() {
  constexpr uint16_t all_form_bits =
      static_cast<uint16_t>((1u << static_cast<uint8_t>(ConSanCapabilityForm::Count)) - 1u);
  for (std::size_t lhs = 0; lhs < kConSanTargetProfiles.size(); ++lhs) {
    const ConSanTargetProfile &profile = kConSanTargetProfiles[lhs];
    if (profile.target == ROCJITSU_CODE_TARGET_INVALID ||
        profile.arch == ROCJITSU_CODE_ARCH_INVALID || !profile.supports_wave64 ||
        profile.exec_register_width_bits != 64u || profile.global_address_width_bits != 64u ||
        profile.lds_address_width_bits != 32u || profile.vgpr_allocation_granularity_wave64 == 0u ||
        (profile.supports_wave32 != (profile.vgpr_allocation_granularity_wave32 != 0u)) ||
        profile.sgpr_allocation_granularity == 0u || profile.ordinary_sgpr_limit == 0u ||
        profile.user_sgpr_initialization_limit == 0u ||
        profile.address_free_private_limit_bytes == 0u ||
        profile.private_allocation_granularity_bytes == 0u ||
        profile.max_group_segment_bytes == 0u ||
        profile.direct_branch_min_displacement_bytes >= 0 ||
        profile.direct_branch_max_displacement_bytes <= 0 ||
        (profile.semantic_form_mask & static_cast<uint16_t>(~all_form_bits)) != 0u ||
        profile.semantic_form_mask == 0u ||
        (profile.has_selectable_vgpr_bank !=
         (profile.accumulator_model == ConSanAccumulatorModel::SelectableVgprBank)) ||
        (profile.has_cluster_facilities &&
         (profile.semantic_form_mask &
          consan_capability_form_bit(ConSanCapabilityForm::ClusterBarrier)) == 0u)) {
      return false;
    }
    if ((profile.reserved_ordinary_sgpr_count == 0u) != (profile.reserved_ordinary_sgpr_base == 0u))
      return false;
    if (static_cast<uint32_t>(profile.reserved_ordinary_sgpr_base) +
            profile.reserved_ordinary_sgpr_count >
        profile.ordinary_sgpr_limit)
      return false;
    for (std::size_t rhs = lhs + 1; rhs < kConSanTargetProfiles.size(); ++rhs) {
      if (profile.target == kConSanTargetProfiles[rhs].target ||
          profile.arch == kConSanTargetProfiles[rhs].arch)
        return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr rj_code_arch_t consan_arch_for_target(rj_code_target_id_t target) {
  const ConSanTargetProfile *profile = consan_target_profile(target);
  return profile ? profile->arch : ROCJITSU_CODE_ARCH_INVALID;
}

[[nodiscard]] constexpr bool consan_is_capability_target(rj_code_target_id_t target) {
  return consan_target_profile(target) != nullptr;
}

[[nodiscard]] constexpr bool consan_is_capability_arch(rj_code_arch_t arch) {
  return consan_target_profile(arch) != nullptr;
}

[[nodiscard]] constexpr bool consan_profile_supports_wave_size(const ConSanTargetProfile &profile,
                                                               uint32_t wave_size) {
  return (wave_size == 32u && profile.supports_wave32) ||
         (wave_size == 64u && profile.supports_wave64);
}

[[nodiscard]] constexpr uint32_t
consan_profile_vgpr_allocation_granularity(const ConSanTargetProfile &profile, uint32_t wave_size) {
  if (!consan_profile_supports_wave_size(profile, wave_size))
    return 0u;
  return wave_size == 32u ? profile.vgpr_allocation_granularity_wave32
                          : profile.vgpr_allocation_granularity_wave64;
}

[[nodiscard]] constexpr std::optional<ConSanKernelTargetProfile>
consan_kernel_target_profile(const ConSanTargetProfile &profile, uint32_t wave_size) {
  const uint32_t vgpr_granularity = consan_profile_vgpr_allocation_granularity(profile, wave_size);
  if (vgpr_granularity == 0u)
    return std::nullopt;
  return ConSanKernelTargetProfile{
      .target = &profile,
      .wave_size = static_cast<uint8_t>(wave_size),
      .active_exec_mask_width_bits = static_cast<uint8_t>(wave_size),
      .vgpr_allocation_granularity = static_cast<uint8_t>(vgpr_granularity),
  };
}

[[nodiscard]] constexpr bool
consan_profile_reserved_sgpr_range_overlaps(const ConSanTargetProfile &profile, uint16_t base,
                                            uint16_t count) {
  return count != 0u && profile.reserved_ordinary_sgpr_count != 0u &&
         static_cast<uint32_t>(base) + count > profile.reserved_ordinary_sgpr_base &&
         static_cast<uint32_t>(profile.reserved_ordinary_sgpr_base) +
                 profile.reserved_ordinary_sgpr_count >
             base;
}

[[nodiscard]] constexpr std::optional<uint32_t>
consan_profile_normalize_private_size(const ConSanTargetProfile &profile,
                                      uint32_t requested_bytes) {
  const uint64_t granularity = profile.private_allocation_granularity_bytes;
  const uint64_t normalized =
      (static_cast<uint64_t>(requested_bytes) + granularity - 1u) / granularity * granularity;
  if (normalized > profile.address_free_private_limit_bytes)
    return std::nullopt;
  return static_cast<uint32_t>(normalized);
}

[[nodiscard]] constexpr std::optional<uint32_t>
consan_address_free_private_limit(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile ? std::optional<uint32_t>(profile->address_free_private_limit_bytes)
                 : std::nullopt;
}

[[nodiscard]] constexpr std::optional<uint32_t>
consan_normalize_address_free_private_size(rj_code_arch_t arch, uint32_t requested_bytes) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile ? consan_profile_normalize_private_size(*profile, requested_bytes) : std::nullopt;
}

[[nodiscard]] constexpr bool consan_uses_gfx9_cdna_encoding(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && (profile->encoding_family == ConSanEncodingFamily::Gfx9Cdna3 ||
                     profile->encoding_family == ConSanEncodingFamily::Gfx9Cdna4);
}

[[nodiscard]] constexpr bool consan_uses_gfx12_encoding(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && profile->encoding_family == ConSanEncodingFamily::Gfx12;
}

[[nodiscard]] constexpr bool consan_uses_gfx11_encoding(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && profile->encoding_family == ConSanEncodingFamily::Gfx11;
}

[[nodiscard]] constexpr bool consan_uses_gfx11_or_gfx12_encoding(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && (profile->encoding_family == ConSanEncodingFamily::Gfx11 ||
                     profile->encoding_family == ConSanEncodingFamily::Gfx12);
}

[[nodiscard]] constexpr bool consan_arch_has_s_call_i64(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && profile->direct_call_form == ConSanDirectCallForm::SCallI64;
}

[[nodiscard]] constexpr bool consan_arch_has_s_call_b64(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && profile->direct_call_form == ConSanDirectCallForm::SCallB64;
}

[[nodiscard]] constexpr bool consan_arch_has_cluster_facilities(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && profile->has_cluster_facilities;
}

[[nodiscard]] constexpr bool consan_arch_has_selectable_vgpr_bank(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && profile->has_selectable_vgpr_bank;
}

[[nodiscard]] constexpr bool
consan_arch_has_descriptor_partitioned_accumulators(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && profile->accumulator_model == ConSanAccumulatorModel::DescriptorPartitioned;
}

/// Some ConSan probes use the code-object dispatch identity literal instead of
/// reserving a guest SGPR pair. The availability is an immutable target fact;
/// the decision to consume it remains engine policy.
[[nodiscard]] constexpr bool consan_arch_uses_literal_dispatch_identity(rj_code_arch_t arch) {
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && profile->dispatch_identity == ConSanDispatchIdentitySource::CodeObjectLiteral;
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
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  return profile && form != ConSanCapabilityForm::Count &&
         (profile->semantic_form_mask & consan_capability_form_bit(form)) != 0u;
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

static_assert(consan_target_profiles_are_valid());
static_assert(consan_capability_enum_is_complete(kConSanCapabilityEngines));
static_assert(consan_capability_enum_is_complete(kConSanCapabilityDomains));
static_assert(consan_capability_enum_is_complete(kConSanCapabilityForms));

} // namespace rocjitsu
