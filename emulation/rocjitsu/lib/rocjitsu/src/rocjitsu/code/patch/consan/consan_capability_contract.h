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

#include "rocjitsu/code/dbt/processor_revision.h"
#include "rocjitsu/code/patch/consan/consan_enum_vocabulary.h"
#include "rocjitsu/code/patch/consan/consan_mode.h"
#include "rocjitsu/code/rj_code.h"

namespace rocjitsu::consan {

struct ProgramAnalysisTargetOperations;

/// Architectural operand encodings used uniformly by every supported target
/// profile.  Concrete target packages own exceptions; common construction and
/// validation use these target-neutral names instead of importing one ISA
/// generation's vocabulary.
inline constexpr uint16_t kAmdGpuExecLo = 126u;
inline constexpr uint16_t kAmdGpuVccLo = 106u;
inline constexpr uint16_t kAmdGpuWorkitemIdX = 0u;
inline constexpr uint8_t kAmdGpuScopeDevice = 2u;

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
enum class CapabilityDomain : uint8_t {
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
enum class CapabilityForm : uint8_t {
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

/// Describes how accumulator registers participate in a kernel's physical
/// vector-register allocation.
///
/// `None` means ConSan does not need a separate accumulator allocation model.
/// `DescriptorPartitioned` means a descriptor field divides one allocation
/// into ordinary VGPR and AccVGPR regions. `SelectableVgprBank` means the ISA
/// selects among physical VGPR banks rather than exposing the CDNA3/CDNA4 descriptor
/// partition. Resource planning uses this type to interpret allocation facts;
/// it does not by itself select an instrumentation strategy.
enum class AccumulatorModel : uint8_t {
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
/// records availability, not policy: a mode may choose not to consume the
/// facility even when the target provides it.
enum class DispatchIdentitySource : uint8_t {
  PreloadedSgprPair,
  CodeObjectLiteral,
};

/// Exact command-processor ABI registers carrying workgroup coordinates.
///
/// Absence from a target profile means that workgroup coordinates use the
/// descriptor-enabled system-SGPR path. Presence makes every concrete TTMP
/// index a target-owned fact while leaving common placement responsible only
/// for interpreting the uniform packed-coordinate representation.
struct CommandProcessorWorkgroupIdentity {
  uint8_t grid_x_ttmp = 0;
  uint8_t grid_yz_ttmp = 0;
  std::optional<uint8_t> cluster_workgroup_id_ttmp;
  /// Number of low bits carrying the variable cluster-local coordinate.
  /// Remaining bits in the same TTMP may describe the fixed cluster shape.
  uint8_t cluster_workgroup_id_low_bit_count = 0;

  bool operator==(const CommandProcessorWorkgroupIdentity &) const = default;
};

/// Names the direct scalar call form available to injected ConSan code.
///
/// `SCallB64` and `SCallI64` have different encodings and return-address
/// behavior, so routing and relay construction must ask this typed target fact
/// rather than infer it from a broad architecture family. The enum describes
/// architectural availability; each mode still decides whether a direct
/// call is suitable at a particular patch site.
enum class DirectCallForm : uint8_t {
  SCallB64,
  SCallI64,
};

/// Selects the target-owned device-cache refresh sequence used before ConSan
/// consumes published evidence. `None` means no explicit refresh is required.
enum class DeviceCacheRefreshForm : uint8_t {
  None,
  Cdna3BufferInvSc1,
  Cdna4BufferInvSc1,
};

/// Describes what may happen to patched code before the target executes it.
///
/// `DirectCodeObject` means the byte layout emitted by RocJitsu is the layout
/// seen by every kernel. `PerKernelOwnerTranslation` means a later translation
/// step may clone or move generated regions independently for each owning
/// kernel while also translating that kernel's descriptor entry. In the
/// latter model, a branch between generated regions is safe only when its
/// route is preserved within one kernel owner's translated image; descriptor
/// redirection remains safe because the translated descriptor and destination
/// move together.
///
/// This is a transport fact, not an ISA-family label. It belongs in the target
/// profile so prologue placement does not infer post-instrumentation behavior
/// from a product name such as gfx1250.
enum class CodeTransportModel : uint8_t {
  DirectCodeObject,
  PerKernelOwnerTranslation,
};

/// Describes the target HWREG field that uniquely identifies one resident
/// wave for the lifetime of that residency.
///
/// ConSan uses this field as an owner identity when a mode does not need a
/// dispatch-global identity. `hwreg_id` is the architectural register number
/// encoded by `s_getreg_b32`; `bit_offset` and `bit_width` select the identity
/// field within that register. This is a target fact, not an emitted
/// instruction or a mode policy: the target operation that consumes it is
/// responsible for validating registers and constructing the instruction
/// sequence, while the mode decides whether a resident-wave owner is the
/// right semantic identity for a particular observation.
struct ResidentWaveIdentityEncoding {
  uint16_t hwreg_id = 0;
  uint8_t bit_offset = 0;
  uint8_t bit_width = 0;

  bool operator==(const ResidentWaveIdentityEncoding &) const = default;
};

/// Normalized target support for address forms that ConSan may need to
/// materialize before publishing atomic evidence.
///
/// The common materializer owns the shared instruction-building mechanism.
/// Concrete targets own which already-normalized forms that mechanism may
/// consume, so mode code and the common emitter do not infer support from a
/// product name or encoding family.
struct AtomicAddressMaterializationCapability {
  bool flat_and_global = false;
  bool buffer_resource = false;
  bool lds_byte_offset_token = false;
  bool scaled_vglobal = false;

  bool operator==(const AtomicAddressMaterializationCapability &) const = default;
};

/// How a scalar-plus-vector memory address extends its 32-bit vector offset
/// before adding it to the scalar base.
enum class VectorOffsetExtension : uint8_t {
  Zero,
  Sign,
};

/// Availability of the explicit vector-offset scaling bit in a target's
/// normalized memory encoding. `Absent` targets do not encode the field;
/// `Disabled` targets encode it but ConSan admits only zero; `Supported`
/// targets admit either value.
enum class ScaleOffsetCapability : uint8_t {
  Absent,
  Disabled,
  Supported,
};

/// Shared heuristic model for the vector-memory address encodings consumed by
/// ConSan. This deliberately captures the small set of semantic dimensions
/// needed by common classifiers instead of making them recover product sets.
/// Exact bit extraction remains in target decoder owners.
struct VectorMemoryCapability {
  uint8_t instruction_word_count = 2;
  uint8_t immediate_offset_bits = 13;
  uint32_t flat_vector_only_saddr = 0;
  uint32_t global_vector_only_saddr = 0x7fu;
  bool supports_flat_scalar_base = false;
  VectorOffsetExtension vector_offset_extension = VectorOffsetExtension::Zero;
  ScaleOffsetCapability scale_offset = ScaleOffsetCapability::Absent;

  bool operator==(const VectorMemoryCapability &) const = default;
};

/// Mnemonic vocabulary used by one target's native LDS instructions. The
/// distinction is semantic only where the common classifier admits a bounded
/// spelling set; exact opcode decoding remains target-owned.
enum class NativeLdsMnemonicDialect : uint8_t {
  LoadStore,
  ReadWrite,
};

/// Target-native LDS encoding facts needed after instruction decoding has
/// classified an access but has deliberately not classified it as a
/// synchronization operation. Most single-range DS/VDS offsets occupy sixteen
/// bits; CDNA3/4 access-only atomic forms expose only their low eight-bit
/// `offset0` field. Keeping that exception here prevents the common inventory
/// from reconstructing an encoding family from a product set.
struct NativeLdsCapability {
  NativeLdsMnemonicDialect mnemonic_dialect = NativeLdsMnemonicDialect::LoadStore;
  uint8_t single_range_atomic_offset_bits = 16;

  bool operator==(const NativeLdsCapability &) const = default;
};

/// Target-wide access-lowering mechanisms for ConSan instrumentation.
///
/// These are deliberately implementation capabilities rather than product
/// families. They state which recovery mechanisms the common access planner
/// may rely on after program analysis has normalized the guest instruction.
struct AccessCapability {
  bool dynamic_stack_uses_scalar_reservoir = false;
  bool native_lds_spill_recovery = false;
  bool clobbered_address_spill_reload = false;

  bool operator==(const AccessCapability &) const = default;
};

/// Target synchronization sequence recognized after normalized instruction
/// decoding. Exact wait decoding remains target-owned; this facet only admits
/// the common same-block group-FLAT acquire/wait association.
struct SynchronizationCapability {
  bool workgroup_flat_acquire_wait_fallback = false;

  bool operator==(const SynchronizationCapability &) const = default;
};

/// Target-wide heuristic for placing temporary scalar instrumentation state.
///
/// `DescriptorPartitioned` protects the compiler's physical-VCC and AccVGPR
/// boundaries and permits owner-component recovery around those boundaries.
/// `LivenessOnly` has neither a partitioned scalar ABI nor an entry-backed
/// spill mechanism. `SpillBacked` may preserve borrowed scalar state at entry
/// and use private storage across full-pressure probes. The model deliberately
/// captures solver-relevant behavior without reproducing architecture families
/// inside placement code.
enum class ScalarPlacementModel : uint8_t {
  Unsupported,
  DescriptorPartitioned,
  LivenessOnly,
  SpillBacked,
};

enum class WorkgroupPayloadConsumption : uint8_t {
  ObservationBody,
  EntryCapture,
};

struct PlacementCapability {
  uint8_t scratch_vgpr_alignment = 1;
  bool branch_only_spill_embeds_setup_state = false;
  bool automatic_dispatch_sgpr_requires_owner_admission = false;
  // Leave scalar capacity for probes when literal launch identity is available.
  bool prefer_literal_dispatch_identity = false;
  WorkgroupPayloadConsumption full_workgroup_payload_consumption =
      WorkgroupPayloadConsumption::ObservationBody;

  bool operator==(const PlacementCapability &) const = default;
};

/// Describes the strength of the stable contract for one
/// target/mode/capability-form combination.
///
/// `Supported` means causal evidence for ConSan or redundant access
/// observation for SuperCollider. `MutationOnly` means the form can participate
/// in fault injection but is not observed as evidence. `AccessOnly` means the
/// access aspect is covered without claiming synchronization semantics.
/// `AssociatedOnly` means the form can be associated with another observed
/// event but is not independently covered. `NotApplicable` is a deliberate
/// target/mode exclusion, whereas `OutOfContract` means the query itself used
/// an unknown target, sentinel value, or otherwise invalid combination. These
/// distinctions keep weaker behavior visible instead of presenting every
/// non-`Supported` result as an accidental omission.
enum class CapabilityDisposition : uint8_t {
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
/// architecture lineage, available identity/call/wait facilities,
/// register-allocation rules, address and memory limits, and normalized
/// semantic-form availability. It deliberately contains no mode choice,
/// decoded-instruction state, resource-allocation decision, or mutable analysis
/// result.
///
/// A profile is selected once from the code-object target and thereafter
/// passed or queried as read-only data. Descriptor-selected facts such as wave
/// size remain in the program inventory and resource plans; they are never
/// written back here. This separation makes architecture support a small
/// auditable table and prevents local lowerers from growing their own competing
/// definitions of the same hardware facts.
struct TargetProfile {
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  AccumulatorModel accumulator_model = AccumulatorModel::None;
  ScalarPlacementModel scalar_placement_model = ScalarPlacementModel::Unsupported;
  /// Target-owned decoder facet. Keeping this registration beside the target
  /// facts prevents every consumer domain from rebuilding the five-target map.
  const ProgramAnalysisTargetOperations *program_analysis = nullptr;
  DispatchIdentitySource dispatch_identity = DispatchIdentitySource::PreloadedSgprPair;
  std::optional<CommandProcessorWorkgroupIdentity> command_processor_workgroup_identity;
  DirectCallForm direct_call_form = DirectCallForm::SCallB64;
  DeviceCacheRefreshForm device_cache_refresh = DeviceCacheRefreshForm::None;
  CodeTransportModel code_transport = CodeTransportModel::DirectCodeObject;
  /// Same-revision selection required for a structural, non-semantic DBT pass.
  ProcessorRevision identity_translation_revision = ProcessorRevision::Unspecified;
  ResidentWaveIdentityEncoding resident_wave_identity;
  AtomicAddressMaterializationCapability atomic_address_materialization;
  VectorMemoryCapability vector_memory;
  NativeLdsCapability native_lds;
  AccessCapability access;
  SynchronizationCapability synchronization;
  PlacementCapability placement;
  bool access_reports_need_explicit_dispatch_identity = true;
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
  bool supports_kernarg_preload_overflow_recovery = false;
  bool has_cluster_facilities = false;
  bool has_selectable_vgpr_bank = false;
  bool requires_even_vgpr_tuples = false;
  /// Alignment of the encoded new/expected data pair for FLAT compare-swap.
  /// This is distinct from general address/data tuple alignment: translated
  /// targets may require even tuples elsewhere without using this legacy CAS
  /// operand layout.
  uint8_t flat_compare_swap_data_pair_alignment = 1;
  /// True when one encoded two-address LDS instruction must be split into two
  /// single-address instructions after relocation changes its address base.
  /// Targets whose encoding can replay the original instruction leave this
  /// false. This is an exact relocation constraint, not a general statement
  /// about whether the target supports two-address LDS operations.
  bool requires_split_two_address_lds_relocation = false;
  uint16_t semantic_form_mask = 0;
  bool requires_supercollider_runtime_flat_group_gate = false;
  /// Whether decoded atomic/fence ordering includes an explicit TH/SC field.
  bool requires_raw_memory_order_qualifier = false;
};

[[nodiscard]] constexpr uint16_t capability_form_bit(CapabilityForm form) {
  const uint8_t index = static_cast<uint8_t>(form);
  return index < static_cast<uint8_t>(CapabilityForm::Count) ? static_cast<uint16_t>(1u << index)
                                                             : 0u;
}

inline constexpr uint16_t kCommonSemanticFormMask =
    capability_form_bit(CapabilityForm::NativeLdsAccess) |
    capability_form_bit(CapabilityForm::GroupFlatAccess) |
    capability_form_bit(CapabilityForm::WorkgroupBarrier) |
    capability_form_bit(CapabilityForm::OrderedFlatAtomic) |
    capability_form_bit(CapabilityForm::OrderedVglobalAtomic) |
    capability_form_bit(CapabilityForm::AddressedOrdinaryFence);

inline constexpr uint16_t kCdnaSemanticFormMask =
    kCommonSemanticFormMask | capability_form_bit(CapabilityForm::RelaxedLdsAtomicAccess);

inline constexpr auto kEnabledModes = [] {
  using E = Mode;
  return make_enum_vocabulary("unknown", enum_entry(E::SuperCollider, "SuperCollider"),
                              enum_entry(E::Default, "ConSan"));
}();

template <typename Values>
[[nodiscard]] constexpr bool enabled_modes_are_complete(const Values &values) {
  return values.size() == 2u && enabled_mode(values[0]).has_value() &&
         enabled_mode(values[1]).has_value() && values[0] != values[1];
}

inline constexpr std::array<CapabilityDomain, 4> kCapabilityDomains = {
    CapabilityDomain::Access,
    CapabilityDomain::Barrier,
    CapabilityDomain::Atomic,
    CapabilityDomain::Fence,
};

inline constexpr auto kCapabilityForms = [] {
  using E = CapabilityForm;
  return make_enum_vocabulary(
      "unknown", enum_entry(E::NativeLdsAccess, "native LDS"),
      enum_entry(E::GroupFlatAccess, "group FLAT"), enum_entry(E::WorkgroupBarrier, "workgroup"),
      enum_entry(E::ClusterBarrier, "cluster"), enum_entry(E::OrderedFlatAtomic, "ordered FLAT"),
      enum_entry(E::OrderedVglobalAtomic, "ordered VGLOBAL"),
      enum_entry(E::OrderedLdsAtomic, "ordered LDS"),
      enum_entry(E::RelaxedLdsAtomicAccess, "relaxed LDS RMW"),
      enum_entry(E::AddressedOrdinaryFence, "addressed ordinary"));
}();

template <typename Values>
[[nodiscard]] constexpr bool capability_enum_is_complete(const Values &values) {
  using Enum = std::remove_cvref_t<decltype(values[0])>;
  constexpr std::size_t count = static_cast<std::size_t>(Enum::Count);
  if (values.size() != count)
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

[[nodiscard]] const TargetProfile *target_profile(rj_code_target_id_t target);
[[nodiscard]] const TargetProfile *target_profile(rj_code_arch_t arch);

template <std::size_t N>
[[nodiscard]] constexpr bool
target_profiles_are_valid(const std::array<TargetProfile, N> &profiles) {
  constexpr uint16_t all_form_bits =
      static_cast<uint16_t>((1u << static_cast<uint8_t>(CapabilityForm::Count)) - 1u);
  if (profiles.empty())
    return false;
  for (std::size_t lhs = 0; lhs < profiles.size(); ++lhs) {
    const TargetProfile &profile = profiles[lhs];
    const auto &workgroup_identity = profile.command_processor_workgroup_identity;
    if (profile.target == ROCJITSU_CODE_TARGET_INVALID ||
        profile.arch == ROCJITSU_CODE_ARCH_INVALID || profile.program_analysis == nullptr ||
        (profile.flat_compare_swap_data_pair_alignment != 1u &&
         profile.flat_compare_swap_data_pair_alignment != 2u) ||
        profile.vgpr_allocation_granularity_wave64 == 0u ||
        profile.sgpr_allocation_granularity == 0u || profile.ordinary_sgpr_limit == 0u ||
        profile.user_sgpr_initialization_limit == 0u ||
        profile.address_free_private_limit_bytes == 0u ||
        profile.private_allocation_granularity_bytes == 0u ||
        profile.max_group_segment_bytes == 0u || profile.resident_wave_identity.hwreg_id > 63u ||
        profile.resident_wave_identity.bit_offset > 31u ||
        profile.resident_wave_identity.bit_width == 0u ||
        profile.resident_wave_identity.bit_width > 32u ||
        static_cast<uint16_t>(profile.resident_wave_identity.bit_offset) +
                profile.resident_wave_identity.bit_width >
            32u ||
        static_cast<uint8_t>(profile.dispatch_identity) >
            static_cast<uint8_t>(DispatchIdentitySource::CodeObjectLiteral) ||
        static_cast<uint8_t>(profile.code_transport) >
            static_cast<uint8_t>(CodeTransportModel::PerKernelOwnerTranslation) ||
        !profile.atomic_address_materialization.flat_and_global ||
        (profile.vector_memory.instruction_word_count != 2u &&
         profile.vector_memory.instruction_word_count != 3u) ||
        (profile.vector_memory.immediate_offset_bits != 13u &&
         profile.vector_memory.immediate_offset_bits != 24u) ||
        (profile.vector_memory.instruction_word_count == 2u) !=
            (profile.vector_memory.immediate_offset_bits == 13u) ||
        profile.vector_memory.flat_vector_only_saddr > 127u ||
        profile.vector_memory.global_vector_only_saddr > 127u ||
        static_cast<uint8_t>(profile.vector_memory.vector_offset_extension) >
            static_cast<uint8_t>(VectorOffsetExtension::Sign) ||
        static_cast<uint8_t>(profile.vector_memory.scale_offset) >
            static_cast<uint8_t>(ScaleOffsetCapability::Supported) ||
        (profile.vector_memory.instruction_word_count == 2u &&
         profile.vector_memory.scale_offset != ScaleOffsetCapability::Absent) ||
        static_cast<uint8_t>(profile.native_lds.mnemonic_dialect) >
            static_cast<uint8_t>(NativeLdsMnemonicDialect::ReadWrite) ||
        (profile.native_lds.single_range_atomic_offset_bits != 8u &&
         profile.native_lds.single_range_atomic_offset_bits != 16u) ||
        (profile.access.clobbered_address_spill_reload &&
         !profile.access.native_lds_spill_recovery) ||
        profile.scalar_placement_model == ScalarPlacementModel::Unsupported ||
        static_cast<uint8_t>(profile.scalar_placement_model) >
            static_cast<uint8_t>(ScalarPlacementModel::SpillBacked) ||
        (profile.placement.scratch_vgpr_alignment != 1u &&
         profile.placement.scratch_vgpr_alignment != 2u) ||
        (profile.requires_even_vgpr_tuples && profile.placement.scratch_vgpr_alignment != 2u) ||
        static_cast<uint8_t>(profile.placement.full_workgroup_payload_consumption) >
            static_cast<uint8_t>(WorkgroupPayloadConsumption::EntryCapture) ||
        (profile.semantic_form_mask & static_cast<uint16_t>(~all_form_bits)) != 0u ||
        profile.semantic_form_mask == 0u ||
        (profile.has_selectable_vgpr_bank !=
         (profile.accumulator_model == AccumulatorModel::SelectableVgprBank)) ||
        (profile.requires_supercollider_runtime_flat_group_gate &&
         !profile.has_selectable_vgpr_bank) ||
        ((profile.placement.prefer_literal_dispatch_identity ||
          !profile.access_reports_need_explicit_dispatch_identity) &&
         profile.dispatch_identity != DispatchIdentitySource::CodeObjectLiteral) ||
        (profile.has_cluster_facilities &&
         (profile.semantic_form_mask & capability_form_bit(CapabilityForm::ClusterBarrier)) ==
             0u) ||
        (workgroup_identity &&
         (workgroup_identity->grid_x_ttmp > 15u || workgroup_identity->grid_yz_ttmp > 15u ||
          workgroup_identity->grid_x_ttmp == workgroup_identity->grid_yz_ttmp ||
          (workgroup_identity->cluster_workgroup_id_ttmp &&
           (*workgroup_identity->cluster_workgroup_id_ttmp > 15u ||
            *workgroup_identity->cluster_workgroup_id_ttmp == workgroup_identity->grid_x_ttmp ||
            *workgroup_identity->cluster_workgroup_id_ttmp ==
                workgroup_identity->grid_yz_ttmp)))) ||
        (workgroup_identity && workgroup_identity->cluster_workgroup_id_ttmp.has_value()) !=
            profile.has_cluster_facilities ||
        (workgroup_identity && ((workgroup_identity->cluster_workgroup_id_ttmp.has_value() &&
                                 (workgroup_identity->cluster_workgroup_id_low_bit_count == 0u ||
                                  workgroup_identity->cluster_workgroup_id_low_bit_count > 32u)) ||
                                (!workgroup_identity->cluster_workgroup_id_ttmp.has_value() &&
                                 workgroup_identity->cluster_workgroup_id_low_bit_count != 0u)))) {
      return false;
    }
    if ((profile.reserved_ordinary_sgpr_count == 0u) != (profile.reserved_ordinary_sgpr_base == 0u))
      return false;
    if (static_cast<uint32_t>(profile.reserved_ordinary_sgpr_base) +
            profile.reserved_ordinary_sgpr_count >
        profile.ordinary_sgpr_limit)
      return false;
    for (std::size_t rhs = lhs + 1; rhs < profiles.size(); ++rhs) {
      if (profile.target == profiles[rhs].target || profile.arch == profiles[rhs].arch)
        return false;
    }
  }
  return true;
}

[[nodiscard]] bool target_profiles_are_valid();

[[nodiscard]] inline rj_code_arch_t arch_for_target(rj_code_target_id_t target) {
  const TargetProfile *profile = target_profile(target);
  return profile ? profile->arch : ROCJITSU_CODE_ARCH_INVALID;
}

[[nodiscard]] inline bool is_capability_arch(rj_code_arch_t arch) {
  return target_profile(arch) != nullptr;
}

/// Return whether a target can recover system-SGPR payload that overflows the
/// descriptor's ordinary preload window. Entry-prologue lowering uses this
/// capability instead of naming the current targets that implement it.
[[nodiscard]] inline bool arch_supports_kernarg_preload_overflow_recovery(rj_code_arch_t arch) {
  const TargetProfile *profile = target_profile(arch);
  return profile && profile->supports_kernarg_preload_overflow_recovery;
}

[[nodiscard]] constexpr bool profile_reserved_sgpr_range_overlaps(const TargetProfile &profile,
                                                                  uint16_t base, uint16_t count) {
  return count != 0u && profile.reserved_ordinary_sgpr_count != 0u &&
         static_cast<uint32_t>(base) + count > profile.reserved_ordinary_sgpr_base &&
         static_cast<uint32_t>(profile.reserved_ordinary_sgpr_base) +
                 profile.reserved_ordinary_sgpr_count >
             base;
}

[[nodiscard]] constexpr std::optional<uint32_t>
profile_normalize_private_size(const TargetProfile &profile, uint32_t requested_bytes) {
  const uint64_t granularity = profile.private_allocation_granularity_bytes;
  const uint64_t normalized =
      (static_cast<uint64_t>(requested_bytes) + granularity - 1u) / granularity * granularity;
  if (normalized > profile.address_free_private_limit_bytes)
    return std::nullopt;
  return static_cast<uint32_t>(normalized);
}

[[nodiscard]] inline std::optional<uint32_t> address_free_private_limit(rj_code_arch_t arch) {
  const TargetProfile *profile = target_profile(arch);
  return profile ? std::optional<uint32_t>(profile->address_free_private_limit_bytes)
                 : std::nullopt;
}

[[nodiscard]] inline std::optional<uint32_t>
normalize_address_free_private_size(rj_code_arch_t arch, uint32_t requested_bytes) {
  const TargetProfile *profile = target_profile(arch);
  return profile ? profile_normalize_private_size(*profile, requested_bytes) : std::nullopt;
}

[[nodiscard]] inline bool arch_has_cluster_facilities(rj_code_arch_t arch) {
  const TargetProfile *profile = target_profile(arch);
  return profile && profile->has_cluster_facilities;
}

[[nodiscard]] inline bool arch_has_selectable_vgpr_bank(rj_code_arch_t arch) {
  const TargetProfile *profile = target_profile(arch);
  return profile && profile->has_selectable_vgpr_bank;
}

[[nodiscard]] constexpr CapabilityDomain capability_domain(CapabilityForm form) {
  switch (form) {
  case CapabilityForm::NativeLdsAccess:
  case CapabilityForm::GroupFlatAccess:
    return CapabilityDomain::Access;
  case CapabilityForm::WorkgroupBarrier:
  case CapabilityForm::ClusterBarrier:
    return CapabilityDomain::Barrier;
  case CapabilityForm::OrderedFlatAtomic:
  case CapabilityForm::OrderedVglobalAtomic:
  case CapabilityForm::OrderedLdsAtomic:
  case CapabilityForm::RelaxedLdsAtomicAccess:
    return CapabilityDomain::Atomic;
  case CapabilityForm::AddressedOrdinaryFence:
    return CapabilityDomain::Fence;
  case CapabilityForm::Count:
    return CapabilityDomain::Count;
  }
  return CapabilityDomain::Count;
}

/// Target-family availability shared with production admission. Exact
/// mnemonic, encoding, scope, and operand checks remain in the decoded
/// inventory and lowerers.
[[nodiscard]] inline bool arch_supports_capability_form(rj_code_arch_t arch, CapabilityForm form) {
  const TargetProfile *profile = target_profile(arch);
  const uint16_t form_bit = capability_form_bit(form);
  return profile && form_bit != 0u && (profile->semantic_form_mask & form_bit) != 0u;
}

/// Return the documented high-level disposition. Exact instruction admission
/// still comes from the decoded semantic inventory and mode lowerers; this
/// function projects their stable target/mode contract without duplicating
/// target-native mnemonic lists.
[[nodiscard]] inline CapabilityDisposition capability_disposition(rj_code_target_id_t target,
                                                                  Mode mode, CapabilityForm form) {
  const rj_code_arch_t arch = arch_for_target(target);
  if (arch == ROCJITSU_CODE_ARCH_INVALID || !enabled_mode(mode) ||
      static_cast<uint8_t>(form) >= static_cast<uint8_t>(CapabilityForm::Count))
    return CapabilityDisposition::OutOfContract;
  if (!arch_supports_capability_form(arch, form))
    return CapabilityDisposition::NotApplicable;

  const bool supercollider = mode == Mode::SuperCollider;

  switch (form) {
  case CapabilityForm::NativeLdsAccess:
  case CapabilityForm::GroupFlatAccess:
    return CapabilityDisposition::Supported;
  case CapabilityForm::WorkgroupBarrier:
    return supercollider ? CapabilityDisposition::MutationOnly : CapabilityDisposition::Supported;
  case CapabilityForm::ClusterBarrier:
    return supercollider ? CapabilityDisposition::MutationOnly : CapabilityDisposition::Supported;
  case CapabilityForm::OrderedFlatAtomic:
    return supercollider ? CapabilityDisposition::MutationOnly : CapabilityDisposition::Supported;
  case CapabilityForm::OrderedVglobalAtomic:
    return supercollider ? CapabilityDisposition::MutationOnly : CapabilityDisposition::Supported;
  case CapabilityForm::OrderedLdsAtomic:
    return supercollider ? CapabilityDisposition::MutationOnly : CapabilityDisposition::Supported;
  case CapabilityForm::RelaxedLdsAtomicAccess:
    return supercollider ? CapabilityDisposition::NotApplicable : CapabilityDisposition::AccessOnly;
  case CapabilityForm::AddressedOrdinaryFence:
    if (supercollider)
      return CapabilityDisposition::MutationOnly;
    return CapabilityDisposition::AssociatedOnly;
  case CapabilityForm::Count:
    return CapabilityDisposition::OutOfContract;
  }
  return CapabilityDisposition::OutOfContract;
}

[[nodiscard]] constexpr std::string_view mode_label(Mode mode) { return kEnabledModes.name(mode); }

[[nodiscard]] constexpr std::string_view capability_form_name(CapabilityForm form) {
  return kCapabilityForms.name(form);
}

inline constexpr auto kCapabilityDispositions = [] {
  using E = CapabilityDisposition;
  return make_enum_vocabulary(
      "unknown", enum_entry(E::OutOfContract, "out of contract"),
      enum_entry(E::NotApplicable, "not applicable"), enum_entry(E::Supported, "supported"),
      enum_entry(E::MutationOnly, "mutation only"), enum_entry(E::AccessOnly, "access only"),
      enum_entry(E::AssociatedOnly, "associated only"));
}();

[[nodiscard]] constexpr std::string_view
capability_disposition_name(CapabilityDisposition disposition) {
  return kCapabilityDispositions.name(disposition);
}

static_assert(enabled_modes_are_complete(kEnabledModes));
static_assert(capability_enum_is_complete(kCapabilityDomains));
static_assert(capability_enum_is_complete(kCapabilityForms));

} // namespace rocjitsu::consan
