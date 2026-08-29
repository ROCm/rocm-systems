// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_target_address.cpp
/// @brief Target-aware planning and emission of normalized MOI atomic addresses.

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <bit>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace rocjitsu {

std::string_view consan_moi_atomic_address_support_name(ConSanMoiAtomicAddressSupport support) {
  switch (support) {
  case ConSanMoiAtomicAddressSupport::Supported:
    return "supported";
  case ConSanMoiAtomicAddressSupport::UnsupportedArchitecture:
    return "unsupported-architecture";
  case ConSanMoiAtomicAddressSupport::UnsupportedAddressKind:
    return "unsupported-address-kind";
  case ConSanMoiAtomicAddressSupport::UnsupportedWidth:
    return "unsupported-width";
  case ConSanMoiAtomicAddressSupport::UnsupportedEncoding:
    return "unsupported-encoding";
  case ConSanMoiAtomicAddressSupport::MissingAddressOperands:
    return "missing-address-operands";
  case ConSanMoiAtomicAddressSupport::UnsupportedInputWidth:
    return "unsupported-input-width";
  case ConSanMoiAtomicAddressSupport::UnsupportedOffset:
    return "unsupported-offset";
  case ConSanMoiAtomicAddressSupport::UnsupportedScope:
    return "unsupported-scope";
  case ConSanMoiAtomicAddressSupport::UnsupportedResourcePlan:
    return "unsupported-resource-plan";
  case ConSanMoiAtomicAddressSupport::UnsupportedScratchShape:
    return "unsupported-scratch-shape";
  case ConSanMoiAtomicAddressSupport::ResultAddressAlias:
    return "result-address-alias";
  case ConSanMoiAtomicAddressSupport::ScratchOperandAlias:
    return "scratch-operand-alias";
  }
  return "unknown";
}

ConSanMoiAtomicAddressPlan plan_consan_moi_atomic_address(
    const ConSanAtomicLoweringForm &form, uint16_t scratch_vgpr, uint16_t scratch_vgpr_count,
    ConSanRegisterAllocationSource resource_source, bool allow_post_guest_spill_operand_overlap) {
  ConSanMoiAtomicAddressPlan plan;
  plan.scratch_vgpr = scratch_vgpr;
  plan.scratch_vgpr_count = scratch_vgpr_count;
  plan.resource_source = resource_source;
  const auto reject = [&](ConSanMoiAtomicAddressSupport support) {
    plan.kind = ConSanMoiAtomicAddressKind::Unsupported;
    plan.support = support;
    return plan;
  };
  const auto overlaps = [](uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                           uint16_t rhs_count) {
    return static_cast<uint32_t>(lhs_base) + lhs_count > rhs_base &&
           static_cast<uint32_t>(rhs_base) + rhs_count > lhs_base;
  };
  const auto usable_resource_source = [](ConSanRegisterAllocationSource source) {
    return source == ConSanRegisterAllocationSource::Explicit ||
           source == ConSanRegisterAllocationSource::LivenessDead ||
           source == ConSanRegisterAllocationSource::DescriptorGrowth ||
           source == ConSanRegisterAllocationSource::SpillRequired;
  };
  if (!usable_resource_source(resource_source))
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedResourcePlan);
  if (form.kind == ConSanAtomicLoweringFormKind::Count || form.value_width_bits == 0u ||
      form.value_register_count == 0u || form.data_register_count == 0u ||
      form.address_vgpr_count == 0u)
    return reject(ConSanMoiAtomicAddressSupport::UnsupportedEncoding);

  const auto valid_scratch = [&](uint16_t compact_count) {
    return (scratch_vgpr_count == compact_count || scratch_vgpr_count >= 7u) &&
           static_cast<uint32_t>(scratch_vgpr) + scratch_vgpr_count <= 256u;
  };
  const auto result_tail = [&] {
    return static_cast<uint16_t>(scratch_vgpr + scratch_vgpr_count - 2u);
  };
  const auto overlaps_operands = [&](uint16_t checked_scratch_count, bool include_address) {
    return (include_address && overlaps(scratch_vgpr, checked_scratch_count, form.address_vgpr,
                                        form.address_vgpr_count)) ||
           overlaps(scratch_vgpr, scratch_vgpr_count, form.data_vgpr, form.data_register_count) ||
           (form.destination_vgpr && form.destination_register_count != 0u &&
            overlaps(scratch_vgpr, scratch_vgpr_count, *form.destination_vgpr,
                     form.destination_register_count));
  };
  const bool returned_value_aliases_address =
      form.returns_old_value && form.destination_vgpr &&
      overlaps(form.address_vgpr, form.address_vgpr_count, *form.destination_vgpr,
               form.destination_register_count);

  if (form.kind == ConSanAtomicLoweringFormKind::LdsVectorOffset) {
    if ((scratch_vgpr_count != 5u && scratch_vgpr_count < 7u) ||
        static_cast<uint32_t>(scratch_vgpr) + scratch_vgpr_count > 256u)
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
    const uint16_t result_address_vgpr = result_tail();
    if (overlaps(result_address_vgpr, 2u, form.address_vgpr, form.address_vgpr_count))
      return reject(ConSanMoiAtomicAddressSupport::ResultAddressAlias);
    if (!allow_post_guest_spill_operand_overlap &&
        overlaps_operands(static_cast<uint16_t>(scratch_vgpr_count - 2u), true))
      return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
    plan.kind = ConSanMoiAtomicAddressKind::LdsByteOffsetToken;
    plan.support = ConSanMoiAtomicAddressSupport::Supported;
    plan.input_address_vgpr = form.address_vgpr;
    plan.input_address_vgpr_count = form.address_vgpr_count;
    plan.signed_byte_offset = form.signed_byte_offset;
    plan.result_address_vgpr = result_address_vgpr;
    plan.result_address_vgpr_count = 2u;
    return plan;
  }
  if (form.kind == ConSanAtomicLoweringFormKind::BufferResourceVectorOffset) {
    if (!form.scalar_base_sgpr)
      return reject(ConSanMoiAtomicAddressSupport::MissingAddressOperands);
    if (!valid_scratch(5u))
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
    const uint16_t result_address_vgpr = result_tail();
    if (!allow_post_guest_spill_operand_overlap && overlaps_operands(0u, false))
      return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
    plan.kind = ConSanMoiAtomicAddressKind::BufferResourceMaterialized;
    plan.support = ConSanMoiAtomicAddressSupport::Supported;
    plan.input_address_vgpr = form.address_vgpr;
    plan.input_address_vgpr_count = form.address_vgpr_count;
    plan.scalar_base_sgpr = form.scalar_base_sgpr;
    plan.scalar_offset_sgpr = form.scalar_offset_sgpr;
    plan.signed_byte_offset = form.signed_byte_offset;
    plan.result_address_vgpr = result_address_vgpr;
    plan.result_address_vgpr_count = 2u;
    return plan;
  }
  if (form.kind == ConSanAtomicLoweringFormKind::FlatScalarVectorAddress ||
      form.kind == ConSanAtomicLoweringFormKind::GlobalScalarVectorAddress) {
    if (!form.scalar_base_sgpr)
      return reject(ConSanMoiAtomicAddressSupport::MissingAddressOperands);
    if (form.kind == ConSanAtomicLoweringFormKind::FlatScalarVectorAddress &&
        form.returns_old_value)
      return reject(ConSanMoiAtomicAddressSupport::ResultAddressAlias);
    if (!valid_scratch(5u))
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
    const uint16_t result_address_vgpr = result_tail();
    if (overlaps(result_address_vgpr, 2u, form.address_vgpr, form.address_vgpr_count))
      return reject(ConSanMoiAtomicAddressSupport::ResultAddressAlias);
    if (!allow_post_guest_spill_operand_overlap &&
        overlaps_operands(static_cast<uint16_t>(scratch_vgpr_count - 2u), true))
      return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
    plan.kind = ConSanMoiAtomicAddressKind::VglobalMaterialized;
    plan.support = ConSanMoiAtomicAddressSupport::Supported;
    plan.input_address_vgpr = form.address_vgpr;
    plan.input_address_vgpr_count = form.address_vgpr_count;
    if (form.scale_vector_offset)
      plan.input_address_scale = static_cast<uint16_t>(form.value_width_bits / 8u);
    plan.sign_extend_vector_offset = form.sign_extend_vector_offset;
    plan.scalar_base_sgpr = form.scalar_base_sgpr;
    plan.signed_byte_offset = form.signed_byte_offset;
    plan.result_address_vgpr = result_address_vgpr;
    plan.result_address_vgpr_count = 2u;
    return plan;
  }

  if (form.kind == ConSanAtomicLoweringFormKind::FlatVectorAddress) {
    const uint16_t minimum_scratch_count = returned_value_aliases_address ? 5u : 3u;
    if (!valid_scratch(minimum_scratch_count))
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
    if (!allow_post_guest_spill_operand_overlap && overlaps_operands(scratch_vgpr_count, true))
      return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
    plan.kind = returned_value_aliases_address
                    ? ConSanMoiAtomicAddressKind::FlatGuestPairMaterialized
                    : ConSanMoiAtomicAddressKind::FlatGuestPair;
    plan.support = ConSanMoiAtomicAddressSupport::Supported;
    plan.input_address_vgpr = form.address_vgpr;
    plan.input_address_vgpr_count = form.address_vgpr_count;
    plan.signed_byte_offset = 0;
    plan.result_address_vgpr = returned_value_aliases_address
                                   ? static_cast<uint16_t>(scratch_vgpr + scratch_vgpr_count - 2u)
                                   : form.address_vgpr;
    plan.result_address_vgpr_count = 2u;
    return plan;
  }

  if (form.kind == ConSanAtomicLoweringFormKind::GlobalVectorAddress) {
    const bool requires_materialization =
        form.signed_byte_offset != 0 || returned_value_aliases_address;
    const uint16_t minimum_scratch_count = requires_materialization ? 5u : 3u;
    if (!valid_scratch(minimum_scratch_count))
      return reject(ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
    if (!requires_materialization) {
      if (!allow_post_guest_spill_operand_overlap && overlaps_operands(scratch_vgpr_count, true))
        return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
      plan.kind = ConSanMoiAtomicAddressKind::VglobalGuestPair;
      plan.support = ConSanMoiAtomicAddressSupport::Supported;
      plan.input_address_vgpr = form.address_vgpr;
      plan.input_address_vgpr_count = form.address_vgpr_count;
      plan.signed_byte_offset = 0;
      plan.result_address_vgpr = form.address_vgpr;
      plan.result_address_vgpr_count = 2u;
      return plan;
    }
    const uint16_t result_address_vgpr = result_tail();
    if (overlaps(result_address_vgpr, 2u, form.address_vgpr, form.address_vgpr_count))
      return reject(ConSanMoiAtomicAddressSupport::ResultAddressAlias);
    if (!allow_post_guest_spill_operand_overlap &&
        overlaps_operands(static_cast<uint16_t>(scratch_vgpr_count - 2u), true))
      return reject(ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
    plan.kind = ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized;
    plan.support = ConSanMoiAtomicAddressSupport::Supported;
    plan.input_address_vgpr = form.address_vgpr;
    plan.input_address_vgpr_count = form.address_vgpr_count;
    plan.signed_byte_offset = form.signed_byte_offset;
    plan.result_address_vgpr = result_address_vgpr;
    plan.result_address_vgpr_count = 2u;
    return plan;
  }
  return reject(ConSanMoiAtomicAddressSupport::UnsupportedAddressKind);
}

ConSanMoiAtomicAddressPlan
plan_consan_moi_atomic_address(const ConSanAtomicSite &site, uint16_t scratch_vgpr,
                               uint16_t scratch_vgpr_count,
                               ConSanRegisterAllocationSource resource_source, rj_code_arch_t arch,
                               bool allow_post_guest_spill_operand_overlap) {
  const auto rejected = [&](ConSanMoiAtomicAddressSupport support) {
    ConSanMoiAtomicAddressPlan result;
    result.kind = ConSanMoiAtomicAddressKind::Unsupported;
    result.support = support;
    result.scratch_vgpr = scratch_vgpr;
    result.scratch_vgpr_count = scratch_vgpr_count;
    result.resource_source = resource_source;
    return result;
  };
  const auto map_reason = [](ConSanAtomicClassifierReason reason) {
    switch (reason) {
    case ConSanAtomicClassifierReason::None:
      return ConSanMoiAtomicAddressSupport::Supported;
    case ConSanAtomicClassifierReason::UnsupportedAddressSource:
      return ConSanMoiAtomicAddressSupport::UnsupportedAddressKind;
    case ConSanAtomicClassifierReason::InvalidAccessWidth:
      return ConSanMoiAtomicAddressSupport::UnsupportedWidth;
    case ConSanAtomicClassifierReason::UnsupportedEncoding:
      return ConSanMoiAtomicAddressSupport::UnsupportedEncoding;
    case ConSanAtomicClassifierReason::NonzeroImmediateOffset:
    case ConSanAtomicClassifierReason::UnsupportedOffset:
      return ConSanMoiAtomicAddressSupport::UnsupportedOffset;
    case ConSanAtomicClassifierReason::MissingOperands:
    case ConSanAtomicClassifierReason::CompareExchangeOutcomeUnavailable:
      return ConSanMoiAtomicAddressSupport::MissingAddressOperands;
    case ConSanAtomicClassifierReason::UnsupportedInputWidth:
      return ConSanMoiAtomicAddressSupport::UnsupportedInputWidth;
    case ConSanAtomicClassifierReason::ResultAddressAlias:
      return ConSanMoiAtomicAddressSupport::ResultAddressAlias;
    case ConSanAtomicClassifierReason::MissingOrderingMetadata:
    case ConSanAtomicClassifierReason::UnsupportedScope:
      return ConSanMoiAtomicAddressSupport::UnsupportedScope;
    case ConSanAtomicClassifierReason::TargetUnavailable:
      return ConSanMoiAtomicAddressSupport::UnsupportedArchitecture;
    case ConSanAtomicClassifierReason::Count:
      break;
    }
    return ConSanMoiAtomicAddressSupport::UnsupportedEncoding;
  };

  const bool ordinary =
      site.mnemonic.find("atomic") == std::string::npos && !site.mnemonic.starts_with("ds_");
  const ConSanAtomicLoweringClassification classification =
      classify_consan_atomic_lowering(site, arch, !ordinary);
  if (!classification.normalized()) {
    if (classification.normalization_reason ==
            ConSanAtomicClassifierReason::UnsupportedAddressSource &&
        ((site.mnemonic.starts_with("ds_") && arch != ROCJITSU_CODE_ARCH_CDNA5) ||
         (site.mnemonic.starts_with("buffer_") && arch != ROCJITSU_CODE_ARCH_CDNA5))) {
      return rejected(ConSanMoiAtomicAddressSupport::UnsupportedArchitecture);
    }
    return rejected(map_reason(classification.normalization_reason));
  }
  if (!classification.address_available())
    return rejected(map_reason(classification.address_reason));
  if (classification.causal_ordering_reason ==
          ConSanAtomicClassifierReason::MissingOrderingMetadata ||
      classification.causal_ordering_reason == ConSanAtomicClassifierReason::UnsupportedScope) {
    return rejected(map_reason(classification.causal_ordering_reason));
  }
  return plan_consan_moi_atomic_address(*classification.form, scratch_vgpr, scratch_vgpr_count,
                                        resource_source, allow_post_guest_spill_operand_overlap);
}

std::optional<std::vector<uint32_t>>
build_consan_moi_atomic_address_materialization(const ConSanMoiAtomicAddressPlan &plan,
                                                uint16_t vcc_save_sgpr, uint16_t scc_save_sgpr,
                                                rj_code_arch_t arch) {
  if (!plan.supported())
    return std::nullopt;
  if (!plan.requires_materialization())
    return std::vector<uint32_t>{};
  // Every current recipe returns its materialized address inside the declared
  // scratch allocation. Validate that shared layout before per-kind emission.
  const uint32_t scratch_end = static_cast<uint32_t>(plan.scratch_vgpr) + plan.scratch_vgpr_count;
  const uint32_t result_end =
      static_cast<uint32_t>(plan.result_address_vgpr) + plan.result_address_vgpr_count;
  if (scratch_end > 256u || plan.result_address_vgpr < plan.scratch_vgpr ||
      result_end > scratch_end)
    return std::nullopt;
  if (plan.kind == ConSanMoiAtomicAddressKind::LdsByteOffsetToken) {
    if (arch != ROCJITSU_CODE_ARCH_CDNA5 || plan.input_address_vgpr_count != 1u ||
        plan.result_address_vgpr_count != 2u || plan.result_address_vgpr >= 255u ||
        plan.signed_byte_offset < 0 || plan.signed_byte_offset > 0xff)
      return std::nullopt;
    std::vector<uint32_t> words;
    words.push_back(build_v_mov_b32_e32(plan.result_address_vgpr,
                                        vector_source_vgpr(plan.input_address_vgpr), arch));
    if (plan.signed_byte_offset != 0) {
      const auto add = instrumentation::build_v_add_u32_literal(
          plan.result_address_vgpr, static_cast<uint32_t>(plan.signed_byte_offset),
          plan.result_address_vgpr, arch);
      if (!add)
        return std::nullopt;
      words.insert(words.end(), add->begin(), add->end());
    }
    const auto tag = build_v_mov_b32_e64_literal(
        static_cast<uint16_t>(plan.result_address_vgpr + 1u), kConSanMoiLdsAddressTokenTag, arch);
    if (!tag)
      return std::nullopt;
    words.insert(words.end(), tag->begin(), tag->end());
    return words;
  }
  const bool legacy_address_materialization =
      (consan_uses_gfx9_cdna_encoding(arch) || arch == ROCJITSU_CODE_ARCH_RDNA3) &&
      (plan.kind == ConSanMoiAtomicAddressKind::FlatGuestPairMaterialized ||
       plan.kind == ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized ||
       plan.kind == ConSanMoiAtomicAddressKind::VglobalMaterialized);
  if (!consan_uses_gfx12_encoding(arch) && !legacy_address_materialization)
    return std::nullopt;
  const bool buffer_resource = plan.kind == ConSanMoiAtomicAddressKind::BufferResourceMaterialized;
  const bool scalar_vector =
      plan.kind == ConSanMoiAtomicAddressKind::VglobalMaterialized || buffer_resource;
  const bool vector_pair = plan.kind == ConSanMoiAtomicAddressKind::FlatGuestPairMaterialized ||
                           plan.kind == ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized;
  const bool supported_scaled_vglobal =
      arch == ROCJITSU_CODE_ARCH_CDNA5 &&
      plan.kind == ConSanMoiAtomicAddressKind::VglobalMaterialized &&
      (plan.input_address_scale == 4u || plan.input_address_scale == 8u);
  if ((!scalar_vector && !vector_pair) ||
      plan.input_address_vgpr_count != (scalar_vector ? 1u : 2u) ||
      (plan.input_address_scale != 1u && !supported_scaled_vglobal) ||
      plan.result_address_vgpr_count != 2u || plan.result_address_vgpr >= 255u ||
      vcc_save_sgpr >= 105u || scc_save_sgpr >= 106u || scc_save_sgpr == vcc_save_sgpr ||
      scc_save_sgpr == vcc_save_sgpr + 1u ||
      (scalar_vector &&
       (!plan.scalar_base_sgpr ||
        (*plan.scalar_base_sgpr >= vcc_save_sgpr && *plan.scalar_base_sgpr <= vcc_save_sgpr + 1u) ||
        (*plan.scalar_base_sgpr + 1u >= vcc_save_sgpr &&
         *plan.scalar_base_sgpr + 1u <= vcc_save_sgpr + 1u) ||
        scc_save_sgpr == *plan.scalar_base_sgpr || scc_save_sgpr == *plan.scalar_base_sgpr + 1u)) ||
      (buffer_resource && plan.scalar_offset_sgpr &&
       ((*plan.scalar_offset_sgpr >= vcc_save_sgpr &&
         *plan.scalar_offset_sgpr <= vcc_save_sgpr + 1u) ||
        scc_save_sgpr == *plan.scalar_offset_sgpr)))
    return std::nullopt;

  constexpr uint16_t kVccLo = 106u;
  const auto save_scc = instrumentation::build_s_cselect_b32(
      scc_save_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  const auto save_vcc = instrumentation::build_s_mov_b64(vcc_save_sgpr, kVccLo, arch);
  const auto restore_vcc = instrumentation::build_s_mov_b64(kVccLo, vcc_save_sgpr, arch);
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(scc_save_sgpr, scalar_positive_inline_u32(0), arch);
  if (!save_scc || !save_vcc || !restore_vcc || !restore_scc)
    return std::nullopt;

  std::vector<uint32_t> words;
  words.reserve(16u);
  words.push_back(*save_scc);
  words.push_back(*save_vcc);
  if (scalar_vector) {
    uint16_t offset_vgpr = plan.input_address_vgpr;
    if (plan.input_address_scale != 1u) {
      const uint16_t scaled_offset_vgpr = plan.scratch_vgpr;
      if (scaled_offset_vgpr >= plan.result_address_vgpr)
        return std::nullopt;
      const auto scale = instrumentation::build_v_lshlrev_b32(
          scaled_offset_vgpr,
          scalar_positive_inline_u32(std::countr_zero(plan.input_address_scale)),
          plan.input_address_vgpr, arch);
      if (!scale)
        return std::nullopt;
      words.push_back(*scale);
      offset_vgpr = scaled_offset_vgpr;
    }
    if (buffer_resource && plan.input_address_vgpr >= plan.result_address_vgpr &&
        plan.input_address_vgpr < plan.result_address_vgpr + 2u) {
      offset_vgpr = plan.scratch_vgpr;
      if (offset_vgpr == plan.input_address_vgpr || offset_vgpr >= plan.result_address_vgpr)
        return std::nullopt;
      words.push_back(
          build_v_mov_b32_e32(offset_vgpr, vector_source_vgpr(plan.input_address_vgpr), arch));
    }
    words.push_back(build_v_mov_b32_e32(plan.result_address_vgpr, *plan.scalar_base_sgpr, arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(plan.result_address_vgpr + 1u),
                                        static_cast<uint16_t>(*plan.scalar_base_sgpr + 1u), arch));
    std::optional<std::vector<uint32_t>> add_vaddr;
    if (plan.sign_extend_vector_offset) {
      // The sign scratch needs two scratch words before the result pair. Keep
      // a defensive check for externally constructed plans.
      if (plan.result_address_vgpr <= plan.scratch_vgpr + 1u)
        return std::nullopt;
      uint16_t sign_vgpr = plan.scratch_vgpr;
      if (sign_vgpr == offset_vgpr)
        ++sign_vgpr;
      // The spacing check and shared window invariant keep this temporary
      // inside the allocation and below the result pair.
      add_vaddr = instrumentation::build_v_add_u64_signed_vgpr_offset(plan.result_address_vgpr,
                                                                      offset_vgpr, sign_vgpr, arch);
    } else {
      add_vaddr =
          instrumentation::build_v_add_u64_vgpr_offset(plan.result_address_vgpr, offset_vgpr, arch);
    }
    if (!add_vaddr)
      return std::nullopt;
    words.insert(words.end(), add_vaddr->begin(), add_vaddr->end());
    if (buffer_resource && plan.scalar_offset_sgpr) {
      const uint16_t scalar_offset_vgpr = plan.scratch_vgpr;
      if (scalar_offset_vgpr == plan.result_address_vgpr ||
          scalar_offset_vgpr == plan.result_address_vgpr + 1u)
        return std::nullopt;
      words.push_back(build_v_mov_b32_e32(scalar_offset_vgpr, *plan.scalar_offset_sgpr, arch));
      const auto add_scalar_offset = instrumentation::build_v_add_u64_vgpr_offset(
          plan.result_address_vgpr, scalar_offset_vgpr, arch);
      if (!add_scalar_offset)
        return std::nullopt;
      words.insert(words.end(), add_scalar_offset->begin(), add_scalar_offset->end());
    }
  } else {
    words.push_back(build_v_mov_b32_e32(plan.result_address_vgpr,
                                        vector_source_vgpr(plan.input_address_vgpr), arch));
    words.push_back(build_v_mov_b32_e32(
        static_cast<uint16_t>(plan.result_address_vgpr + 1u),
        vector_source_vgpr(static_cast<uint16_t>(plan.input_address_vgpr + 1u)), arch));
  }
  if (plan.signed_byte_offset != 0) {
    const auto add_displacement = instrumentation::build_v_add_u64_signed_i24(
        plan.result_address_vgpr, plan.signed_byte_offset, arch);
    if (!add_displacement)
      return std::nullopt;
    words.insert(words.end(), add_displacement->begin(), add_displacement->end());
  }
  words.push_back(*restore_vcc);
  // Keep SCC restoration last: scalar comparisons after this point would
  // overwrite the guest condition code again.
  words.push_back(*restore_scc);
  return words;
}

} // namespace rocjitsu
