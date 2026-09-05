// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_atomic_classifier.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops.h"

namespace rocjitsu {
namespace {

using Reason = ConSanAtomicClassifierReason;

[[nodiscard]] ConSanAtomicLoweringClassification reject(Reason reason) {
  return {
      .form = std::nullopt,
      .normalization_reason = reason,
      .address_reason = reason,
      .exact_ordering_reason = reason,
      .causal_ordering_reason = reason,
  };
}

[[nodiscard]] ConSanAtomicLoweringClassification normalized(ConSanAtomicLoweringForm form,
                                                            Reason address_reason,
                                                            Reason exact_reason,
                                                            Reason causal_reason) {
  return {
      .form = std::move(form),
      .normalization_reason = Reason::None,
      .address_reason = address_reason,
      .exact_ordering_reason = exact_reason,
      .causal_ordering_reason = causal_reason,
  };
}

[[nodiscard]] bool is_flat_family(const ConSanAtomicSite &site, bool is_rmw) {
  return site.mnemonic.starts_with("flat_atomic") ||
         (!is_rmw &&
          (site.mnemonic.starts_with("flat_load") || site.mnemonic.starts_with("flat_store")));
}

[[nodiscard]] bool is_global_family(const ConSanAtomicSite &site, bool is_rmw) {
  return site.mnemonic.starts_with("global_atomic") ||
         (!is_rmw &&
          (site.mnemonic.starts_with("global_load") || site.mnemonic.starts_with("global_store")));
}

[[nodiscard]] Reason causal_reason(const ConSanAtomicSite &site,
                                   const ConSanAtomicLoweringForm &form) {
  if (!site.scope)
    return Reason::MissingOrderingMetadata;
  if (!consan_memory_scope_is_supported(*site.scope) || *site.scope == ConSanMemoryScope::Wavefront)
    return Reason::UnsupportedScope;
  if (form.kind == ConSanAtomicLoweringFormKind::LdsVectorOffset &&
      *site.scope != ConSanMemoryScope::Workgroup)
    return Reason::UnsupportedScope;
  if (form.compare_exchange && (!site.returns_old_value.value_or(false) || !site.destination_vgpr))
    return Reason::CompareExchangeOutcomeUnavailable;
  return Reason::None;
}

[[nodiscard]] Reason exact_reason(const ConSanAtomicSite &site,
                                  const ConSanAtomicLoweringForm &form) {
  if (form.kind != ConSanAtomicLoweringFormKind::FlatVectorAddress &&
      form.kind != ConSanAtomicLoweringFormKind::FlatScalarVectorAddress &&
      form.kind != ConSanAtomicLoweringFormKind::GlobalVectorAddress &&
      form.kind != ConSanAtomicLoweringFormKind::GlobalScalarVectorAddress)
    return Reason::UnsupportedAddressSource;
  if (form.value_width_bits != 32u)
    return Reason::InvalidAccessWidth;
  if ((form.kind == ConSanAtomicLoweringFormKind::FlatVectorAddress ||
       form.kind == ConSanAtomicLoweringFormKind::FlatScalarVectorAddress) &&
      form.signed_byte_offset != 0)
    return Reason::NonzeroImmediateOffset;
  if (form.compare_exchange && site.returns_old_value && !*site.returns_old_value)
    return Reason::CompareExchangeOutcomeUnavailable;
  if (form.compare_exchange && !site.destination_vgpr)
    return Reason::MissingOperands;
  if (!site.scope || !site.raw_th || !site.returns_old_value)
    return Reason::MissingOrderingMetadata;
  if (!consan_memory_scope_is_supported(*site.scope) || *site.scope == ConSanMemoryScope::Wavefront)
    return Reason::UnsupportedScope;
  return Reason::None;
}

} // namespace

ConSanAtomicLoweringClassification
classify_consan_atomic_lowering(const ConSanAtomicSite &site, rj_code_arch_t arch, bool is_rmw) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (!target)
    return reject(Reason::TargetUnavailable);
  const ConSanVectorMemoryCapability &memory = target->vector_memory;

  const bool compare_exchange = is_rmw && consan_atomic_is_compare_exchange(site);
  const uint16_t value_register_count = static_cast<uint16_t>((site.width_bits + 31u) / 32u);
  const uint16_t data_register_count =
      static_cast<uint16_t>(value_register_count * (compare_exchange ? 2u : 1u));
  const bool ordinary_load =
      !is_rmw && site.destination_vgpr && site.data_vgpr && site.destination_vgpr == site.data_vgpr;
  const uint16_t destination_register_count =
      site.returns_old_value.value_or(false) || ordinary_load ? value_register_count : 0u;
  if ((site.data_vgpr && static_cast<uint32_t>(*site.data_vgpr) + data_register_count > 256u) ||
      (site.destination_vgpr && destination_register_count != 0u &&
       static_cast<uint32_t>(*site.destination_vgpr) + destination_register_count > 256u)) {
    return reject(Reason::UnsupportedInputWidth);
  }
  const auto finish = [&](ConSanAtomicLoweringForm form, Reason address_reason = Reason::None) {
    return normalized(form, address_reason,
                      address_reason == Reason::None ? exact_reason(site, form) : address_reason,
                      address_reason == Reason::None ? causal_reason(site, form) : address_reason);
  };

  if (site.mnemonic.starts_with("ds_")) {
    if (!target->atomic_address_materialization.lds_byte_offset_token)
      return reject(Reason::UnsupportedAddressSource);
    if (site.width_bits != 32u)
      return reject(Reason::InvalidAccessWidth);
    if (site.size != 2u * sizeof(uint32_t) || !site.raw_addr || !site.raw_data0 ||
        !site.raw_ioffset)
      return reject(Reason::UnsupportedEncoding);
    if (!site.address_vgpr || !site.data_vgpr || *site.raw_addr != *site.address_vgpr ||
        *site.raw_data0 != *site.data_vgpr)
      return reject(Reason::MissingOperands);
    if (*site.raw_ioffset < 0 || *site.raw_ioffset > 0xff)
      return reject(Reason::UnsupportedOffset);
    return finish({
        .kind = ConSanAtomicLoweringFormKind::LdsVectorOffset,
        .instruction_size = site.size,
        .value_width_bits = site.width_bits,
        .value_register_count = value_register_count,
        .data_register_count = data_register_count,
        .destination_register_count = destination_register_count,
        .address_vgpr = *site.address_vgpr,
        .address_vgpr_count = 1u,
        .data_vgpr = *site.data_vgpr,
        .destination_vgpr = site.destination_vgpr,
        .scalar_base_sgpr = std::nullopt,
        .scalar_offset_sgpr = std::nullopt,
        .signed_byte_offset = *site.raw_ioffset,
        .is_rmw = is_rmw,
        .compare_exchange = compare_exchange,
        .returns_old_value = site.returns_old_value.value_or(false),
    });
  }

  if (site.mnemonic.starts_with("buffer_")) {
    constexpr uint32_t kNullScalarOffset = 0x7cu;
    constexpr int32_t kSigned24Min = -(1 << 23);
    constexpr int32_t kSigned24Max = (1 << 23) - 1;
    if (!target->program_analysis || !target->program_analysis->decode_buffer_memory)
      return reject(Reason::UnsupportedAddressSource);
    if (site.width_bits == 0u || site.width_bits > 128u)
      return reject(Reason::InvalidAccessWidth);
    if (site.size != 3u * sizeof(uint32_t) || !site.raw_rsrc || !site.raw_soffset ||
        !site.raw_vaddr || !site.raw_ioffset || !site.raw_offen || !site.raw_idxen ||
        !*site.raw_offen || *site.raw_idxen)
      return reject(Reason::UnsupportedEncoding);
    if (!site.address_vgpr || !site.scalar_address_sgpr || !site.data_vgpr ||
        *site.raw_vaddr != *site.address_vgpr || *site.raw_rsrc != *site.scalar_address_sgpr)
      return reject(Reason::MissingOperands);
    if ((*site.scalar_address_sgpr & 3u) != 0u || *site.scalar_address_sgpr > 124u ||
        (*site.raw_soffset != kNullScalarOffset && *site.raw_soffset > 127u) ||
        *site.address_vgpr > 255u)
      return reject(Reason::UnsupportedInputWidth);
    if (*site.raw_ioffset < kSigned24Min || *site.raw_ioffset > kSigned24Max)
      return reject(Reason::UnsupportedOffset);
    return finish({
        .kind = ConSanAtomicLoweringFormKind::BufferResourceVectorOffset,
        .instruction_size = site.size,
        .value_width_bits = site.width_bits,
        .value_register_count = value_register_count,
        .data_register_count = data_register_count,
        .destination_register_count = destination_register_count,
        .address_vgpr = *site.address_vgpr,
        .address_vgpr_count = 1u,
        .data_vgpr = *site.data_vgpr,
        .destination_vgpr = site.destination_vgpr,
        .scalar_base_sgpr = site.scalar_address_sgpr,
        .scalar_offset_sgpr = *site.raw_soffset == kNullScalarOffset
                                  ? std::nullopt
                                  : std::optional<uint16_t>(*site.raw_soffset),
        .signed_byte_offset = *site.raw_ioffset,
        .is_rmw = is_rmw,
        .compare_exchange = compare_exchange,
        .returns_old_value = site.returns_old_value.value_or(false),
    });
  }

  const bool flat = is_flat_family(site, is_rmw);
  const bool global = is_global_family(site, is_rmw);
  if (!flat && !global)
    return reject(Reason::UnsupportedAddressSource);
  if (site.width_bits != 32u && site.width_bits != 64u)
    return reject(Reason::InvalidAccessWidth);

  const uint32_t expected_size = memory.instruction_word_count * sizeof(uint32_t);
  if (site.size != expected_size || !site.raw_saddr || !site.raw_vaddr || !site.raw_ioffset)
    return reject(Reason::UnsupportedEncoding);
  if ((memory.scale_offset == ConSanScaleOffsetCapability::Supported && !site.raw_scale_offset) ||
      (memory.scale_offset != ConSanScaleOffsetCapability::Supported &&
       site.raw_scale_offset.value_or(false)))
    return reject(Reason::UnsupportedEncoding);
  if (!site.address_vgpr || !site.data_vgpr || *site.raw_vaddr != *site.address_vgpr)
    return reject(Reason::MissingOperands);

  const uint32_t vector_only_saddr =
      flat ? memory.flat_vector_only_saddr : memory.global_vector_only_saddr;
  const int32_t offset_min = -(1 << (memory.immediate_offset_bits - 1u));
  const int32_t offset_max = (1 << (memory.immediate_offset_bits - 1u)) - 1;

  ConSanAtomicLoweringFormKind kind;
  uint16_t address_register_count;
  std::optional<uint16_t> scalar_base;
  if (*site.raw_saddr == vector_only_saddr) {
    if (*site.address_vgpr >= 255u)
      return reject(Reason::UnsupportedInputWidth);
    if (flat && *site.raw_ioffset != 0)
      return reject(Reason::UnsupportedOffset);
    if (global && (*site.raw_ioffset < offset_min || *site.raw_ioffset > offset_max))
      return reject(Reason::UnsupportedOffset);
    kind = flat ? ConSanAtomicLoweringFormKind::FlatVectorAddress
                : ConSanAtomicLoweringFormKind::GlobalVectorAddress;
    address_register_count = 2u;
  } else {
    if (!memory.supports_flat_scalar_base && flat)
      return reject(Reason::UnsupportedEncoding);
    if (!site.scalar_address_sgpr || *site.raw_saddr != *site.scalar_address_sgpr)
      return reject(Reason::UnsupportedInputWidth);
    if (*site.scalar_address_sgpr > 104u || (*site.scalar_address_sgpr & 1u) != 0u)
      return reject(Reason::UnsupportedEncoding);
    if (*site.raw_ioffset < offset_min || *site.raw_ioffset > offset_max)
      return reject(Reason::UnsupportedOffset);
    if (*site.address_vgpr > 255u)
      return reject(Reason::UnsupportedInputWidth);
    kind = flat ? ConSanAtomicLoweringFormKind::FlatScalarVectorAddress
                : ConSanAtomicLoweringFormKind::GlobalScalarVectorAddress;
    address_register_count = 1u;
    scalar_base = site.scalar_address_sgpr;
  }

  ConSanAtomicLoweringForm form{
      .kind = kind,
      .instruction_size = site.size,
      .value_width_bits = site.width_bits,
      .value_register_count = value_register_count,
      .data_register_count = data_register_count,
      .destination_register_count = destination_register_count,
      .address_vgpr = *site.address_vgpr,
      .address_vgpr_count = address_register_count,
      .data_vgpr = *site.data_vgpr,
      .destination_vgpr = site.destination_vgpr,
      .scalar_base_sgpr = scalar_base,
      .scalar_offset_sgpr = std::nullopt,
      .signed_byte_offset = *site.raw_ioffset,
      .scale_vector_offset = site.raw_scale_offset.value_or(false),
      .sign_extend_vector_offset =
          memory.vector_offset_extension == ConSanVectorOffsetExtension::Sign,
      .is_rmw = is_rmw,
      .compare_exchange = compare_exchange,
      .returns_old_value = site.returns_old_value.value_or(false),
  };
  const Reason address_reason = kind == ConSanAtomicLoweringFormKind::FlatScalarVectorAddress &&
                                        site.returns_old_value.value_or(false)
                                    ? Reason::ResultAddressAlias
                                    : Reason::None;
  return finish(std::move(form), address_reason);
}

} // namespace rocjitsu
