// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_atomic_classifier.h"

#include "rocjitsu/code/patch/consan/consan.h"

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
  if (!site.raw_scope)
    return Reason::MissingOrderingMetadata;
  if (*site.raw_scope < 1u || *site.raw_scope > 3u)
    return Reason::UnsupportedScope;
  if (form.kind == ConSanAtomicLoweringFormKind::LdsVectorOffset && *site.raw_scope != 1u)
    return Reason::UnsupportedScope;
  if (form.compare_exchange && (!site.returns_old_value.value_or(false) || !site.dst_vgpr))
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
  if (form.compare_exchange && !site.dst_vgpr)
    return Reason::MissingOperands;
  if (!site.raw_scope || !site.raw_th || !site.returns_old_value)
    return Reason::MissingOrderingMetadata;
  if (*site.raw_scope < 1u || *site.raw_scope > 3u)
    return Reason::UnsupportedScope;
  return Reason::None;
}

} // namespace

ConSanAtomicLoweringClassification
classify_consan_atomic_lowering(const ConSanAtomicSite &site, rj_code_arch_t arch, bool is_rmw) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (!target)
    return reject(Reason::TargetUnavailable);
  const bool gfx9_cdna_encoding = consan_uses_gfx9_cdna_encoding(arch);
  const bool gfx12_encoding = target->encoding_family == ConSanEncodingFamily::Gfx12;
  const bool gfx12_cdna_execution =
      gfx12_encoding && target->architecture_family == ConSanArchitectureFamily::Cdna;

  const bool compare_exchange = is_rmw && consan_atomic_is_compare_exchange(site);
  const uint16_t value_register_count = static_cast<uint16_t>((site.width_bits + 31u) / 32u);
  const uint16_t data_register_count =
      static_cast<uint16_t>(value_register_count * (compare_exchange ? 2u : 1u));
  const bool ordinary_load =
      !is_rmw && site.dst_vgpr && site.data_vgpr && site.dst_vgpr == site.data_vgpr;
  const uint16_t destination_register_count =
      site.returns_old_value.value_or(false) || ordinary_load ? value_register_count : 0u;
  if ((site.data_vgpr && static_cast<uint32_t>(*site.data_vgpr) + data_register_count > 256u) ||
      (site.dst_vgpr && destination_register_count != 0u &&
       static_cast<uint32_t>(*site.dst_vgpr) + destination_register_count > 256u)) {
    return reject(Reason::UnsupportedInputWidth);
  }
  const auto finish = [&](ConSanAtomicLoweringForm form, Reason address_reason = Reason::None) {
    return normalized(form, address_reason,
                      address_reason == Reason::None ? exact_reason(site, form) : address_reason,
                      address_reason == Reason::None ? causal_reason(site, form) : address_reason);
  };

  if (site.mnemonic.starts_with("ds_")) {
    if (!gfx12_cdna_execution)
      return reject(Reason::UnsupportedAddressSource);
    if (site.width_bits != 32u)
      return reject(Reason::InvalidAccessWidth);
    if (site.size != 2u * sizeof(uint32_t) || !site.raw_addr || !site.raw_data0 ||
        !site.raw_ioffset)
      return reject(Reason::UnsupportedEncoding);
    if (!site.addr_vgpr || !site.data_vgpr || *site.raw_addr != *site.addr_vgpr ||
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
        .address_vgpr = *site.addr_vgpr,
        .address_vgpr_count = 1u,
        .data_vgpr = *site.data_vgpr,
        .destination_vgpr = site.dst_vgpr,
        .scalar_base_sgpr = std::nullopt,
        .scalar_offset_sgpr = std::nullopt,
        .signed_byte_offset = *site.raw_ioffset,
        .scope = site.raw_scope.value_or(0u),
        .is_rmw = is_rmw,
        .compare_exchange = compare_exchange,
        .returns_old_value = site.returns_old_value.value_or(false),
    });
  }

  if (site.mnemonic.starts_with("buffer_")) {
    constexpr uint32_t kNullScalarOffset = 0x7cu;
    constexpr int32_t kSigned24Min = -(1 << 23);
    constexpr int32_t kSigned24Max = (1 << 23) - 1;
    if (!gfx12_cdna_execution)
      return reject(Reason::UnsupportedAddressSource);
    if (site.width_bits == 0u || site.width_bits > 128u)
      return reject(Reason::InvalidAccessWidth);
    if (site.size != 3u * sizeof(uint32_t) || !site.raw_rsrc || !site.raw_soffset ||
        !site.raw_vaddr || !site.raw_ioffset || !site.raw_offen || !site.raw_idxen ||
        !*site.raw_offen || *site.raw_idxen)
      return reject(Reason::UnsupportedEncoding);
    if (!site.addr_vgpr || !site.saddr_sgpr || !site.data_vgpr ||
        *site.raw_vaddr != *site.addr_vgpr || *site.raw_rsrc != *site.saddr_sgpr)
      return reject(Reason::MissingOperands);
    if ((*site.saddr_sgpr & 3u) != 0u || *site.saddr_sgpr > 124u ||
        (*site.raw_soffset != kNullScalarOffset && *site.raw_soffset > 127u) ||
        *site.addr_vgpr > 255u)
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
        .address_vgpr = *site.addr_vgpr,
        .address_vgpr_count = 1u,
        .data_vgpr = *site.data_vgpr,
        .destination_vgpr = site.dst_vgpr,
        .scalar_base_sgpr = site.saddr_sgpr,
        .scalar_offset_sgpr = *site.raw_soffset == kNullScalarOffset
                                  ? std::nullopt
                                  : std::optional<uint16_t>(*site.raw_soffset),
        .signed_byte_offset = *site.raw_ioffset,
        .scope = site.raw_scope.value_or(0u),
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

  const bool two_word_encoding = !gfx12_encoding;
  const uint32_t expected_size = two_word_encoding ? 2u * sizeof(uint32_t) : 3u * sizeof(uint32_t);
  if (site.size != expected_size || !site.raw_saddr || !site.raw_vaddr || !site.raw_ioffset)
    return reject(Reason::UnsupportedEncoding);
  if ((gfx12_cdna_execution && !site.raw_scale_offset) ||
      (!gfx12_cdna_execution && site.raw_scale_offset.value_or(false)))
    return reject(Reason::UnsupportedEncoding);
  if (!site.addr_vgpr || !site.data_vgpr || *site.raw_vaddr != *site.addr_vgpr)
    return reject(Reason::MissingOperands);

  constexpr uint32_t kCdnaGlobalNoSaddr = 0x7fu;
  constexpr uint32_t kGfx11NoSaddr = 0x7cu;
  const uint32_t vector_only_saddr =
      gfx9_cdna_encoding ? (global ? kCdnaGlobalNoSaddr : 0u) : kGfx11NoSaddr;
  constexpr int32_t kSigned13Min = -(1 << 12);
  constexpr int32_t kSigned13Max = (1 << 12) - 1;
  constexpr int32_t kSigned24Min = -(1 << 23);
  constexpr int32_t kSigned24Max = (1 << 23) - 1;
  const int32_t offset_min = two_word_encoding ? kSigned13Min : kSigned24Min;
  const int32_t offset_max = two_word_encoding ? kSigned13Max : kSigned24Max;

  ConSanAtomicLoweringFormKind kind;
  uint16_t address_register_count;
  std::optional<uint16_t> scalar_base;
  if (*site.raw_saddr == vector_only_saddr) {
    if (*site.addr_vgpr >= 255u)
      return reject(Reason::UnsupportedInputWidth);
    if (flat && *site.raw_ioffset != 0)
      return reject(Reason::UnsupportedOffset);
    if (global && (*site.raw_ioffset < offset_min || *site.raw_ioffset > offset_max))
      return reject(Reason::UnsupportedOffset);
    kind = flat ? ConSanAtomicLoweringFormKind::FlatVectorAddress
                : ConSanAtomicLoweringFormKind::GlobalVectorAddress;
    address_register_count = 2u;
  } else {
    if (!gfx12_encoding && flat)
      return reject(Reason::UnsupportedEncoding);
    if (!site.saddr_sgpr || *site.raw_saddr != *site.saddr_sgpr)
      return reject(Reason::UnsupportedInputWidth);
    if (*site.saddr_sgpr > 104u || (*site.saddr_sgpr & 1u) != 0u)
      return reject(Reason::UnsupportedEncoding);
    if (*site.raw_ioffset < offset_min || *site.raw_ioffset > offset_max)
      return reject(Reason::UnsupportedOffset);
    if (*site.addr_vgpr > 255u)
      return reject(Reason::UnsupportedInputWidth);
    kind = flat ? ConSanAtomicLoweringFormKind::FlatScalarVectorAddress
                : ConSanAtomicLoweringFormKind::GlobalScalarVectorAddress;
    address_register_count = 1u;
    scalar_base = site.saddr_sgpr;
  }

  ConSanAtomicLoweringForm form{
      .kind = kind,
      .instruction_size = site.size,
      .value_width_bits = site.width_bits,
      .value_register_count = value_register_count,
      .data_register_count = data_register_count,
      .destination_register_count = destination_register_count,
      .address_vgpr = *site.addr_vgpr,
      .address_vgpr_count = address_register_count,
      .data_vgpr = *site.data_vgpr,
      .destination_vgpr = site.dst_vgpr,
      .scalar_base_sgpr = scalar_base,
      .scalar_offset_sgpr = std::nullopt,
      .signed_byte_offset = *site.raw_ioffset,
      .scope = site.raw_scope.value_or(0u),
      .scale_vector_offset = site.raw_scale_offset.value_or(false),
      .sign_extend_vector_offset = gfx9_cdna_encoding,
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

std::string_view consan_atomic_classifier_reason_name(ConSanAtomicClassifierReason reason) {
  switch (reason) {
  case Reason::None:
    return "supported";
  case Reason::UnsupportedAddressSource:
    return "non-flat-address";
  case Reason::InvalidAccessWidth:
    return "unsupported-width";
  case Reason::UnsupportedEncoding:
    return "unsupported-encoding";
  case Reason::NonzeroImmediateOffset:
    return "nonzero-offset";
  case Reason::MissingOperands:
    return "missing-operands";
  case Reason::UnsupportedInputWidth:
    return "unsupported-input-width";
  case Reason::UnsupportedOffset:
    return "unsupported-offset";
  case Reason::ResultAddressAlias:
    return "result-address-alias";
  case Reason::CompareExchangeOutcomeUnavailable:
    return "compare-exchange-outcome-unavailable";
  case Reason::MissingOrderingMetadata:
    return "missing-ordering-metadata";
  case Reason::UnsupportedScope:
    return "unsupported-scope";
  case Reason::TargetUnavailable:
    return "target-unavailable";
  case Reason::Count:
    break;
  }
  return "unknown";
}

} // namespace rocjitsu
