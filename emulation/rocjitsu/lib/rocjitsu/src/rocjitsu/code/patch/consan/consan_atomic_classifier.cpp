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
      .exact_ordering_reason = reason,
  };
}

[[nodiscard]] ConSanAtomicLoweringClassification normalized(ConSanAtomicLoweringForm form,
                                                            Reason exact_reason) {
  return {
      .form = std::move(form),
      .normalization_reason = Reason::None,
      .exact_ordering_reason = exact_reason,
  };
}

} // namespace

ConSanAtomicLoweringClassification
classify_consan_atomic_lowering(const ConSanAtomicSite &site, rj_code_arch_t arch, bool is_rmw) {
  if (!consan_is_capability_arch(arch))
    return reject(Reason::TargetUnavailable);

  const bool is_flat = site.mnemonic.starts_with("flat_atomic") ||
                       (!is_rmw && (site.mnemonic.starts_with("flat_load") ||
                                    site.mnemonic.starts_with("flat_store")));
  const bool is_vglobal = site.mnemonic.starts_with("global_atomic") ||
                          (!is_rmw && (site.mnemonic.starts_with("global_load") ||
                                       site.mnemonic.starts_with("global_store")));
  if (!is_flat && !is_vglobal)
    return reject(Reason::UnsupportedAddressSource);
  if (site.width_bits != 32u)
    return reject(Reason::InvalidAccessWidth);

  const bool is_cdna = consan_uses_gfx9_cdna_encoding(arch);
  const bool is_rdna3 = consan_uses_gfx11_encoding(arch);
  const bool is_gfx12 = consan_uses_gfx12_encoding(arch);
  const bool scalar_pair = site.raw_saddr && site.saddr_sgpr &&
                           *site.raw_saddr == *site.saddr_sgpr && *site.saddr_sgpr <= 104u &&
                           (*site.saddr_sgpr & 1u) == 0u;
  const bool gfx12_flat_saddr = site.raw_saddr && (*site.raw_saddr == 0x7cu || scalar_pair);
  const bool gfx12_flat_encoding =
      is_flat && is_gfx12 && site.size == 3u * sizeof(uint32_t) && gfx12_flat_saddr;
  const bool cdna_flat_encoding = is_flat && is_cdna && site.size == 2u * sizeof(uint32_t) &&
                                  site.raw_saddr && *site.raw_saddr == 0u;
  const bool rdna3_flat_encoding = is_flat && is_rdna3 && site.size == 2u * sizeof(uint32_t) &&
                                   site.raw_saddr && *site.raw_saddr == 0x7cu;
  const bool vglobal_saddr =
      site.raw_saddr && (*site.raw_saddr == (is_cdna ? 0x7fu : 0x7cu) || scalar_pair);
  const bool pre_gfx12_vglobal_encoding =
      is_vglobal && (is_cdna || is_rdna3) && site.size == 2u * sizeof(uint32_t) && vglobal_saddr;
  const bool gfx12_vglobal_encoding = is_vglobal && is_gfx12 && site.size == 3u * sizeof(uint32_t);
  if (!gfx12_flat_encoding && !cdna_flat_encoding && !rdna3_flat_encoding &&
      !pre_gfx12_vglobal_encoding && !gfx12_vglobal_encoding) {
    return reject(Reason::UnsupportedEncoding);
  }
  if (!site.raw_ioffset || (is_flat && *site.raw_ioffset != 0))
    return reject(is_flat && site.raw_ioffset ? Reason::NonzeroImmediateOffset
                                              : Reason::UnsupportedEncoding);
  if (!site.addr_vgpr || *site.addr_vgpr >= 255u || !site.data_vgpr)
    return reject(Reason::MissingOperands);

  const bool compare_exchange = is_rmw && consan_atomic_is_compare_exchange(site);
  ConSanAtomicLoweringForm form{
      .kind = is_flat ? (scalar_pair ? ConSanAtomicLoweringFormKind::FlatScalarVectorAddress
                                     : ConSanAtomicLoweringFormKind::FlatVectorAddress)
                      : (scalar_pair ? ConSanAtomicLoweringFormKind::GlobalScalarVectorAddress
                                     : ConSanAtomicLoweringFormKind::GlobalVectorAddress),
      .instruction_size = site.size,
      .value_width_bits = site.width_bits,
      .value_register_count = 1u,
      .address_vgpr = *site.addr_vgpr,
      .address_vgpr_count = static_cast<uint16_t>(scalar_pair ? 1u : 2u),
      .data_vgpr = *site.data_vgpr,
      .destination_vgpr = site.dst_vgpr,
      .scalar_base_sgpr = scalar_pair ? site.saddr_sgpr : std::nullopt,
      .signed_byte_offset = *site.raw_ioffset,
      .scope = site.raw_scope.value_or(0u),
      .scale_vector_offset = site.raw_scale_offset.value_or(false),
      .is_rmw = is_rmw,
      .compare_exchange = compare_exchange,
      .returns_old_value = site.returns_old_value.value_or(false),
  };

  if (compare_exchange && *site.data_vgpr >= 255u)
    return normalized(std::move(form), Reason::MissingOperands);
  if (compare_exchange && site.returns_old_value && !*site.returns_old_value)
    return normalized(std::move(form), Reason::CompareExchangeOutcomeUnavailable);
  if (compare_exchange && !site.dst_vgpr)
    return normalized(std::move(form), Reason::MissingOperands);
  if (!site.raw_scope || !site.raw_th || !site.returns_old_value)
    return normalized(std::move(form), Reason::MissingOrderingMetadata);
  if (*site.raw_scope < 1u || *site.raw_scope > 3u)
    return normalized(std::move(form), Reason::UnsupportedScope);
  return normalized(std::move(form), Reason::None);
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
