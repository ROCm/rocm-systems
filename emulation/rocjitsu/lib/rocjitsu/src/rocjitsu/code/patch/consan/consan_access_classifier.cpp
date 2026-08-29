// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_access_classifier.h"

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"

#include <array>
#include <ranges>

namespace rocjitsu {
namespace {

using Reason = ConSanAccessClassifierReason;
using TwoRangeShape = consan_detail::DecodedNativeLdsTwoRangeShape;

[[nodiscard]] ConSanAccessLoweringClassification reject(Reason reason) {
  return {
      .form = std::nullopt,
      .normalization_reason = reason,
      .replay_guest_access = {.reason = reason},
      .compare_observed_value = {.reason = reason},
  };
}

[[nodiscard]] Reason inventory_reason(const ConSanAccessInventorySite &access) {
  if (access.exclusions.empty() && !access.ranges.empty())
    return Reason::None;
  if (access.exclusions.empty())
    return Reason::RangeEncodingUnavailable;
  switch (access.exclusions.front().reason) {
  case ConSanInventoryExclusionReason::NonAccessInstruction:
    return Reason::NonAccessInstruction;
  case ConSanInventoryExclusionReason::InvalidInstructionSize:
    return Reason::InvalidInstructionSize;
  case ConSanInventoryExclusionReason::InvalidAccessWidth:
    return Reason::InvalidAccessWidth;
  case ConSanInventoryExclusionReason::MissingAddressOperand:
    return Reason::MissingAddressOperand;
  case ConSanInventoryExclusionReason::RangeEncodingUnavailable:
    return Reason::RangeEncodingUnavailable;
  case ConSanInventoryExclusionReason::Count:
    break;
  }
  return Reason::RangeEncodingUnavailable;
}

template <typename Range>
[[nodiscard]] bool named(std::string_view mnemonic, const Range &supported) {
  return std::ranges::any_of(supported,
                             [mnemonic](std::string_view value) { return value == mnemonic; });
}

[[nodiscard]] bool is_relaxed_lds_atomic(std::string_view mnemonic) {
  constexpr std::array forms = {"ds_add_f32", "ds_add_f64",          "ds_add_u32",
                                "ds_add_u64", "ds_cmpstore_rtn_b32", "ds_cmpst_rtn_b32"};
  return named(mnemonic, forms);
}

[[nodiscard]] bool is_replayable_single_range_native_lds(std::string_view mnemonic,
                                                         rj_code_arch_t arch) {
  if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
       consan_uses_gfx12_encoding(arch)) &&
      (mnemonic == "ds_load_b96" || mnemonic == "ds_store_b96")) {
    return true;
  }
  if (consan_uses_gfx9_cdna_encoding(arch) &&
      (mnemonic == "ds_read_b96" || mnemonic == "ds_write_b96")) {
    return true;
  }
  constexpr std::array always = {
      "ds_load_i8",         "ds_load_u8",          "ds_load_i16",       "ds_load_u16",
      "ds_load_u8_d16",     "ds_load_u8_d16_hi",   "ds_load_i8_d16",    "ds_load_i8_d16_hi",
      "ds_load_u16_d16",    "ds_load_u16_d16_hi",  "ds_store_b8",       "ds_store_b16",
      "ds_store_b8_d16_hi", "ds_store_b16_d16_hi", "ds_read_u8",        "ds_read_u16",
      "ds_write_b8",        "ds_write_b16",        "ds_load_b32",       "ds_load_b64",
      "ds_load_b128",       "ds_load_tr8_b64",     "ds_load_tr16_b128", "ds_read_b32",
      "ds_read_b64",        "ds_read_b64_tr_b16",  "ds_read_b128",      "ds_store_b32",
      "ds_store_b64",       "ds_store_b128",       "ds_write_b32",      "ds_write_b64",
      "ds_write_b128",      "ds_add_f32",          "ds_add_f64",        "ds_add_u32",
      "ds_add_u64",         "ds_cmpstore_rtn_b32", "ds_cmpst_rtn_b32",
  };
  if (named(mnemonic, always))
    return true;
  constexpr std::array gfx9 = {
      "ds_read_i8",         "ds_read_u8_d16",     "ds_read_u8_d16_hi",
      "ds_read_i8_d16",     "ds_read_i8_d16_hi",  "ds_read_u16_d16",
      "ds_read_u16_d16_hi", "ds_write_b8_d16_hi", "ds_write_b16_d16_hi",
  };
  return consan_uses_gfx9_cdna_encoding(arch) && named(mnemonic, gfx9);
}

[[nodiscard]] bool is_replayable_flat_access(std::string_view mnemonic) {
  if (consan_flat_load_subword_semantics(mnemonic) || consan_flat_store_subword_semantics(mnemonic))
    return true;
  constexpr std::array forms = {
      "flat_load_b32",     "flat_load_b64",    "flat_load_b128",     "flat_store_b32",
      "flat_store_b64",    "flat_store_b128",  "flat_load_dword",    "flat_load_dwordx2",
      "flat_load_dwordx4", "flat_store_dword", "flat_store_dwordx2", "flat_store_dwordx4",
      "flat_load_ushort",  "flat_store_short", "flat_load_u16",      "flat_store_b16",
  };
  return named(mnemonic, forms);
}

[[nodiscard]] uint32_t vector_flat_no_saddr(rj_code_arch_t arch) {
  if (consan_uses_gfx9_cdna_encoding(arch))
    return 0u;
  if (arch == ROCJITSU_CODE_ARCH_CDNA5)
    return static_cast<uint32_t>(cdna5::OPR_SREG_NULL);
  return static_cast<uint32_t>(rdna4::OPR_SREG_NULL);
}

[[nodiscard]] std::optional<uint16_t>
native_data_register_count(const ConSanAccessInventorySite &access,
                           const std::optional<TwoRangeShape> &two_address) {
  if (access.decoded_width_bits == 8u || access.decoded_width_bits == 16u)
    return 1u;
  if (two_address)
    return static_cast<uint16_t>(2u * two_address->element_width_bits / 32u);
  if (access.decoded_width_bits == 32u || access.decoded_width_bits == 64u ||
      access.decoded_width_bits == 96u || access.decoded_width_bits == 128u)
    return static_cast<uint16_t>(access.decoded_width_bits / 32u);
  return std::nullopt;
}

[[nodiscard]] Reason native_compare_support(const ConSanAccessInventorySite &access,
                                            rj_code_arch_t arch,
                                            const std::optional<TwoRangeShape> &two_address,
                                            uint16_t data_register_count) {
  if (access.instruction_size != 2u * sizeof(uint32_t))
    return Reason::UnsupportedEncoding;
  if (!access.operands.address_vgpr)
    return Reason::MissingAddressOperand;

  constexpr std::array reads = {
      "ds_load_i8",
      "ds_load_u8",
      "ds_load_i16",
      "ds_load_b32",
      "ds_load_b64",
      "ds_load_b96",
      "ds_load_b128",
      "ds_load_tr8_b64",
      "ds_load_tr16_b128",
      "ds_load_2addr_b32",
      "ds_load_2addr_b64",
      "ds_load_2addr_stride64_b32",
      "ds_load_2addr_stride64_b64",
      "ds_load_u16",
      "ds_load_u8_d16",
      "ds_load_u8_d16_hi",
      "ds_load_i8_d16",
      "ds_load_i8_d16_hi",
      "ds_load_u16_d16",
      "ds_load_u16_d16_hi",
      "ds_read_u8_d16",
      "ds_read_u8_d16_hi",
      "ds_read_i8_d16",
      "ds_read_i8_d16_hi",
      "ds_read_u16_d16",
      "ds_read_u16_d16_hi",
      "ds_read_b32",
      "ds_read_b64",
      "ds_read_b96",
      "ds_read_b128",
      "ds_read_i16",
      "ds_read_u16",
      "ds_read_b64_tr_b16",
      "ds_read2_b32",
      "ds_read2st64_b32",
      "ds_read2_b64",
      "ds_read2st64_b64",
  };
  constexpr std::array writes = {
      "ds_store_b8",
      "ds_store_b8_d16_hi",
      "ds_store_b16",
      "ds_store_b16_d16_hi",
      "ds_store_b32",
      "ds_store_b64",
      "ds_store_b96",
      "ds_store_b128",
      "ds_store_2addr_b32",
      "ds_store_2addr_b64",
      "ds_store_2addr_stride64_b32",
      "ds_store_2addr_stride64_b64",
      "ds_write_b32",
      "ds_write_b16",
      "ds_write_b64",
      "ds_write_b96",
      "ds_write_b128",
      "ds_write2_b32",
      "ds_write2st64_b32",
      "ds_write2_b64",
      "ds_write2st64_b64",
  };

  if (access.kind == ConSanLdsAccessKind::Read) {
    if (!named(access.mnemonic, reads))
      return Reason::UnsupportedMnemonic;
    if (access.operands.destination_accvgpr) {
      const ConSanTargetProfile *profile = consan_target_profile(arch);
      return profile &&
                     profile->accumulator_model == ConSanAccumulatorModel::DescriptorPartitioned &&
                     static_cast<uint32_t>(*access.operands.destination_accvgpr) +
                             data_register_count <=
                         256u
                 ? Reason::None
                 : Reason::OperandRegisterRange;
    }
    if (!access.operands.destination_vgpr)
      return Reason::MissingResultOperand;
    const uint32_t limit = consan_arch_has_selectable_vgpr_bank(arch) ? 1024u : 256u;
    return static_cast<uint32_t>(*access.operands.destination_vgpr) + data_register_count <= limit
               ? Reason::None
               : Reason::OperandRegisterRange;
  }

  if (access.kind != ConSanLdsAccessKind::Write || !named(access.mnemonic, writes))
    return Reason::UnsupportedMnemonic;
  if (two_address) {
    const uint32_t per_range = two_address->element_width_bits / 32u;
    if (!access.operands.data_vgpr || !access.operands.second_data_vgpr)
      return Reason::MissingDataOperand;
    return static_cast<uint32_t>(*access.operands.data_vgpr) + per_range <= 256u &&
                   static_cast<uint32_t>(*access.operands.second_data_vgpr) + per_range <= 256u
               ? Reason::None
               : Reason::OperandRegisterRange;
  }
  if (!access.operands.data_vgpr)
    return Reason::MissingDataOperand;
  return static_cast<uint32_t>(*access.operands.data_vgpr) + data_register_count <= 256u
             ? Reason::None
             : Reason::OperandRegisterRange;
}

[[nodiscard]] std::optional<uint16_t>
flat_data_register_count(const ConSanAccessInventorySite &access) {
  if (consan_flat_load_subword_semantics(access.mnemonic) ||
      consan_flat_store_subword_semantics(access.mnemonic) || access.decoded_width_bits == 16u)
    return 1u;
  if (access.decoded_width_bits == 32u || access.decoded_width_bits == 64u ||
      access.decoded_width_bits == 128u)
    return static_cast<uint16_t>(access.decoded_width_bits / 32u);
  return std::nullopt;
}

[[nodiscard]] Reason flat_compare_support(const ConSanAccessInventorySite &access,
                                          uint16_t data_register_count) {
  constexpr std::array reads = {"flat_load_b32",    "flat_load_b64",     "flat_load_b128",
                                "flat_load_dword",  "flat_load_dwordx2", "flat_load_dwordx4",
                                "flat_load_ushort", "flat_load_u16"};
  constexpr std::array writes = {"flat_store_b32",   "flat_store_b64",     "flat_store_b128",
                                 "flat_store_dword", "flat_store_dwordx2", "flat_store_dwordx4",
                                 "flat_store_short", "flat_store_b16"};
  if (access.kind == ConSanLdsAccessKind::Read) {
    if (!consan_flat_load_subword_semantics(access.mnemonic) && !named(access.mnemonic, reads))
      return Reason::UnsupportedMnemonic;
    if (!access.operands.destination_vgpr)
      return Reason::MissingResultOperand;
    return static_cast<uint32_t>(*access.operands.destination_vgpr) + data_register_count <= 256u
               ? Reason::None
               : Reason::OperandRegisterRange;
  }
  if (access.kind != ConSanLdsAccessKind::Write ||
      (!consan_flat_store_subword_semantics(access.mnemonic) && !named(access.mnemonic, writes)))
    return Reason::UnsupportedMnemonic;
  if (!access.operands.data_vgpr)
    return Reason::MissingDataOperand;
  return static_cast<uint32_t>(*access.operands.data_vgpr) + data_register_count <= 256u
             ? Reason::None
             : Reason::OperandRegisterRange;
}

} // namespace

ConSanAccessLoweringClassification
classify_consan_access_lowering(const ConSanAccessInventorySite &access, rj_code_arch_t arch) {
  if (const Reason reason = inventory_reason(access); reason != Reason::None)
    return reject(reason);
  if (access.file_offset > access.physical_id.code_object.byte_size ||
      access.instruction_size > access.physical_id.code_object.byte_size - access.file_offset)
    return reject(Reason::InstructionOutOfBounds);
  if (consan_target_profile(arch) == nullptr)
    return reject(Reason::TargetUnavailable);

  ConSanAccessLoweringForm form{
      .access_kind = access.kind,
      .instruction_size = access.instruction_size,
      .element_width_bits = access.decoded_width_bits,
      .range_count = static_cast<uint32_t>(access.ranges.size()),
      .address_vgpr = access.operands.address_vgpr,
      .destination_vgpr = access.operands.destination_vgpr,
      .destination_accvgpr = access.operands.destination_accvgpr,
      .data_vgpr = access.operands.data_vgpr,
      .second_data_vgpr = access.operands.second_data_vgpr,
      .scalar_address_sgpr = std::nullopt,
      .immediate_byte_offset = access.operands.raw_ioffset,
      .scale_immediate = access.operands.raw_scale_offset.value_or(false),
  };
  Reason replay = Reason::UnsupportedMnemonic;
  Reason compare = Reason::UnsupportedMnemonic;

  if (access.origin == ConSanAccessOrigin::DirectToLds) {
    form.kind = access.operands.address_vgpr
                    ? ConSanAccessLoweringFormKind::DirectToLdsExplicitAddress
                    : ConSanAccessLoweringFormKind::DirectToLdsLaneAddressed;
    form.address_vgpr_count = access.operands.address_vgpr ? 1u : 0u;
    form.data_register_count = static_cast<uint16_t>((access.decoded_width_bits + 31u) / 32u);
    replay = Reason::None;
  } else if (access.origin == ConSanAccessOrigin::NativeLds) {
    const auto two_address = consan_detail::decode_native_lds_two_range_shape(access.mnemonic);
    form.kind = two_address ? ConSanAccessLoweringFormKind::NativeTwoRange
                            : ConSanAccessLoweringFormKind::NativeSingleRange;
    form.element_width_bits =
        two_address ? two_address->element_width_bits : access.decoded_width_bits;
    form.encoded_offset_scale_bytes = two_address ? two_address->offset_scale_bytes : 1u;
    const auto register_count = native_data_register_count(access, two_address);
    if (!register_count)
      return reject(Reason::UnsupportedMnemonic);
    form.address_vgpr_count = 1u;
    form.data_register_count = *register_count;
    replay = is_replayable_single_range_native_lds(access.mnemonic, arch) || two_address ||
                     is_relaxed_lds_atomic(access.mnemonic)
                 ? Reason::None
                 : Reason::UnsupportedMnemonic;
    compare = native_compare_support(access, arch, two_address, *register_count);
  } else if (access.origin == ConSanAccessOrigin::Flat) {
    if ((access.instruction_size != 2u * sizeof(uint32_t) &&
         access.instruction_size != 3u * sizeof(uint32_t)) ||
        !access.operands.address_vgpr)
      return reject(access.operands.address_vgpr ? Reason::UnsupportedEncoding
                                                 : Reason::MissingAddressOperand);
    const auto register_count = flat_data_register_count(access);
    if (!register_count)
      return reject(Reason::UnsupportedMnemonic);
    form.data_register_count = *register_count;
    const bool scalar_vector_address = consan_uses_gfx12_encoding(arch) &&
                                       access.operands.raw_saddr &&
                                       *access.operands.raw_saddr != vector_flat_no_saddr(arch);
    form.kind = scalar_vector_address ? ConSanAccessLoweringFormKind::FlatScalarVectorAddress
                                      : ConSanAccessLoweringFormKind::FlatVectorAddress;
    form.address_vgpr_count = scalar_vector_address ? 1u : 2u;
    if (scalar_vector_address)
      form.scalar_address_sgpr = static_cast<uint16_t>(*access.operands.raw_saddr);

    const bool gfx12_encoding = access.instruction_size == 3u * sizeof(uint32_t);
    const bool cdna4_encoding =
        access.instruction_size == 2u * sizeof(uint32_t) && access.operands.raw_segment == 0u;
    if (!gfx12_encoding && !cdna4_encoding) {
      replay = Reason::UnsupportedEncoding;
    } else if (!access.operands.raw_ioffset ||
               (consan_uses_gfx12_encoding(arch) &&
                (!access.operands.raw_saddr || !access.operands.raw_scale_offset))) {
      replay = Reason::UnsupportedEncoding;
    } else if (*access.operands.raw_ioffset != 0 && !consan_uses_gfx12_encoding(arch)) {
      replay = Reason::NonzeroImmediateOffset;
    } else if (scalar_vector_address &&
               (*access.operands.raw_saddr > 104u || (*access.operands.raw_saddr & 1u) != 0u)) {
      replay = Reason::OperandRegisterRange;
    } else if (*access.operands.address_vgpr >= 255u && !scalar_vector_address) {
      replay = Reason::ReservedAddressRegister;
    } else {
      replay =
          is_replayable_flat_access(access.mnemonic) ? Reason::None : Reason::UnsupportedMnemonic;
    }
    compare = flat_compare_support(access, *register_count);
  } else {
    return reject(Reason::UnsupportedEncoding);
  }

  form.element_register_count = static_cast<uint16_t>((form.element_width_bits + 31u) / 32u);
  form.destination_register_count = form.destination_vgpr ? form.data_register_count : 0u;

  return {
      .form = form,
      .normalization_reason = Reason::None,
      .replay_guest_access = {.reason = replay},
      .compare_observed_value = {.reason = compare},
  };
}

} // namespace rocjitsu
