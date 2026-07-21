// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translation_profile.cpp
/// @brief Full RocJITsu DBT registration for every supported translation pair.

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/semantic/rules.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

namespace rocjitsu {

namespace {

[[nodiscard]] EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest,
                                                             rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3;
  return nullptr;
}

[[nodiscard]] const InstructionLegalization *lookup_cdna4_to_rdna4(const Instruction &inst) {
  return lookup(kLegalization_cdna4_to_rdna4, inst.encoding_id(), inst.opcode());
}

[[nodiscard]] const InstructionLegalization *lookup_cdna4_to_cdna3(const Instruction &inst) {
  return lookup(kLegalization_cdna4_to_cdna3, inst.encoding_id(), inst.opcode());
}

[[nodiscard]] const InstructionLegalization *lookup_cdna4_to_rdna3(const Instruction &inst) {
  return lookup(kLegalization_cdna4_to_rdna3, inst.encoding_id(), inst.opcode());
}

[[nodiscard]] LegalizationLookupFn select_legalization(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return lookup_cdna4_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return lookup_cdna4_to_cdna3;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return lookup_cdna4_to_rdna3;
  return nullptr;
}

[[nodiscard]] std::span<const TranslationRule> select_semantic_rules(rj_code_arch_t guest,
                                                                     rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return semantic_expand_rules_cdna4_to_rdna4();
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return semantic_expand_rules_cdna4_to_cdna3();
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return semantic_expand_rules_cdna4_to_rdna3();
  return {};
}

[[nodiscard]] std::unique_ptr<Decoder> create_registered_decoder(rj_code_arch_t arch) {
  return Decoder::create(arch);
}

[[nodiscard]] TranslationProfile select_translation_profile(rj_code_arch_t guest,
                                                            rj_code_arch_t host,
                                                            ProcessorRevision input_revision,
                                                            ProcessorRevision output_revision) {
  if (guest == ROCJITSU_CODE_ARCH_GFX1250 && host == ROCJITSU_CODE_ARCH_GFX1250 &&
      input_revision == ProcessorRevision::Gfx1250B0 &&
      output_revision == ProcessorRevision::Gfx1250A0)
    return gfx1250_b0_to_a0_translation_profile();

  return {.encoding_translate = select_encoding_translator(guest, host),
          .legalization_lookup = select_legalization(guest, host),
          .semantic_rules = select_semantic_rules(guest, host),
          .create_decoder = create_registered_decoder};
}

} // namespace

BinaryTranslator::BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                   uint32_t target_mach, BinaryTranslatorOptions options)
    : BinaryTranslator(guest_arch, host_arch, target_mach, options,
                       select_translation_profile(guest_arch, host_arch, options.input_revision,
                                                  options.output_revision)) {}

} // namespace rocjitsu
