// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu::consan {
namespace {

struct TargetCase {
  rj_code_arch_t arch;
  rj_code_target_id_t target;
  std::string_view native_store;
  std::string_view native_b96_store;
};

constexpr std::array kTargets = {
    TargetCase{ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_TARGET_GFX942, "ds_write_b32",
               "ds_write_b96"},
    TargetCase{ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950, "ds_write_b32",
               "ds_write_b96"},
    TargetCase{ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_TARGET_GFX1100, "ds_store_b32",
               "ds_store_b96"},
    TargetCase{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250, "ds_store_b32",
               "ds_store_b96"},
    TargetCase{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201, "ds_store_b32",
               "ds_store_b96"},
};

ProgramSite native_store_site(std::string_view mnemonic) {
  ProgramSite site;
  site.origin = AccessOrigin::NativeLds;
  site.kind = LdsAccessKind::Write;
  site.physical_id.original_text_offset = 8;
  site.decoded_site().text_offset = 8;
  site.decoded_site().file_offset = 8;
  site.decoded_site().size = 8;
  site.decoded_width_bits = 32;
  site.operands.address_vgpr = 3;
  site.operands.data_vgpr = 7;
  site.decoded_site().mnemonic = std::string(mnemonic);
  return site;
}

ProgramSite native_access_site(std::string_view mnemonic, LdsAccessKind kind, uint32_t width_bits) {
  ProgramSite site = native_store_site(mnemonic);
  site.kind = kind;
  site.decoded_width_bits = width_bits;
  if (kind == LdsAccessKind::Read) {
    site.operands.data_vgpr.reset();
    site.operands.destination_vgpr = 7;
  }
  return site;
}

ProgramSite complete_site(ProgramSite site, rj_code_arch_t arch, rj_code_target_id_t target);

TEST(ConSanAccessClassifier, NativeB96SpellingIsOwnedByTheTargetClassifier) {
  for (const TargetCase &target : kTargets) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    ProgramSite expected = native_store_site(target.native_b96_store);
    expected.decoded_width_bits = 96;
    const ProgramSite admitted = complete_site(std::move(expected), target.arch, target.target);
    EXPECT_TRUE(admitted.lowering.replay_guest_access.available());

    const std::string_view other_spelling =
        target.native_b96_store == "ds_write_b96" ? "ds_store_b96" : "ds_write_b96";
    ProgramSite mismatched = native_store_site(other_spelling);
    mismatched.decoded_width_bits = 96;
    const ProgramSite rejected = complete_site(std::move(mismatched), target.arch, target.target);
    EXPECT_EQ(rejected.lowering.replay_guest_access.reason,
              AccessClassifierReason::UnsupportedMnemonic);
  }
}

ProgramSite flat_store_site(uint32_t instruction_size) {
  ProgramSite site;
  site.origin = AccessOrigin::Flat;
  site.kind = LdsAccessKind::Write;
  site.physical_id.original_text_offset = 8;
  site.decoded_site().text_offset = 8;
  site.decoded_site().file_offset = 8;
  site.decoded_site().size = instruction_size;
  site.decoded_width_bits = 32;
  site.operands.address_vgpr = 3;
  site.operands.data_vgpr = 7;
  site.operands.raw_segment = 0;
  site.operands.raw_ioffset = 0;
  site.flat_address_space_hint = FlatAddressSpaceHint::Group;
  site.decoded_site().mnemonic = "flat_store_b32";
  return site;
}

ProgramSite complete_site(ProgramSite site, rj_code_arch_t arch, rj_code_target_id_t target) {
  const std::array<uint8_t, 64> bytes = {};
  ProgramInventoryBuilder builder(bytes);
  builder.set_code_object_facts(true, 0, arch, target);
  ProgramContainer kernel{ProgramContainerKind::Kernel};
  kernel.name = "classifier_kernel";
  kernel.descriptor_file_offset = 48;
  kernel.entry_text_offset = 0;
  site.container = kernel.id;
  builder.add_kernel(kernel);
  builder.add_access_site(std::move(site));
  builder.publish_decoded_accesses(bytes);
  return builder.view().access_sites().front();
}

TEST(ConSanAccessClassifier, NativeReplayAndValueComparisonNormalizeOnAllFiveTargets) {
  for (const TargetCase &target : kTargets) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const ProgramSite site =
        complete_site(native_store_site(target.native_store), target.arch, target.target);
    ASSERT_TRUE(site.lowering.normalized());
    ASSERT_TRUE(site.lowering.form);
    EXPECT_EQ(site.lowering.form->kind, AccessLoweringFormKind::NativeSingleRange);
    EXPECT_EQ(site.lowering.form->range_count, 1u);
    EXPECT_EQ(site.lowering.form->element_width_bits, 32u);
    EXPECT_EQ(site.lowering.form->element_register_count, 1u);
    EXPECT_EQ(site.lowering.form->data_register_count, 1u);
    EXPECT_EQ(site.lowering.form->destination_register_count, 0u);
    EXPECT_EQ(site.lowering.form->address_vgpr_count, 1u);
    EXPECT_TRUE(site.lowering.replay_guest_access.available());
    EXPECT_TRUE(site.lowering.compare_observed_value.available());
    EXPECT_EQ(site.lowering, classify_access_lowering(site, target.arch));
  }
}

TEST(ConSanAccessClassifier, LaneAddressedDirectToLdsWritesSupportValueComparison) {
  ProgramSite input;
  input.origin = AccessOrigin::DirectToLds;
  input.kind = LdsAccessKind::Write;
  input.physical_id.original_text_offset = 8u;
  input.decoded_site().text_offset = 8u;
  input.decoded_site().file_offset = 8u;
  input.decoded_site().size = 8u;
  input.decoded_width_bits = 128u;
  input.decoded_site().mnemonic = "buffer_load_dwordx4";
  input.operands.direct_memory_address_vgpr = 3u;

  const ProgramSite site =
      complete_site(std::move(input), ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950);

  ASSERT_TRUE(site.lowering.form);
  EXPECT_EQ(site.lowering.form->kind, AccessLoweringFormKind::DirectToLdsLaneAddressed);
  EXPECT_TRUE(site.lowering.replay_guest_access.available());
  EXPECT_TRUE(site.lowering.compare_observed_value.available());
}

TEST(ConSanAccessClassifier, FlatEncodingDifferencesProduceOneNormalizedVocabulary) {
  for (const TargetCase &target : kTargets) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const VectorMemoryCapability &memory = target_profile(target.arch)->vector_memory;
    ProgramSite input = flat_store_site(memory.instruction_word_count * sizeof(uint32_t));
    if (memory.instruction_word_count == 3u) {
      input.operands.raw_saddr = 124;
      input.operands.raw_scale_offset = true;
    }
    const ProgramSite site = complete_site(std::move(input), target.arch, target.target);
    ASSERT_TRUE(site.lowering.normalized());
    ASSERT_TRUE(site.lowering.form);
    EXPECT_EQ(site.lowering.form->kind, AccessLoweringFormKind::FlatVectorAddress);
    EXPECT_EQ(site.lowering.form->address_vgpr_count, 2u);
    EXPECT_EQ(site.lowering.form->element_register_count, 1u);
    EXPECT_EQ(site.lowering.form->destination_register_count, 0u);
    EXPECT_TRUE(site.lowering.replay_guest_access.available());
    EXPECT_TRUE(site.lowering.compare_observed_value.available());
  }
}

TEST(ConSanAccessClassifier, FlatInstructionShapeIsOwnedByTheTargetProfile) {
  for (const TargetCase &target : kTargets) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const VectorMemoryCapability &memory = target_profile(target.arch)->vector_memory;
    const uint32_t wrong_word_count = memory.instruction_word_count == 2u ? 3u : 2u;
    ProgramSite input = flat_store_site(wrong_word_count * sizeof(uint32_t));
    input.operands.raw_saddr = 124;
    input.operands.raw_scale_offset = true;
    const ProgramSite site = complete_site(std::move(input), target.arch, target.target);
    EXPECT_FALSE(site.lowering.form);
    EXPECT_EQ(site.lowering.normalization_reason, AccessClassifierReason::UnsupportedEncoding);
  }
}

TEST(ConSanAccessClassifier, ReplayAdmissionVocabularyIsPrivateToClassifier) {
  struct Case {
    std::string_view mnemonic;
    LdsAccessKind kind;
    uint32_t width_bits;
    bool supported;
  };
  constexpr std::array cases = {
      Case{"flat_load_b128", LdsAccessKind::Read, 128u, true},
      Case{"flat_store_short", LdsAccessKind::Write, 16u, true},
      Case{"flat_load_dwordx3", LdsAccessKind::Read, 96u, false},
      Case{"global_load_dword", LdsAccessKind::Read, 32u, false},
  };
  for (const TargetCase &target : kTargets) {
    for (const Case &test : cases) {
      SCOPED_TRACE(rj_code_target_name(target.target));
      SCOPED_TRACE(test.mnemonic);
      const VectorMemoryCapability &memory = target_profile(target.arch)->vector_memory;
      ProgramSite input = flat_store_site(memory.instruction_word_count * sizeof(uint32_t));
      input.decoded_site().mnemonic = test.mnemonic;
      input.kind = test.kind;
      input.decoded_width_bits = test.width_bits;
      if (test.kind == LdsAccessKind::Read) {
        input.operands.data_vgpr.reset();
        input.operands.destination_vgpr = 7;
      }
      if (memory.instruction_word_count == 3u) {
        input.operands.raw_saddr = 124;
        input.operands.raw_scale_offset = true;
      }
      const ProgramSite site = complete_site(std::move(input), target.arch, target.target);
      EXPECT_EQ(site.lowering.replay_guest_access.available(), test.supported);
      if (test.supported && test.kind == LdsAccessKind::Read) {
        ASSERT_TRUE(site.lowering.form);
        EXPECT_EQ(site.lowering.form->destination_register_count,
                  site.lowering.form->data_register_count);
      }
      if (!test.supported) {
        EXPECT_EQ(site.lowering.replay_guest_access.reason,
                  AccessClassifierReason::UnsupportedMnemonic);
      }
    }
  }
}

TEST(ConSanAccessClassifier, MechanismSpecificRejectionsRemainTypedAndIndependent) {
  ProgramSite cdna4_flat = flat_store_site(8);
  cdna4_flat.operands.raw_ioffset = 4;
  const ProgramSite nonzero =
      complete_site(std::move(cdna4_flat), ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(nonzero.lowering.replay_guest_access.reason,
            AccessClassifierReason::NonzeroImmediateOffset);
  EXPECT_TRUE(nonzero.lowering.compare_observed_value.available());

  ProgramSite rdna4_flat = flat_store_site(12);
  rdna4_flat.operands.address_vgpr = 255;
  rdna4_flat.operands.raw_saddr = 124;
  rdna4_flat.operands.raw_scale_offset = true;
  const ProgramSite reserved =
      complete_site(std::move(rdna4_flat), ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_EQ(reserved.lowering.replay_guest_access.reason,
            AccessClassifierReason::ReservedAddressRegister);
  EXPECT_TRUE(reserved.lowering.compare_observed_value.available());

  ProgramSite invalid_scalar = flat_store_site(12);
  invalid_scalar.operands.raw_saddr = 105;
  invalid_scalar.operands.scalar_address_sgpr = 105;
  invalid_scalar.operands.raw_scale_offset = true;
  const ProgramSite scalar = complete_site(std::move(invalid_scalar), ROCJITSU_CODE_ARCH_RDNA4,
                                           ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_EQ(scalar.lowering.replay_guest_access.reason,
            AccessClassifierReason::OperandRegisterRange);
  EXPECT_TRUE(scalar.lowering.compare_observed_value.available());

  ProgramSite missing_result = native_store_site("ds_load_b32");
  missing_result.kind = LdsAccessKind::Read;
  missing_result.operands.data_vgpr.reset();
  const ProgramSite load = complete_site(std::move(missing_result), ROCJITSU_CODE_ARCH_RDNA4,
                                         ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_TRUE(load.lowering.replay_guest_access.available());
  EXPECT_EQ(load.lowering.compare_observed_value.reason,
            AccessClassifierReason::MissingResultOperand);
}

TEST(ConSanAccessClassifier, CommonNormalizationFailuresRejectEveryMechanism) {
  ProgramSite site = native_store_site("ds_store_b32");
  site.operands.address_vgpr.reset();
  const ProgramSite missing_address =
      complete_site(std::move(site), ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_FALSE(missing_address.lowering.form);
  EXPECT_EQ(missing_address.lowering.normalization_reason,
            AccessClassifierReason::MissingAddressOperand);
  EXPECT_EQ(missing_address.lowering.replay_guest_access.reason,
            AccessClassifierReason::MissingAddressOperand);
  EXPECT_EQ(missing_address.lowering.compare_observed_value.reason,
            AccessClassifierReason::MissingAddressOperand);
}

TEST(ConSanAccessClassifier, ComparisonOperandDetailsAreClassifierOwned) {
  for (const TargetCase &target : kTargets) {
    SCOPED_TRACE(rj_code_target_name(target.target));

    ProgramSite high_store =
        complete_site(native_access_site("ds_store_b8_d16_hi", LdsAccessKind::Write, 8u),
                      target.arch, target.target);
    ASSERT_TRUE(high_store.lowering.form);
    EXPECT_EQ(high_store.lowering.form->register_value_placement,
              AccessRegisterValuePlacement::High16);
    EXPECT_FALSE(high_store.lowering.form->destination_preserves_unwritten_bits);

    ProgramSite partial_load =
        complete_site(native_access_site("ds_load_u16_d16", LdsAccessKind::Read, 16u), target.arch,
                      target.target);
    ASSERT_TRUE(partial_load.lowering.form);
    EXPECT_EQ(partial_load.lowering.form->register_value_placement,
              AccessRegisterValuePlacement::Low16);
    EXPECT_TRUE(partial_load.lowering.form->destination_preserves_unwritten_bits);

    ProgramSite wide_load = complete_site(
        native_access_site("ds_load_b64", LdsAccessKind::Read, 64u), target.arch, target.target);
    ASSERT_TRUE(wide_load.lowering.form);
    EXPECT_EQ(wide_load.lowering.form->destination_allocation_headroom, 1u);
    EXPECT_EQ(wide_load.lowering.form->data_register_alignment,
              target_profile(target.arch)->requires_even_vgpr_tuples ? 2u : 1u);
  }
}

TEST(ConSanAccessClassifier, Cdna5ComparisonAllowsWideStoreAcrossVgprBankBoundary) {
  ProgramSite input = native_access_site("ds_store_b128", LdsAccessKind::Write, 128u);
  input.operands.data_vgpr = 254u;

  const ProgramSite cdna5 =
      complete_site(input, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250);
  EXPECT_TRUE(cdna5.lowering.compare_observed_value.available());

  const ProgramSite rdna4 =
      complete_site(std::move(input), ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_EQ(rdna4.lowering.compare_observed_value.reason,
            AccessClassifierReason::OperandRegisterRange);
}

} // namespace
} // namespace rocjitsu::consan
