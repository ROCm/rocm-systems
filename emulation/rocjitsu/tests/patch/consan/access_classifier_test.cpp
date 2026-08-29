// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
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

ConSanAccessInventorySite native_store_site(std::string_view mnemonic) {
  ConSanAccessInventorySite site;
  site.origin = ConSanAccessOrigin::NativeLds;
  site.kind = ConSanLdsAccessKind::Write;
  site.physical_id.original_text_offset = 8;
  site.file_offset = 8;
  site.instruction_size = 8;
  site.decoded_width_bits = 32;
  site.operands.address_vgpr = 3;
  site.operands.data_vgpr = 7;
  site.mnemonic = std::string(mnemonic);
  return site;
}

ConSanAccessInventorySite native_access_site(std::string_view mnemonic, ConSanLdsAccessKind kind,
                                             uint32_t width_bits) {
  ConSanAccessInventorySite site = native_store_site(mnemonic);
  site.kind = kind;
  site.decoded_width_bits = width_bits;
  if (kind == ConSanLdsAccessKind::Read) {
    site.operands.data_vgpr.reset();
    site.operands.destination_vgpr = 7;
  }
  return site;
}

ConSanAccessInventorySite complete_site(ConSanAccessInventorySite site, rj_code_arch_t arch,
                                        rj_code_target_id_t target);

TEST(ConSanAccessClassifier, NativeB96SpellingIsOwnedByTheTargetClassifier) {
  for (const TargetCase &target : kTargets) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    ConSanAccessInventorySite expected = native_store_site(target.native_b96_store);
    expected.decoded_width_bits = 96;
    const ConSanAccessInventorySite admitted =
        complete_site(std::move(expected), target.arch, target.target);
    EXPECT_TRUE(admitted.lowering.replay_guest_access.available());

    const std::string_view other_spelling =
        target.native_b96_store == "ds_write_b96" ? "ds_store_b96" : "ds_write_b96";
    ConSanAccessInventorySite mismatched = native_store_site(other_spelling);
    mismatched.decoded_width_bits = 96;
    const ConSanAccessInventorySite rejected =
        complete_site(std::move(mismatched), target.arch, target.target);
    EXPECT_EQ(rejected.lowering.replay_guest_access.reason,
              ConSanAccessClassifierReason::UnsupportedMnemonic);
  }
}

ConSanAccessInventorySite flat_store_site(uint32_t instruction_size) {
  ConSanAccessInventorySite site;
  site.origin = ConSanAccessOrigin::Flat;
  site.kind = ConSanLdsAccessKind::Write;
  site.physical_id.original_text_offset = 8;
  site.file_offset = 8;
  site.instruction_size = instruction_size;
  site.decoded_width_bits = 32;
  site.operands.address_vgpr = 3;
  site.operands.data_vgpr = 7;
  site.operands.raw_segment = 0;
  site.operands.raw_ioffset = 0;
  site.flat_address_space_hint = ConSanFlatAddressSpaceHint::Group;
  site.mnemonic = "flat_store_b32";
  return site;
}

ConSanAccessInventorySite complete_site(ConSanAccessInventorySite site, rj_code_arch_t arch,
                                        rj_code_target_id_t target) {
  const std::array<uint8_t, 64> bytes = {};
  ProgramInventoryBuilder builder(bytes);
  builder.set_code_object_facts(true, 0, arch, target);
  ConSanKernelInfo kernel;
  kernel.name = "classifier_kernel";
  kernel.descriptor_file_offset = 48;
  kernel.entry_text_offset = 0;
  site.container = consan_program_container_ref(kernel);
  builder.kernels().push_back(kernel);
  builder.access_sites().push_back(std::move(site));
  builder.publish_decoded_accesses(bytes);
  return builder.view().access_sites().front();
}

TEST(ConSanAccessClassifier, NativeReplayAndValueComparisonNormalizeOnAllFiveTargets) {
  for (const TargetCase &target : kTargets) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const ConSanAccessInventorySite site =
        complete_site(native_store_site(target.native_store), target.arch, target.target);
    ASSERT_TRUE(site.lowering.normalized());
    ASSERT_TRUE(site.lowering.form);
    EXPECT_EQ(site.lowering.form->kind, ConSanAccessLoweringFormKind::NativeSingleRange);
    EXPECT_EQ(site.lowering.form->range_count, 1u);
    EXPECT_EQ(site.lowering.form->element_width_bits, 32u);
    EXPECT_EQ(site.lowering.form->element_register_count, 1u);
    EXPECT_EQ(site.lowering.form->data_register_count, 1u);
    EXPECT_EQ(site.lowering.form->destination_register_count, 0u);
    EXPECT_EQ(site.lowering.form->address_vgpr_count, 1u);
    EXPECT_TRUE(site.lowering.replay_guest_access.available());
    EXPECT_TRUE(site.lowering.compare_observed_value.available());
    EXPECT_EQ(site.lowering, classify_consan_access_lowering(site, target.arch));
  }
}

TEST(ConSanAccessClassifier, FlatEncodingDifferencesProduceOneNormalizedVocabulary) {
  for (const TargetCase &target : kTargets) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const bool gfx12 = consan_uses_gfx12_encoding(target.arch);
    ConSanAccessInventorySite input = flat_store_site(gfx12 ? 12u : 8u);
    if (gfx12) {
      input.operands.raw_saddr = 124;
      input.operands.raw_scale_offset = true;
    }
    const ConSanAccessInventorySite site =
        complete_site(std::move(input), target.arch, target.target);
    ASSERT_TRUE(site.lowering.normalized());
    ASSERT_TRUE(site.lowering.form);
    EXPECT_EQ(site.lowering.form->kind, ConSanAccessLoweringFormKind::FlatVectorAddress);
    EXPECT_EQ(site.lowering.form->address_vgpr_count, 2u);
    EXPECT_EQ(site.lowering.form->element_register_count, 1u);
    EXPECT_EQ(site.lowering.form->destination_register_count, 0u);
    EXPECT_TRUE(site.lowering.replay_guest_access.available());
    EXPECT_TRUE(site.lowering.compare_observed_value.available());
  }
}

TEST(ConSanAccessClassifier, ReplayAdmissionVocabularyIsPrivateToClassifier) {
  struct Case {
    std::string_view mnemonic;
    ConSanLdsAccessKind kind;
    uint32_t width_bits;
    bool supported;
  };
  constexpr std::array cases = {
      Case{"flat_load_b128", ConSanLdsAccessKind::Read, 128u, true},
      Case{"flat_store_short", ConSanLdsAccessKind::Write, 16u, true},
      Case{"flat_load_dwordx3", ConSanLdsAccessKind::Read, 96u, false},
      Case{"global_load_dword", ConSanLdsAccessKind::Read, 32u, false},
  };
  for (const TargetCase &target : kTargets) {
    for (const Case &test : cases) {
      SCOPED_TRACE(rj_code_target_name(target.target));
      SCOPED_TRACE(test.mnemonic);
      const bool gfx12 = consan_uses_gfx12_encoding(target.arch);
      ConSanAccessInventorySite input = flat_store_site(gfx12 ? 12u : 8u);
      input.mnemonic = test.mnemonic;
      input.kind = test.kind;
      input.decoded_width_bits = test.width_bits;
      if (test.kind == ConSanLdsAccessKind::Read) {
        input.operands.data_vgpr.reset();
        input.operands.destination_vgpr = 7;
      }
      if (gfx12) {
        input.operands.raw_saddr = 124;
        input.operands.raw_scale_offset = true;
      }
      const ConSanAccessInventorySite site =
          complete_site(std::move(input), target.arch, target.target);
      EXPECT_EQ(site.lowering.replay_guest_access.available(), test.supported);
      if (test.supported && test.kind == ConSanLdsAccessKind::Read) {
        ASSERT_TRUE(site.lowering.form);
        EXPECT_EQ(site.lowering.form->destination_register_count,
                  site.lowering.form->data_register_count);
      }
      if (!test.supported) {
        EXPECT_EQ(site.lowering.replay_guest_access.reason,
                  ConSanAccessClassifierReason::UnsupportedMnemonic);
      }
    }
  }
}

TEST(ConSanAccessClassifier, MechanismSpecificRejectionsRemainTypedAndIndependent) {
  ConSanAccessInventorySite cdna4_flat = flat_store_site(8);
  cdna4_flat.operands.raw_ioffset = 4;
  const ConSanAccessInventorySite nonzero =
      complete_site(std::move(cdna4_flat), ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(nonzero.lowering.replay_guest_access.reason,
            ConSanAccessClassifierReason::NonzeroImmediateOffset);
  EXPECT_TRUE(nonzero.lowering.compare_observed_value.available());

  ConSanAccessInventorySite rdna4_flat = flat_store_site(12);
  rdna4_flat.operands.address_vgpr = 255;
  rdna4_flat.operands.raw_saddr = 124;
  rdna4_flat.operands.raw_scale_offset = true;
  const ConSanAccessInventorySite reserved =
      complete_site(std::move(rdna4_flat), ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_EQ(reserved.lowering.replay_guest_access.reason,
            ConSanAccessClassifierReason::ReservedAddressRegister);
  EXPECT_TRUE(reserved.lowering.compare_observed_value.available());

  ConSanAccessInventorySite invalid_scalar = flat_store_site(12);
  invalid_scalar.operands.raw_saddr = 105;
  invalid_scalar.operands.raw_scale_offset = true;
  const ConSanAccessInventorySite scalar = complete_site(
      std::move(invalid_scalar), ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_EQ(scalar.lowering.replay_guest_access.reason,
            ConSanAccessClassifierReason::OperandRegisterRange);
  EXPECT_TRUE(scalar.lowering.compare_observed_value.available());

  ConSanAccessInventorySite missing_result = native_store_site("ds_load_b32");
  missing_result.kind = ConSanLdsAccessKind::Read;
  missing_result.operands.data_vgpr.reset();
  const ConSanAccessInventorySite load = complete_site(
      std::move(missing_result), ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_TRUE(load.lowering.replay_guest_access.available());
  EXPECT_EQ(load.lowering.compare_observed_value.reason,
            ConSanAccessClassifierReason::MissingResultOperand);
}

TEST(ConSanAccessClassifier, CommonNormalizationFailuresRejectEveryMechanism) {
  ConSanAccessInventorySite site = native_store_site("ds_store_b32");
  site.operands.address_vgpr.reset();
  const ConSanAccessInventorySite missing_address =
      complete_site(std::move(site), ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_FALSE(missing_address.lowering.form);
  EXPECT_EQ(missing_address.lowering.normalization_reason,
            ConSanAccessClassifierReason::MissingAddressOperand);
  EXPECT_EQ(missing_address.lowering.replay_guest_access.reason,
            ConSanAccessClassifierReason::MissingAddressOperand);
  EXPECT_EQ(missing_address.lowering.compare_observed_value.reason,
            ConSanAccessClassifierReason::MissingAddressOperand);
}

TEST(ConSanAccessClassifier, ComparisonOperandDetailsAreClassifierOwned) {
  for (const TargetCase &target : kTargets) {
    SCOPED_TRACE(rj_code_target_name(target.target));

    ConSanAccessInventorySite high_store =
        complete_site(native_access_site("ds_store_b8_d16_hi", ConSanLdsAccessKind::Write, 8u),
                      target.arch, target.target);
    ASSERT_TRUE(high_store.lowering.form);
    EXPECT_EQ(high_store.lowering.form->register_value_placement,
              ConSanAccessRegisterValuePlacement::High16);
    EXPECT_FALSE(high_store.lowering.form->destination_preserves_unwritten_bits);

    ConSanAccessInventorySite partial_load =
        complete_site(native_access_site("ds_load_u16_d16", ConSanLdsAccessKind::Read, 16u),
                      target.arch, target.target);
    ASSERT_TRUE(partial_load.lowering.form);
    EXPECT_EQ(partial_load.lowering.form->register_value_placement,
              ConSanAccessRegisterValuePlacement::Low16);
    EXPECT_TRUE(partial_load.lowering.form->destination_preserves_unwritten_bits);

    ConSanAccessInventorySite wide_load =
        complete_site(native_access_site("ds_load_b64", ConSanLdsAccessKind::Read, 64u),
                      target.arch, target.target);
    ASSERT_TRUE(wide_load.lowering.form);
    EXPECT_EQ(wide_load.lowering.form->destination_allocation_headroom, 1u);
    EXPECT_EQ(wide_load.lowering.form->data_register_alignment,
              consan_target_profile(target.arch)->requires_even_vgpr_tuples ? 2u : 1u);
  }
}

} // namespace
} // namespace rocjitsu
