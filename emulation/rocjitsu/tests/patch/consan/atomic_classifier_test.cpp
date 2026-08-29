// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

struct AtomicTargetCase {
  rj_code_arch_t arch;
  uint32_t instruction_size;
  uint32_t vector_only_saddr;
};

constexpr std::array kAtomicTargets = {
    AtomicTargetCase{ROCJITSU_CODE_ARCH_CDNA3, 8u, 0u},
    AtomicTargetCase{ROCJITSU_CODE_ARCH_CDNA4, 8u, 0u},
    AtomicTargetCase{ROCJITSU_CODE_ARCH_RDNA3, 8u, 0x7cu},
    AtomicTargetCase{ROCJITSU_CODE_ARCH_CDNA5, 12u, 0x7cu},
    AtomicTargetCase{ROCJITSU_CODE_ARCH_RDNA4, 12u, 0x7cu},
};

ConSanAtomicSite exact_flat_atomic(const AtomicTargetCase &target) {
  ConSanAtomicSite site;
  site.size = target.instruction_size;
  site.width_bits = 32u;
  site.dst_vgpr = 9u;
  site.addr_vgpr = 3u;
  site.data_vgpr = 7u;
  site.raw_saddr = target.vector_only_saddr;
  if (target.arch == ROCJITSU_CODE_ARCH_CDNA5)
    site.raw_scale_offset = false;
  site.raw_vaddr = 3u;
  site.raw_ioffset = 0;
  site.raw_scope = 1u;
  site.raw_th = 0u;
  site.returns_old_value = true;
  site.mnemonic = "flat_atomic_add";
  return site;
}

TEST(ConSanAtomicClassifier, ExactFlatOrderingNormalizesOnAllFiveTargets) {
  for (const AtomicTargetCase &target : kAtomicTargets) {
    SCOPED_TRACE(static_cast<uint32_t>(target.arch));
    const ConSanAtomicLoweringClassification classification =
        classify_consan_atomic_lowering(exact_flat_atomic(target), target.arch);
    ASSERT_TRUE(classification.normalized());
    ASSERT_TRUE(classification.exact_ordering_available());
    ASSERT_TRUE(classification.form);
    EXPECT_EQ(classification.form->kind, ConSanAtomicLoweringFormKind::FlatVectorAddress);
    EXPECT_EQ(classification.form->instruction_size, target.instruction_size);
    EXPECT_EQ(classification.form->value_width_bits, 32u);
    EXPECT_EQ(classification.form->address_vgpr, 3u);
    EXPECT_EQ(classification.form->address_vgpr_count, 2u);
    EXPECT_EQ(classification.form->data_vgpr, 7u);
    EXPECT_EQ(classification.form->data_register_count, 1u);
    EXPECT_EQ(classification.form->destination_vgpr, 9u);
    EXPECT_EQ(classification.form->destination_register_count, 1u);
    EXPECT_EQ(classification.form->scope, 1u);
  }
}

TEST(ConSanAtomicClassifier, OrderedOrdinaryFormsOwnGuestDestinationShape) {
  const AtomicTargetCase target{ROCJITSU_CODE_ARCH_RDNA4, 12u, 0x7cu};
  ConSanAtomicSite load = exact_flat_atomic(target);
  load.mnemonic = "flat_load_dword";
  load.data_vgpr = load.dst_vgpr;
  load.returns_old_value = false;
  const ConSanAtomicLoweringClassification load_classification =
      classify_consan_atomic_lowering(load, target.arch, /*is_rmw=*/false);
  ASSERT_TRUE(load_classification.normalized());
  ASSERT_TRUE(load_classification.form);
  EXPECT_FALSE(load_classification.form->is_rmw);
  EXPECT_EQ(load_classification.form->destination_vgpr, 9u);
  EXPECT_EQ(load_classification.form->destination_register_count, 1u);

  ConSanAtomicSite store = load;
  store.mnemonic = "flat_store_dword";
  store.dst_vgpr.reset();
  store.data_vgpr = 7u;
  const ConSanAtomicLoweringClassification store_classification =
      classify_consan_atomic_lowering(store, target.arch, /*is_rmw=*/false);
  ASSERT_TRUE(store_classification.normalized());
  ASSERT_TRUE(store_classification.form);
  EXPECT_EQ(store_classification.form->destination_vgpr, std::nullopt);
  EXPECT_EQ(store_classification.form->destination_register_count, 0u);
}

TEST(ConSanAtomicClassifier, Gfx12ScalarVectorAddressHasOneNormalizedForm) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4}) {
    AtomicTargetCase target{arch, 12u, 8u};
    ConSanAtomicSite site = exact_flat_atomic(target);
    site.saddr_sgpr = 8u;
    site.dst_vgpr.reset();
    site.returns_old_value = false;
    const ConSanAtomicLoweringClassification classification =
        classify_consan_atomic_lowering(site, arch);
    ASSERT_TRUE(classification.exact_ordering_available());
    ASSERT_TRUE(classification.form);
    EXPECT_EQ(classification.form->kind, ConSanAtomicLoweringFormKind::FlatScalarVectorAddress);
    EXPECT_EQ(classification.form->address_vgpr_count, 1u);
    EXPECT_EQ(classification.form->scalar_base_sgpr, 8u);
  }
}

TEST(ConSanAtomicClassifier, ExactOperationRejectionsRemainTypedAfterNormalization) {
  const AtomicTargetCase target{ROCJITSU_CODE_ARCH_RDNA4, 12u, 0x7cu};
  ConSanAtomicSite site = exact_flat_atomic(target);
  site.mnemonic = "flat_atomic_cmpswap";
  site.returns_old_value = false;
  site.dst_vgpr.reset();
  const ConSanAtomicLoweringClassification no_outcome =
      classify_consan_atomic_lowering(site, target.arch);
  ASSERT_TRUE(no_outcome.normalized());
  EXPECT_EQ(no_outcome.exact_ordering_reason,
            ConSanAtomicClassifierReason::CompareExchangeOutcomeUnavailable);

  site = exact_flat_atomic(target);
  site.raw_scope = 0u;
  const ConSanAtomicLoweringClassification wave_scope =
      classify_consan_atomic_lowering(site, target.arch);
  ASSERT_TRUE(wave_scope.normalized());
  EXPECT_EQ(wave_scope.exact_ordering_reason, ConSanAtomicClassifierReason::UnsupportedScope);

  site = exact_flat_atomic(target);
  site.raw_ioffset = 4;
  const ConSanAtomicLoweringClassification nonzero =
      classify_consan_atomic_lowering(site, target.arch);
  EXPECT_FALSE(nonzero.normalized());
  EXPECT_EQ(nonzero.normalization_reason, ConSanAtomicClassifierReason::UnsupportedOffset);

  site = exact_flat_atomic(target);
  site.mnemonic = "flat_atomic_cmpswap";
  site.width_bits = 64u;
  site.data_vgpr = 254u;
  const ConSanAtomicLoweringClassification overflowing_data =
      classify_consan_atomic_lowering(site, target.arch);
  EXPECT_FALSE(overflowing_data.normalized());
  EXPECT_EQ(overflowing_data.normalization_reason,
            ConSanAtomicClassifierReason::UnsupportedInputWidth);
}

TEST(ConSanAtomicClassifier, CausalTargetFormsNormalizeLdsBufferAndSignedGlobalAddresses) {
  ConSanAtomicSite lds;
  lds.size = 8u;
  lds.width_bits = 32u;
  lds.dst_vgpr = 9u;
  lds.addr_vgpr = 3u;
  lds.data_vgpr = 7u;
  lds.raw_addr = 3u;
  lds.raw_data0 = 7u;
  lds.raw_ioffset = 12;
  lds.raw_scope = 1u;
  lds.returns_old_value = true;
  lds.mnemonic = "ds_add_u32";
  const ConSanAtomicLoweringClassification lds_form =
      classify_consan_atomic_lowering(lds, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(lds_form.address_available());
  EXPECT_TRUE(lds_form.causal_ordering_available());
  EXPECT_FALSE(lds_form.exact_ordering_available());
  ASSERT_TRUE(lds_form.form);
  EXPECT_EQ(lds_form.form->kind, ConSanAtomicLoweringFormKind::LdsVectorOffset);
  EXPECT_EQ(lds_form.form->signed_byte_offset, 12);

  ConSanAtomicSite buffer;
  buffer.size = 12u;
  buffer.width_bits = 32u;
  buffer.dst_vgpr = 11u;
  buffer.addr_vgpr = 5u;
  buffer.data_vgpr = 8u;
  buffer.saddr_sgpr = 24u;
  buffer.raw_rsrc = 24u;
  buffer.raw_soffset = 10u;
  buffer.raw_vaddr = 5u;
  buffer.raw_ioffset = -16;
  buffer.raw_offen = true;
  buffer.raw_idxen = false;
  buffer.raw_scope = 2u;
  buffer.returns_old_value = true;
  buffer.mnemonic = "buffer_atomic_add_u32";
  const ConSanAtomicLoweringClassification buffer_form =
      classify_consan_atomic_lowering(buffer, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(buffer_form.address_available());
  EXPECT_TRUE(buffer_form.causal_ordering_available());
  EXPECT_FALSE(buffer_form.exact_ordering_available());
  ASSERT_TRUE(buffer_form.form);
  EXPECT_EQ(buffer_form.form->kind, ConSanAtomicLoweringFormKind::BufferResourceVectorOffset);
  EXPECT_EQ(buffer_form.form->scalar_base_sgpr, 24u);
  EXPECT_EQ(buffer_form.form->scalar_offset_sgpr, 10u);

  AtomicTargetCase cdna4{ROCJITSU_CODE_ARCH_CDNA4, 8u, 8u};
  ConSanAtomicSite global = exact_flat_atomic(cdna4);
  global.mnemonic = "global_atomic_add_u32";
  global.saddr_sgpr = 8u;
  const ConSanAtomicLoweringClassification global_form =
      classify_consan_atomic_lowering(global, cdna4.arch);
  ASSERT_TRUE(global_form.address_available());
  ASSERT_TRUE(global_form.form);
  EXPECT_EQ(global_form.form->kind, ConSanAtomicLoweringFormKind::GlobalScalarVectorAddress);
  EXPECT_TRUE(global_form.form->sign_extend_vector_offset);
}

TEST(ConSanAtomicClassifier, AddressPlannerConsumesTheNormalizedFormWithoutReclassification) {
  const AtomicTargetCase target{ROCJITSU_CODE_ARCH_RDNA4, 12u, 0x7cu};
  ConSanAtomicSite site = exact_flat_atomic(target);
  const ConSanAtomicLoweringClassification classification =
      classify_consan_atomic_lowering(site, target.arch);
  ASSERT_TRUE(classification.address_available());
  ASSERT_TRUE(classification.form);

  site.raw_saddr.reset();
  EXPECT_EQ(plan_consan_moi_atomic_address(
                site, 20u, 3u, ConSanRegisterAllocationSource::LivenessDead, target.arch)
                .support,
            ConSanMoiAtomicAddressSupport::UnsupportedEncoding);
  const ConSanMoiAtomicAddressPlan plan = plan_consan_moi_atomic_address(
      *classification.form, 20u, 3u, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_TRUE(plan.supported());
  EXPECT_EQ(plan.kind, ConSanMoiAtomicAddressKind::FlatGuestPair);
}

} // namespace
} // namespace rocjitsu
