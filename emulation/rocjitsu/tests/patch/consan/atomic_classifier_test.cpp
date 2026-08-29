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
    EXPECT_EQ(classification.form->scope, 1u);
  }
}

TEST(ConSanAtomicClassifier, Gfx12ScalarVectorAddressHasOneNormalizedForm) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4}) {
    AtomicTargetCase target{arch, 12u, 8u};
    ConSanAtomicSite site = exact_flat_atomic(target);
    site.saddr_sgpr = 8u;
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
  EXPECT_EQ(nonzero.normalization_reason, ConSanAtomicClassifierReason::NonzeroImmediateOffset);
}

} // namespace
} // namespace rocjitsu
