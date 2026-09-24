// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/targets/rdna4/consan_atomic_observation.h"

namespace rocjitsu::consan {
namespace {

TEST(ConSan, AtomicObservationReturnRewritePreservesOperationAndAddress) {
  for (uint32_t family : {0xecu, 0xeeu})
    for (uint32_t op : {53u, 61u})
      for (uint32_t scope : {0u, 1u, 2u, 3u}) {
        rdna4::VflatMachineInst original{};
        original.encoding = family;
        original.op = op;
        original.saddr = 6;
        original.vaddr = 17;
        original.vsrc = 23;
        original.ioffset = 0xfffffcu;
        original.scope = scope;
        original.nv = 1;
        original.vdst = 255;
        std::array<uint32_t, 3> words;
        std::memcpy(words.data(), &original, sizeof(original));
        const auto result = detail::build_rdna4_atomic_observation(
            {reinterpret_cast<const uint8_t *>(words.data()), sizeof(words)}, 42);
        ASSERT_TRUE(result);
        EXPECT_EQ((*result)[0], words[0]);
        EXPECT_EQ((*result)[2], words[2]);
        constexpr uint32_t changed = 0xffu | (7u << 20);
        EXPECT_EQ((*result)[1] & ~changed, words[1] & ~changed);
        EXPECT_EQ((*result)[1] & changed, 42u | (1u << 20));
      }
}
TEST(ConSan, AtomicObservationReturnRewriteRejectsUnsupportedForms) {
  std::array<uint32_t, 3> words{0xee0f400cu, 0x01980000u, 0x00000002u};
  const auto build = [&](uint16_t destination = 42) {
    return detail::build_rdna4_atomic_observation(
        {reinterpret_cast<const uint8_t *>(words.data()), sizeof(words)}, destination);
  };
  EXPECT_FALSE(build()); // Already returning; cannot steal its guest destination.
  words[1] &= ~(7u << 20);
  ASSERT_TRUE(build());
  EXPECT_FALSE(build(256));
  words[1] |= 2u << 20;
  EXPECT_FALSE(build()); // Different temporal policy.
  words[1] &= ~(7u << 20);
  words[0] = 0xee01400cu;
  EXPECT_FALSE(build()); // An ordinary memory operation is not an RMW.
}

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

AtomicSite exact_flat_atomic(const AtomicTargetCase &target) {
  AtomicSite site;
  site.size = target.instruction_size;
  site.width_bits = 32u;
  site.destination_vgpr = 9u;
  site.address_vgpr = 3u;
  site.data_vgpr = 7u;
  site.raw_saddr = target.vector_only_saddr;
  if (target.arch == ROCJITSU_CODE_ARCH_CDNA5)
    site.raw_scale_offset = false;
  site.raw_vaddr = 3u;
  site.raw_ioffset = 0;
  site.scope = MemoryScope::Workgroup;
  site.raw_th = 0u;
  site.returns_old_value = true;
  site.mnemonic = "flat_atomic_add";
  return site;
}

TEST(ConSanAtomicClassifier, ExactFlatOrderingNormalizesOnAllFiveTargets) {
  for (const AtomicTargetCase &target : kAtomicTargets) {
    SCOPED_TRACE(static_cast<uint32_t>(target.arch));
    const AtomicLoweringClassification classification =
        classify_atomic_lowering(exact_flat_atomic(target), target.arch);
    ASSERT_TRUE(classification.normalized());
    ASSERT_TRUE(classification.exact_ordering_available());
    ASSERT_TRUE(classification.form);
    EXPECT_EQ(classification.form->kind, AtomicLoweringFormKind::FlatVectorAddress);
    EXPECT_EQ(classification.form->instruction_size, target.instruction_size);
    EXPECT_EQ(classification.form->value_width_bits, 32u);
    EXPECT_EQ(classification.form->address_vgpr, 3u);
    EXPECT_EQ(classification.form->address_vgpr_count, 2u);
    EXPECT_EQ(classification.form->data_vgpr, 7u);
    EXPECT_EQ(classification.form->data_register_count, 1u);
    EXPECT_EQ(classification.form->destination_vgpr, 9u);
    EXPECT_EQ(classification.form->destination_register_count, 1u);
  }
}

TEST(ConSanAtomicClassifier, Exact64BitFlatOrderingNormalizesOnAllFiveTargets) {
  for (const AtomicTargetCase &target : kAtomicTargets) {
    SCOPED_TRACE(static_cast<uint32_t>(target.arch));
    AtomicSite site = exact_flat_atomic(target);
    site.mnemonic = "flat_atomic_cmpswap_b64";
    site.width_bits = 64u;
    site.destination_vgpr = 9u;
    site.data_vgpr = 12u;
    const AtomicLoweringClassification classification = classify_atomic_lowering(site, target.arch);
    ASSERT_TRUE(classification.normalized());
    ASSERT_TRUE(classification.exact_ordering_available());
    ASSERT_TRUE(classification.form);
    EXPECT_EQ(classification.form->value_width_bits, 64u);
    EXPECT_EQ(classification.form->value_register_count, 2u);
    EXPECT_EQ(classification.form->data_register_count, 4u);
    EXPECT_EQ(classification.form->destination_register_count, 2u);
  }
}

TEST(ConSanAtomicClassifier, OrderedOrdinaryFormsOwnGuestDestinationShape) {
  const AtomicTargetCase target{ROCJITSU_CODE_ARCH_RDNA4, 12u, 0x7cu};
  AtomicSite load = exact_flat_atomic(target);
  load.mnemonic = "flat_load_dword";
  load.data_vgpr = load.destination_vgpr;
  load.returns_old_value = false;
  const AtomicLoweringClassification load_classification =
      classify_atomic_lowering(load, target.arch, /*is_rmw=*/false);
  ASSERT_TRUE(load_classification.normalized());
  ASSERT_TRUE(load_classification.form);
  EXPECT_FALSE(load_classification.form->is_rmw);
  EXPECT_EQ(load_classification.form->destination_vgpr, 9u);
  EXPECT_EQ(load_classification.form->destination_register_count, 1u);

  AtomicSite store = load;
  store.mnemonic = "flat_store_dword";
  store.destination_vgpr.reset();
  store.data_vgpr = 7u;
  const AtomicLoweringClassification store_classification =
      classify_atomic_lowering(store, target.arch, /*is_rmw=*/false);
  ASSERT_TRUE(store_classification.normalized());
  ASSERT_TRUE(store_classification.form);
  EXPECT_EQ(store_classification.form->destination_vgpr, std::nullopt);
  EXPECT_EQ(store_classification.form->destination_register_count, 0u);
}

TEST(ConSanAtomicClassifier, Rdna4Cdna5ScalarVectorAddressHasOneNormalizedForm) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA4}) {
    AtomicTargetCase target{arch, 12u, 8u};
    AtomicSite site = exact_flat_atomic(target);
    site.scalar_address_sgpr = 8u;
    site.destination_vgpr.reset();
    site.returns_old_value = false;
    const AtomicLoweringClassification classification = classify_atomic_lowering(site, arch);
    ASSERT_TRUE(classification.exact_ordering_available());
    ASSERT_TRUE(classification.form);
    EXPECT_EQ(classification.form->kind, AtomicLoweringFormKind::FlatScalarVectorAddress);
    EXPECT_EQ(classification.form->address_vgpr_count, 1u);
    EXPECT_EQ(classification.form->scalar_base_sgpr, 8u);
  }
}

TEST(ConSanAtomicClassifier, FlatDisplacementsAreMaterializedOnlyForExtendedEncodings) {
  for (const auto &target : kAtomicTargets) {
    SCOPED_TRACE(target.arch);
    for (const int32_t offset : {-(1 << 23), -12, 12, (1 << 23) - 1}) {
      SCOPED_TRACE(offset);
      for (const bool rmw : {false, true}) {
        auto site = exact_flat_atomic(target);
        site.raw_ioffset = offset;
        if (!rmw) {
          site.mnemonic = "flat_store_b32";
          site.returns_old_value = false;
          site.destination_vgpr.reset();
        }
        const auto classification = classify_atomic_lowering(site, target.arch, rmw);
        if (target.instruction_size != 12u) {
          EXPECT_FALSE(classification.address_available());
          continue;
        }
        ASSERT_TRUE(classification.exact_ordering_available());
        ASSERT_TRUE(classification.form);
        const auto plan = plan_atomic_address(*classification.form, 32u, 7u,
                                              RegisterAllocationSource::DescriptorGrowth);
        ASSERT_TRUE(plan.supported());
        EXPECT_EQ(plan.kind, AtomicAddressKind::FlatGuestPairMaterialized);
        EXPECT_EQ(plan.signed_byte_offset, offset);
        EXPECT_EQ(plan.input_address_vgpr, 3u);
        EXPECT_EQ(plan.result_address_vgpr, 37u);
        EXPECT_TRUE(build_atomic_address_materialization(plan, 82u, 84u, target.arch));
      }
    }
  }
}

TEST(ConSanAtomicClassifier, ExactOperationRejectionsRemainTypedAfterNormalization) {
  const AtomicTargetCase target{ROCJITSU_CODE_ARCH_RDNA4, 12u, 0x7cu};
  AtomicSite site = exact_flat_atomic(target);
  site.mnemonic = "flat_atomic_cmpswap";
  site.returns_old_value = false;
  site.destination_vgpr.reset();
  const AtomicLoweringClassification no_outcome = classify_atomic_lowering(site, target.arch);
  ASSERT_TRUE(no_outcome.normalized());
  EXPECT_EQ(no_outcome.exact_ordering_reason,
            AtomicClassifierReason::CompareExchangeOutcomeUnavailable);

  site = exact_flat_atomic(target);
  site.scope = MemoryScope::Wavefront;
  const AtomicLoweringClassification wave_scope = classify_atomic_lowering(site, target.arch);
  ASSERT_TRUE(wave_scope.normalized());
  EXPECT_EQ(wave_scope.exact_ordering_reason, AtomicClassifierReason::UnsupportedScope);

  site = exact_flat_atomic(target);
  site.raw_ioffset = 1 << 23;
  const AtomicLoweringClassification nonzero = classify_atomic_lowering(site, target.arch);
  EXPECT_FALSE(nonzero.normalized());
  EXPECT_EQ(nonzero.normalization_reason, AtomicClassifierReason::UnsupportedOffset);

  site = exact_flat_atomic(target);
  site.mnemonic = "flat_atomic_cmpswap";
  site.width_bits = 64u;
  site.data_vgpr = 254u;
  const AtomicLoweringClassification overflowing_data = classify_atomic_lowering(site, target.arch);
  EXPECT_FALSE(overflowing_data.normalized());
  EXPECT_EQ(overflowing_data.normalization_reason, AtomicClassifierReason::UnsupportedInputWidth);
}

TEST(ConSanAtomicClassifier, CausalTargetFormsNormalizeLdsBufferAndSignedGlobalAddresses) {
  AtomicSite lds;
  lds.size = 8u;
  lds.width_bits = 32u;
  lds.destination_vgpr = 9u;
  lds.address_vgpr = 3u;
  lds.data_vgpr = 7u;
  lds.raw_addr = 3u;
  lds.raw_data0 = 7u;
  lds.raw_ioffset = 12;
  lds.scope = MemoryScope::Workgroup;
  lds.returns_old_value = true;
  lds.mnemonic = "ds_add_u32";
  const AtomicLoweringClassification lds_form =
      classify_atomic_lowering(lds, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(lds_form.address_available());
  EXPECT_TRUE(lds_form.causal_ordering_available());
  EXPECT_FALSE(lds_form.exact_ordering_available());
  ASSERT_TRUE(lds_form.form);
  EXPECT_EQ(lds_form.form->kind, AtomicLoweringFormKind::LdsVectorOffset);
  EXPECT_EQ(lds_form.form->signed_byte_offset, 12);

  AtomicSite buffer;
  buffer.size = 12u;
  buffer.width_bits = 32u;
  buffer.destination_vgpr = 11u;
  buffer.address_vgpr = 5u;
  buffer.data_vgpr = 8u;
  buffer.scalar_address_sgpr = 24u;
  buffer.raw_rsrc = 24u;
  buffer.raw_soffset = 10u;
  buffer.raw_vaddr = 5u;
  buffer.raw_ioffset = -16;
  buffer.raw_offen = true;
  buffer.raw_idxen = false;
  buffer.scope = MemoryScope::Agent;
  buffer.returns_old_value = true;
  buffer.mnemonic = "buffer_atomic_add_u32";
  const AtomicLoweringClassification buffer_form =
      classify_atomic_lowering(buffer, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(buffer_form.address_available());
  EXPECT_TRUE(buffer_form.causal_ordering_available());
  EXPECT_FALSE(buffer_form.exact_ordering_available());
  ASSERT_TRUE(buffer_form.form);
  EXPECT_EQ(buffer_form.form->kind, AtomicLoweringFormKind::BufferResourceVectorOffset);
  EXPECT_EQ(buffer_form.form->scalar_base_sgpr, 24u);
  EXPECT_EQ(buffer_form.form->scalar_offset_sgpr, 10u);
  EXPECT_EQ(classify_atomic_lowering(buffer, ROCJITSU_CODE_ARCH_RDNA4).normalization_reason,
            AtomicClassifierReason::UnsupportedAddressSource);

  AtomicTargetCase cdna4{ROCJITSU_CODE_ARCH_CDNA4, 8u, 8u};
  AtomicSite global = exact_flat_atomic(cdna4);
  global.mnemonic = "global_atomic_add_u32";
  global.scalar_address_sgpr = 8u;
  const AtomicLoweringClassification global_form = classify_atomic_lowering(global, cdna4.arch);
  ASSERT_TRUE(global_form.address_available());
  ASSERT_TRUE(global_form.form);
  EXPECT_EQ(global_form.form->kind, AtomicLoweringFormKind::GlobalScalarVectorAddress);
  EXPECT_TRUE(global_form.form->sign_extend_vector_offset);
}

TEST(ConSanAtomicClassifier, AddressPlannerConsumesTheNormalizedFormWithoutReclassification) {
  const AtomicTargetCase target{ROCJITSU_CODE_ARCH_RDNA4, 12u, 0x7cu};
  AtomicSite site = exact_flat_atomic(target);
  const AtomicLoweringClassification classification = classify_atomic_lowering(site, target.arch);
  ASSERT_TRUE(classification.address_available());
  ASSERT_TRUE(classification.form);

  site.raw_saddr.reset();
  EXPECT_EQ(plan_atomic_address(site, 20u, 3u, RegisterAllocationSource::LivenessDead, target.arch)
                .support,
            AtomicAddressSupport::UnsupportedEncoding);
  const AtomicAddressPlan plan =
      plan_atomic_address(*classification.form, 20u, 3u, RegisterAllocationSource::LivenessDead);
  EXPECT_TRUE(plan.supported());
  EXPECT_EQ(plan.kind, AtomicAddressKind::FlatGuestPair);
}

} // namespace
} // namespace rocjitsu::consan
