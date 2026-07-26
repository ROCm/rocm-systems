// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu {
namespace {

TEST(ConSanInstructionBuilder, EncodesInlineAtomicAddressOperations) {
  const auto xor_word = build_v_xor_b32_e32(
      /*vdst=*/7, vector_source_vgpr(8), /*vsrc1=*/9, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(xor_word);
  EXPECT_EQ(*xor_word, pack_vop2(/*op=*/29, /*vdst=*/7, vector_source_vgpr(8), /*vsrc1=*/9));

  const auto store = build_flat_store_b32_vaddr_vsrc(
      /*vaddr=*/10, /*vsrc=*/12, ROCJITSU_CODE_ARCH_RDNA4, /*byte_offset=*/20);
  ASSERT_TRUE(store);
  EXPECT_EQ((*store)[2], 10u | (20u << 8u));
  const auto load = build_flat_load_b32_vaddr_vdst(
      /*vaddr=*/10, /*vdst=*/12, ROCJITSU_CODE_ARCH_RDNA4, /*byte_offset=*/16);
  ASSERT_TRUE(load);
  EXPECT_EQ((*load)[2], 10u | (16u << 8u));
  const auto atomic_snapshot = build_flat_atomic_add_u64_vaddr_vsrc_vdst(
      /*vaddr=*/10, /*vsrc=*/12, /*vdst=*/14, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_snapshot);
  EXPECT_EQ((*atomic_snapshot)[0], 0xEC10C07Cu);
  EXPECT_EQ((*atomic_snapshot)[1], 14u | (2u << 18u) | (1u << 20u) | (12u << 23u));
  EXPECT_EQ((*atomic_snapshot)[2], 10u);

  const auto add = build_v_add_u64_vgpr_offset(
      /*address_vgpr=*/10, /*offset_vgpr=*/12, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(add);
  EXPECT_EQ((*add)[2], pack_sopp(rdna4::kSWaitAlu, 0xfffdu));
  EXPECT_EQ((*add)[0] & 0xffu, 10u);
  EXPECT_EQ(((*add)[1] >> 18u) & 0x1ffu, scalar_positive_inline_u32(0));
  EXPECT_EQ((*add)[3] & 0xffu, 11u);
}

TEST(ConSanInstructionBuilder, EncodesLdsStoreExchangeReturnB64) {
  const auto exchange = build_ds_storexchg_rtn_b64(
      /*vdst=*/20, /*vaddr=*/8, /*vdata=*/12, /*byte_offset=*/16, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(exchange);
  EXPECT_EQ((*exchange)[0], 0xD9B40010u);
  // VDATA names the complete pair; VDATA1 is reserved for this opcode and
  // must remain zero. LLVM's gfx1201 assembler emits 0x14000c08 here.
  EXPECT_EQ((*exchange)[1], 0x14000C08u);

  EXPECT_FALSE(build_ds_storexchg_rtn_b64(255, 8, 12, 0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_ds_storexchg_rtn_b64(20, 8, 255, 0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_ds_storexchg_rtn_b64(20, 8, 12, 0, ROCJITSU_CODE_ARCH_RDNA3));
}

TEST(ConSanInstructionBuilder, RejectsInvalidInlineAtomicAddressOperands) {
  EXPECT_FALSE(build_v_xor_b32_e32(256, 0, 0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_flat_store_b32_vaddr_vsrc(0, 0, ROCJITSU_CODE_ARCH_RDNA4, 0x1000000u));
  EXPECT_FALSE(build_flat_load_b32_vaddr_vdst(0, 0, ROCJITSU_CODE_ARCH_RDNA4, 0x1000000u));
  EXPECT_FALSE(
      build_flat_atomic_add_u64_vaddr_vsrc_vdst(255, 0, 0, true, 2, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_add_u64_vgpr_offset(255, 0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_add_u64_vgpr_offset(0, 0, ROCJITSU_CODE_ARCH_RDNA3));
  EXPECT_FALSE(build_v_add_u64_signed_i24(255, 0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_add_u64_signed_i24(0, -(1 << 23) - 1, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_add_u64_signed_i24(0, 1 << 23, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_FALSE(build_v_add_u64_signed_i24(0, 0, ROCJITSU_CODE_ARCH_RDNA3));
}

TEST(ConSanInstructionBuilder, EncodesSignedAddressDisplacementWithCarry) {
  const auto positive =
      build_v_add_u64_signed_i24(/*address_vgpr=*/10, 0x7fffff, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(positive);
  EXPECT_EQ((*positive)[2], 0x007fffffu);
  EXPECT_EQ((*positive)[3], pack_sopp(rdna4::kSWaitAlu, 0xfffdu));
  EXPECT_EQ((*positive)[0] & 0xffu, 10u);
  EXPECT_EQ((*positive)[4] & 0xffu, 11u);
  EXPECT_EQ(((*positive)[0] >> 8u) & 0x7fu, kRdna4VccLo);
  EXPECT_EQ(((*positive)[1] >> 18u) & 0x1ffu, scalar_positive_inline_u32(0));
  EXPECT_EQ(((*positive)[4] >> 8u) & 0x7fu, kRdna4VccLo);
  EXPECT_EQ(((*positive)[5] >> 18u) & 0x1ffu, kRdna4VccLo);
  EXPECT_EQ((*positive)[5] & 0x1ffu, scalar_positive_inline_u32(0));

  const auto negative =
      build_v_add_u64_signed_i24(/*address_vgpr=*/10, -4, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(negative);
  EXPECT_EQ((*negative)[2], 0xfffffffcu);
  // The high add consumes carry and adds the sign-extension dword (-1).
  EXPECT_EQ((*negative)[5] & 0x1ffu, 193u);
}

TEST(ConSanInstructionBuilder, SignExtendsCdnaVgprOffsetBefore64BitAdd) {
  constexpr uint16_t kAddressVgpr = 12u;
  constexpr uint16_t kOffsetVgpr = 4u;
  constexpr uint16_t kSignVgpr = 5u;
  constexpr std::array<uint32_t, 3> kExpected = {
      0x220a089fu, // v_ashrrev_i32_e32 v5, 31, v4
      0x32181904u, // v_add_co_u32_e32 v12, vcc, v4, v12
      0x381a1b05u, // v_addc_co_u32_e32 v13, vcc, v5, v13, vcc
  };
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    const auto add = instrumentation::build_v_add_u64_signed_vgpr_offset(kAddressVgpr, kOffsetVgpr,
                                                                         kSignVgpr, arch);
    ASSERT_TRUE(add);
    EXPECT_EQ(*add, (std::vector<uint32_t>{kExpected.begin(), kExpected.end()}));
  }
  EXPECT_FALSE(instrumentation::build_v_add_u64_signed_vgpr_offset(
      kAddressVgpr, kOffsetVgpr, kOffsetVgpr, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(instrumentation::build_v_add_u64_signed_vgpr_offset(
      kAddressVgpr, kOffsetVgpr, kAddressVgpr, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(instrumentation::build_v_add_u64_signed_vgpr_offset(
      kAddressVgpr, kOffsetVgpr, kAddressVgpr + 1u, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(instrumentation::build_v_add_u64_signed_vgpr_offset(
      /*address_vgpr=*/255u, kOffsetVgpr, kSignVgpr, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(instrumentation::build_v_add_u64_signed_vgpr_offset(
      kAddressVgpr, /*offset_vgpr=*/256u, kSignVgpr, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(instrumentation::build_v_add_u64_signed_vgpr_offset(
      kAddressVgpr, kOffsetVgpr, /*sign_vgpr=*/256u, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_FALSE(instrumentation::build_v_add_u64_signed_vgpr_offset(
      kAddressVgpr, kOffsetVgpr, kSignVgpr, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSanResourcePlan, PrefersDeadWindowInsideCurrentAllocation) {
  RegisterSet live;
  live.expand({RegClass::VGPR, 0, 4});
  const ConSanRegisterRequest request = vgpr_request(3, 8, 8);

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(plan.base, 4);
  EXPECT_EQ(plan.required_descriptor_count, 8);
}

TEST(ConSanResourcePlan, GrowsAboveGuestReferencesAfterDeadSearchFails) {
  RegisterSet live;
  live.expand({RegClass::VGPR, 0, 8});
  const ConSanRegisterRequest request = vgpr_request(3, 8, 8);

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.base, 8);
  EXPECT_EQ(plan.required_descriptor_count, 11);
}

TEST(ConSanResourcePlan, GrowsPastReservedTailWindow) {
  RegisterSet live;
  live.expand({RegClass::VGPR, 0, 8});
  ConSanRegisterRequest request = vgpr_request(3, 8, 8);
  request.forbidden.expand({RegClass::VGPR, 8, 2});

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.base, 10);
  EXPECT_EQ(plan.required_descriptor_count, 13);
}

TEST(ConSanResourcePlan, GrowsIntoDeadWindowBelowHighGuestReference) {
  RegisterSet live;
  live.expand({RegClass::VGPR, 0, 8});
  live.expand({RegClass::VGPR, 18, 230});
  ConSanRegisterRequest request = vgpr_request(10, 8, 248);

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.base, 8);
  EXPECT_EQ(plan.required_descriptor_count, 18);
}

TEST(ConSanResourcePlan, SpillWindowCanGrowMixedOwnerDescriptors) {
  RegisterSet live;
  expand_all_vgprs(live);
  ConSanRegisterRequest request = vgpr_request(10, 8, 248);
  request.force_spill = true;
  request.allow_spill_descriptor_growth = true;
  request.forbidden.expand({RegClass::VGPR, 248, 2});

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.base, 0);
  EXPECT_EQ(plan.required_descriptor_count, 10);
}

TEST(ConSanResourcePlan, ForceSpillWithoutGrowthRejectsWindowAboveAllocation) {
  RegisterSet live;
  expand_all_vgprs(live);
  ConSanRegisterRequest request = vgpr_request(10, 8, 248);
  request.force_spill = true;
  request.forbidden.expand({RegClass::VGPR, 248, 2});

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::NoLegalWindow);
}

TEST(ConSanResourcePlan, FullRegisterFileSelectsAllowedLiveVictim) {
  RegisterSet live;
  expand_all_vgprs(live);
  ConSanRegisterRequest request = vgpr_request(3, 256, 256);
  request.forbidden.expand({RegClass::VGPR, 0, 2});

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.base, 2);
  EXPECT_EQ(plan.required_descriptor_count, 256);
}

TEST(ConSanResourcePlan, ForceSpillBypassesDeadAndGrowthWindows) {
  RegisterSet live;
  ConSanRegisterRequest request = vgpr_request(3, 8, 8);
  request.force_spill = true;
  request.forbidden.expand({RegClass::VGPR, 0, 1});

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.base, 1);
  EXPECT_EQ(plan.required_descriptor_count, 8);
}

TEST(ConSanResourcePlan, ExplicitOverrideCannotClobberLiveGuestValue) {
  RegisterSet live;
  live.expand({RegClass::VGPR, 4, 1});
  ConSanRegisterRequest request = vgpr_request(1, 8, 8);
  request.explicit_base = 4;

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::ExplicitLive);
}

TEST(ConSanResourcePlan, ExplicitFreshOverrideCarriesDescriptorRequirement) {
  RegisterSet live;
  ConSanRegisterRequest request = vgpr_request(3, 8, 8);
  request.alignment = 2;
  request.explicit_base = 10;

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::Explicit);
  EXPECT_EQ(plan.base, 10);
  EXPECT_EQ(plan.required_descriptor_count, 13);
}

TEST(ConSanResourcePlan, ForbiddenFullFileHasTypedFailure) {
  RegisterSet live;
  expand_all_vgprs(live);
  ConSanRegisterRequest request = vgpr_request(3, 256, 256);
  expand_all_vgprs(request.forbidden);

  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::NoLegalWindow);
}

} // namespace
} // namespace rocjitsu
