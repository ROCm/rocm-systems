// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu {
namespace {

struct InlineReleaseSequenceTarget {
  rj_code_arch_t arch;
  std::string_view label;
  uint16_t release_transaction_vsrc;
  uint16_t token_transaction_vsrc;
};

[[nodiscard]] std::vector<uint8_t>
make_inline_release_sequence_fixture(const InlineReleaseSequenceTarget &target,
                                     std::vector<uint32_t> &guest_atomic_words) {
  const auto atomic = instrumentation::build_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/0, /*return_old_value=*/false,
      /*scope=*/2, target.arch);
  if (!atomic)
    return {};
  guest_atomic_words = *atomic;

  std::vector<uint32_t> text_words;
  if (target.arch == ROCJITSU_CODE_ARCH_CDNA3) {
    const auto release = cdna3::build_mubuf(cdna3::kBufferWbl2Mubuf, {.sc1 = 1});
    const auto wait = build_cdna3_s_wait_vmcnt_lgkmcnt0(target.arch);
    if (!wait)
      return {};
    text_words.insert(text_words.end(), release.begin(), release.end());
    text_words.push_back(*wait);
  } else if (target.arch == ROCJITSU_CODE_ARCH_CDNA4) {
    const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
    const auto wait = build_cdna4_s_wait_flat0(target.arch);
    if (!wait)
      return {};
    text_words.insert(text_words.end(), release.begin(), release.end());
    text_words.push_back(*wait);
  } else {
    // global_wb is the target-native release member paired with the gfx12
    // FLAT atomic. Both admitted gfx12 decoders use this three-dword form.
    text_words.insert(text_words.end(), {0xEE0B0000u, 0x00000000u, 0x00000000u});
  }
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  text_words.resize(800, build_s_nop(0, target.arch));
  text_words.push_back(build_s_endpgm(target.arch));

  switch (target.arch) {
  case ROCJITSU_CODE_ARCH_CDNA3:
    return make_cdna3_lds_code_object(text_words, "atomic_release_sequence");
  case ROCJITSU_CODE_ARCH_CDNA4:
    return make_cdna4_lds_code_object(text_words, "atomic_release_sequence");
  case ROCJITSU_CODE_ARCH_RDNA4:
    return make_rdna4_lds_code_object(text_words, "atomic_release_sequence");
  case ROCJITSU_CODE_ARCH_GFX1250:
    return make_gfx1250_code_object(text_words, "atomic_release_sequence");
  default:
    return {};
  }
}

TEST(ConSanMoi, InlineShadowSkipsUnusedAtomicTransactionScratch) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_FALSE(result.resource_plans.empty());
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind != ConSanResourceSiteKind::Access || plan.scratch_vgpr_count <= 17u;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no relevant atomic sites") != std::string::npos;
  }));
}

TEST(ConSanMoi, Cdna4InlineAtomicAcquireReleaseEmitsNativeTransaction) {
  const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto acquire = cdna4::build_mubuf(cdna4::kBufferInvMubuf, {.sc1 = 1});
  const auto atomic = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/5, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait = build_cdna4_s_wait_flat0(ROCJITSU_CODE_ARCH_CDNA4);
  const auto instrumentation_wait =
      instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic && wait && instrumentation_wait);
  std::vector<uint32_t> text_words;
  text_words.insert(text_words.end(), release.begin(), release.end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), acquire.begin(), acquire.end());
  text_words.resize(800, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "atomic_acquire_release");
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors)
                               << " dispositions="
                               << testing::PrintToString(result.site_dispositions)
                               << " resources=" << testing::PrintToString(result.resource_plans);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            1u);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_EQ(*patch->scratch_vgpr, 8u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto release_claim_and_commit = build_cdna4_flat_atomic_cmpswap_b32(
      /*vaddr=*/8, /*vsrc=*/12, /*vdst=*/12, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto acquired_token_transaction = build_cdna4_flat_atomic_cmpswap_b32(
      /*vaddr=*/8, /*vsrc=*/26, /*vdst=*/26, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(release_claim_and_commit && acquired_token_transaction);
  EXPECT_TRUE(contains_subsequence(cave_words, *atomic));
  EXPECT_EQ(count_subsequence(cave_words, *release_claim_and_commit), 2u);
  EXPECT_EQ(count_subsequence(cave_words, *acquired_token_transaction), 15u);
  const auto expect_unified_atomic_wait = [&](const auto &transaction) {
    std::vector<uint32_t> drained(transaction.begin(), transaction.end());
    drained.push_back(*instrumentation_wait);
    EXPECT_EQ(count_subsequence(cave_words, drained), count_subsequence(cave_words, transaction));
    drained.push_back(*instrumentation_wait);
    EXPECT_EQ(count_subsequence(cave_words, drained), 0u)
        << "CDNA4 uses one unified FLAT wait, not a second store-side wait";
  };
  expect_unified_atomic_wait(*release_claim_and_commit);
  expect_unified_atomic_wait(*acquired_token_transaction);
}

TEST(ConSanMoi, SupportedTargetsInlineAtomicReleaseCarriesClaimedPredecessor) {
  constexpr std::array targets = {
      InlineReleaseSequenceTarget{ROCJITSU_CODE_ARCH_CDNA3, "gfx942/cdna3",
                                  /*release_transaction_vsrc=*/12,
                                  /*token_transaction_vsrc=*/26},
      InlineReleaseSequenceTarget{ROCJITSU_CODE_ARCH_CDNA4, "gfx950/cdna4",
                                  /*release_transaction_vsrc=*/12,
                                  /*token_transaction_vsrc=*/26},
      InlineReleaseSequenceTarget{ROCJITSU_CODE_ARCH_RDNA4, "gfx1201/rdna4",
                                  /*release_transaction_vsrc=*/13,
                                  /*token_transaction_vsrc=*/27},
      InlineReleaseSequenceTarget{ROCJITSU_CODE_ARCH_GFX1250, "gfx1250",
                                  /*release_transaction_vsrc=*/13,
                                  /*token_transaction_vsrc=*/27},
  };

  for (const InlineReleaseSequenceTarget &target : targets) {
    SCOPED_TRACE(target.label);
    std::vector<uint32_t> atomic_words;
    const std::vector<uint8_t> bytes = make_inline_release_sequence_fixture(target, atomic_words);
    ASSERT_FALSE(bytes.empty());
    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_track_atomics = true;
    options.scratch_vgpr = 8;
    options.moi_exec_save_sgpr = 80;
    options.moi_owner_vgpr = 40;
    options.moi_epoch_vgpr = 41;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
    options.max_patches = 1;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                                 << " errors=" << testing::PrintToString(result.errors)
                                 << " dispositions="
                                 << testing::PrintToString(result.site_dispositions)
                                 << " resources=" << testing::PrintToString(result.resource_plans);
    EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
    EXPECT_TRUE(result.final_validation_passed);
    const auto patch = std::ranges::find(
        result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end());

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    const std::vector<uint32_t> cave_words =
        text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
    const auto release_claim_and_commit = instrumentation::build_flat_atomic_cmpswap_b32(
        /*vaddr=*/8, target.release_transaction_vsrc, target.release_transaction_vsrc,
        /*return_old_value=*/true, /*scope=*/2, target.arch);
    const auto acquired_token_transaction = instrumentation::build_flat_atomic_cmpswap_b32(
        /*vaddr=*/8, target.token_transaction_vsrc, target.token_transaction_vsrc,
        /*return_old_value=*/true, /*scope=*/2, target.arch);
    ASSERT_TRUE(release_claim_and_commit && acquired_token_transaction);

    EXPECT_EQ(count_subsequence(cave_words, atomic_words), 1u);
    EXPECT_EQ(count_subsequence(cave_words, *release_claim_and_commit), 2u);
    // Five direct/inherited token slots each reserve, roll back if needed,
    // then commit. These 15 CAS sites exist only on the predecessor-import path.
    EXPECT_EQ(count_subsequence(cave_words, *acquired_token_transaction), 15u);
    const auto advance_consumer_segment =
        instrumentation::build_v_add_u32(*options.moi_epoch_vgpr, scalar_positive_inline_u32(1),
                                         *options.moi_epoch_vgpr, target.arch);
    ASSERT_TRUE(advance_consumer_segment);
    EXPECT_EQ(count_subsequence(cave_words, *advance_consumer_segment), 0u)
        << "release-sequence inheritance must not advance the consumer segment";
  }
}

TEST(ConSanMoi, SharedHelperInlineAtomicSpillUsesAutomaticStateAcrossOwners) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.helper_atomic_acquire_release = true;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  ASSERT_TRUE(result.resolved_moi_workgroup_key_vgpr);
  EXPECT_EQ(*result.resolved_moi_workgroup_key_vgpr, *result.resolved_moi_epoch_vgpr + 1u);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan->scratch_vgpr_count, 26u);
  ASSERT_EQ(plan->owner_descriptor_file_offsets.size(), 2u);
  const auto atomic_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
  });
  ASSERT_NE(atomic_patch, result.patches.end());
  EXPECT_EQ(atomic_patch->spilled_vgpr_count, 26u);
  EXPECT_EQ(atomic_patch->required_private_segment_size, 136u);
  EXPECT_EQ(atomic_patch->owner_descriptor_file_offsets, plan->owner_descriptor_file_offsets);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, atomic_patch->trampoline_offset, atomic_patch->trampoline_size);
  ASSERT_TRUE(atomic_patch->scratch_vgpr);
  const uint16_t scratch_vgpr = *atomic_patch->scratch_vgpr;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  const auto acquire_version = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      scratch_vgpr, value_vgpr, value_vgpr, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto acquire_wait = instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(acquire_version);
  ASSERT_TRUE(acquire_wait);
  std::vector<uint32_t> acquire_valid = {
      build_v_mov_b32_e32(value_vgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4)};
  acquire_valid.insert(acquire_valid.end(), acquire_version->begin(), acquire_version->end());
  acquire_valid.push_back(*acquire_wait);
  const auto release_claim = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      scratch_vgpr, static_cast<uint16_t>(scratch_vgpr + 5u),
      static_cast<uint16_t>(scratch_vgpr + 5u), /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(release_claim);
  const auto acquire_position =
      std::search(cave_words.begin(), cave_words.end(), acquire_valid.begin(), acquire_valid.end());
  const auto release_position = std::search(cave_words.begin(), cave_words.end(),
                                            release_claim->begin(), release_claim->end());
  ASSERT_NE(acquire_position, cave_words.end());
  ASSERT_NE(release_position, cave_words.end());
  EXPECT_EQ(count_subsequence(cave_words, *release_claim), 2u)
      << "RDNA4 acquire-release must claim and commit one serialized transaction";
  EXPECT_LT(acquire_position, release_position)
      << "acquire-release must import the prior handoff before replacing it";
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                          }),
            2);
}

TEST(ConSanMoi, GenerationTaggedLocalAtomicLookupUsesPersistentWorkgroupKey) {
  std::vector<uint8_t> bytes = make_rdna4_lds_and_ordered_flat_atomic_handoff_code_object();
  ASSERT_FALSE(bytes.empty());
  mutate_first_kernel_descriptor(
      bytes, [](KD &descriptor) { descriptor.group_segment_fixed_size = 1024u; });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_inline_workgroup_shadow = true;
  options.scratch_vgpr = 16;
  options.moi_owner_vgpr = 48;
  options.moi_epoch_vgpr = 49;
  options.moi_workgroup_key_vgpr = 50;
  options.moi_exec_save_sgpr = 40;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_report_generation = 2u;
  options.max_patches = 16;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto load_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore &&
           patch.anchor_offset == 14u * sizeof(uint32_t);
  });
  ASSERT_NE(load_patch, result.patches.end());
  EXPECT_GT(load_patch->workgroup_shadow_size, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, load_patch->trampoline_offset, load_patch->trampoline_size);
  const uint32_t materialize_workgroup_key = build_v_mov_b32_e32(
      static_cast<uint16_t>(*options.scratch_vgpr + 19u),
      vector_source_vgpr(*options.moi_workgroup_key_vgpr), ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_NE(std::ranges::find(words, materialize_workgroup_key), words.end())
      << "local-shadow atomic tokens are keyed by workgroup, not report generation";
}

TEST(ConSanMoi, InlineAtomicReleaseHashUsesFullAddressAndPowerOfTwoCapacity) {
  EXPECT_EQ(consan_moi_inline_atomic_release_slot_index(0x0000000000000100ull, 64), 44u);
  EXPECT_EQ(consan_moi_inline_atomic_release_slot_index(0x0000000000000104ull, 64), 23u);
  EXPECT_EQ(consan_moi_inline_atomic_release_slot_index(0x0000000100000100ull, 64), 23u);
  EXPECT_EQ(consan_moi_inline_atomic_release_slot_index(0x0000003f00000100ull, 64), 40u);
  EXPECT_NE(consan_moi_inline_atomic_release_slot_index(0x0000000000000100ull, 64),
            consan_moi_inline_atomic_release_slot_index(0x0000000000000200ull, 64));
  EXPECT_EQ(consan_moi_inline_atomic_release_slot_index(0x1234567800000100ull, 1), 0u);
  EXPECT_EQ(consan_moi_inline_atomic_release_slot_index(0x1234567800000100ull, 0), 0u);
  EXPECT_EQ(consan_moi_inline_atomic_release_slot_index(0x1234567800000100ull, 48), 0u);
}

TEST(ConSanMoi, InlineAcquiredTokenHashSeparatesAuthorizationAndReleaseSequenceNamespaces) {
  constexpr uint32_t capacity = 64;
  constexpr uint32_t workgroup = 31;
  constexpr uint32_t consumer = 7;
  constexpr uint32_t producer = 3;
  const uint32_t authorization = consan_moi_inline_acquired_epoch_token_slot_index(
      workgroup, consumer, producer, capacity, /*release_sequence=*/false);
  const uint32_t release_sequence = consan_moi_inline_acquired_epoch_token_slot_index(
      workgroup, consumer, producer, capacity, /*release_sequence=*/true);
  EXPECT_LT(authorization, capacity);
  EXPECT_LT(release_sequence, capacity);
  EXPECT_NE(authorization, release_sequence);
}

TEST(ConSanMoi, InlineAcquiredTokenHashDoesNotAliasDistinctCausalEdgesByConstruction) {
  constexpr uint32_t capacity = 64;
  constexpr uint32_t workgroup = 1;
  // The former XOR/shift combiner mapped both of these release-sequence
  // edges to slot 63. They occur naturally when waves acquire an atomic in
  // the order 2 -> 3 -> 1.
  const uint32_t one_after_three = consan_moi_inline_acquired_epoch_token_slot_index(
      workgroup, /*consumer_owner_id=*/1, /*producer_owner_id=*/3, capacity,
      /*release_sequence=*/true);
  const uint32_t three_after_two = consan_moi_inline_acquired_epoch_token_slot_index(
      workgroup, /*consumer_owner_id=*/3, /*producer_owner_id=*/2, capacity,
      /*release_sequence=*/true);
  EXPECT_NE(one_after_three, three_after_two);

  // Reversing an edge must not be an algebraic alias. Direct mapping can
  // still have ordinary finite-table collisions, but the identity combiner
  // must preserve field roles before the final capacity mask.
  for (uint32_t consumer = 1; consumer <= 4; ++consumer) {
    for (uint32_t producer = consumer + 1; producer <= 4; ++producer) {
      EXPECT_NE(consan_moi_inline_acquired_epoch_token_slot_index(workgroup, consumer, producer,
                                                                  capacity),
                consan_moi_inline_acquired_epoch_token_slot_index(workgroup, producer, consumer,
                                                                  capacity));
    }
  }
}

TEST(ConSanMoi, InlineAtomicDirectMappedTablePinsLookupAndReplacementContract) {
  constexpr uint32_t capacity = 4;
  std::array<ConSanMoiInlineAtomicReleaseSlot, capacity> table{};
  constexpr uint64_t address_a = 0x1234567800000100ull;
  constexpr uint64_t address_b = address_a + capacity * sizeof(uint32_t);
  static_assert(consan_moi_inline_atomic_release_slot_index(address_a, capacity) ==
                consan_moi_inline_atomic_release_slot_index(address_b, capacity));
  const uint32_t slot_index = consan_moi_inline_atomic_release_slot_index(address_a, capacity);

  EXPECT_EQ(consan_moi_inline_atomic_release_lookup(table[slot_index], address_a, 7),
            ConSanMoiInlineAtomicLookup::Empty);

  table[slot_index] = {/*version=*/2, /*owner_id=*/3, /*epoch_plus_one=*/11,
                       /*workgroup_key=*/0, address_a};
  EXPECT_EQ(consan_moi_inline_atomic_release_lookup(table[slot_index], address_a, 3),
            ConSanMoiInlineAtomicLookup::ExactSameOwner);
  EXPECT_EQ(consan_moi_inline_atomic_release_lookup(table[slot_index], address_a, 7),
            ConSanMoiInlineAtomicLookup::ExactOtherOwner);
  EXPECT_EQ(consan_moi_inline_atomic_release_lookup(table[slot_index], address_b, 7),
            ConSanMoiInlineAtomicLookup::Collision);

  table[slot_index] = {/*version=*/2, /*owner_id=*/5, /*epoch_plus_one=*/13,
                       /*workgroup_key=*/0, address_b};
  EXPECT_EQ(consan_moi_inline_atomic_release_lookup(table[slot_index], address_a, 7),
            ConSanMoiInlineAtomicLookup::Collision);
  EXPECT_EQ(consan_moi_inline_atomic_release_lookup(table[slot_index], address_b, 7),
            ConSanMoiInlineAtomicLookup::ExactOtherOwner);
}

TEST(ConSanMoi, InlineAtomicDirectMappedTableUsesEverySlotAtCapacity) {
  constexpr uint32_t capacity = 4;
  constexpr uint64_t base = 0x1234567800000100ull;
  std::array<ConSanMoiInlineAtomicReleaseSlot, capacity> table{};
  for (uint32_t i = 0; i < capacity; ++i) {
    const uint64_t address = base + i * sizeof(uint32_t);
    const uint32_t slot_index = consan_moi_inline_atomic_release_slot_index(address, capacity);
    table[slot_index] = {/*version=*/2, /*owner_id=*/i + 1u, /*epoch_plus_one=*/i + 10u,
                         /*workgroup_key=*/0, address};
  }
  for (uint32_t i = 0; i < capacity; ++i) {
    const uint64_t address = base + i * sizeof(uint32_t);
    const uint32_t slot_index = consan_moi_inline_atomic_release_slot_index(address, capacity);
    EXPECT_EQ(consan_moi_inline_atomic_release_lookup(table[slot_index], address, 99),
              ConSanMoiInlineAtomicLookup::ExactOtherOwner);
  }
}

TEST(ConSanMoi, InlineCausalSnapshotAbiCapturesCanonicalBoundedFrontier) {
  static_assert(sizeof(ConSanMoiInlineCausalSnapshotEntry) == 8);
  static_assert(sizeof(ConSanMoiInlineCausalSnapshot) == 40);
  static_assert(alignof(ConSanMoiInlineCausalSnapshot) == 8);
  constexpr uint64_t dispatch = 0xD150000000000001ull;
  constexpr uint32_t workgroup = 19;
  constexpr uint32_t releaser = 7;
  constexpr auto token = [](uint64_t token_dispatch, uint32_t token_workgroup, uint32_t consumer,
                            uint32_t producer, uint32_t epoch_plus_one, uint32_t version = 2u,
                            ConSanMoiInlineTokenEvidenceKind kind =
                                ConSanMoiInlineTokenEvidenceKind::Direct) {
    return ConSanMoiInlineCausalTokenView{
        .version_before = version,
        .version_after = version,
        .dispatch_id = token_dispatch,
        .workgroup_key = token_workgroup,
        .consumer_owner_id = consumer,
        .producer_owner_id = producer,
        .producer_epoch_plus_one = epoch_plus_one,
        .kind = kind,
        .source_release_address = 0x4000,
        .source_release_version = 2,
        .consumer_epoch_plus_one = 1,
    };
  };
  constexpr std::array tokens = {
      token(dispatch, workgroup, releaser, 5, 13, 2, ConSanMoiInlineTokenEvidenceKind::Inherited),
      token(dispatch, workgroup, releaser, 3, 11),
      token(dispatch + 1u, workgroup, releaser, 2, 9),
      token(dispatch, workgroup, releaser + 1u, 2, 9),
      ConSanMoiInlineCausalTokenView{},
  };
  const auto snapshot =
      consan_moi_inline_capture_causal_snapshot(tokens, dispatch, workgroup, releaser);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(snapshot, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Usable);
  ASSERT_EQ(snapshot.entry_count, 2u);
  EXPECT_EQ(snapshot.entries[0], (ConSanMoiInlineCausalSnapshotEntry{3, 11}));
  EXPECT_EQ(snapshot.entries[1], (ConSanMoiInlineCausalSnapshotEntry{5, 13}));
  EXPECT_EQ(snapshot.entries[2], ConSanMoiInlineCausalSnapshotEntry{});

  std::array<ConSanMoiInlineCausalTokenView, 5> fan_in{};
  for (uint32_t i = 0; i < fan_in.size(); ++i)
    fan_in[i] = token(dispatch, workgroup, releaser, i + 1u, i + 2u);
  const auto overflow =
      consan_moi_inline_capture_causal_snapshot(fan_in, dispatch, workgroup, releaser);
  EXPECT_EQ(overflow.entry_count, kConSanMoiInlineCausalSnapshotEntryCapacity);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(overflow, releaser),
            ConSanMoiInlineCausalSnapshotStatus::CapacityOverflow);

  const auto incomplete =
      consan_moi_inline_capture_causal_snapshot(tokens, dispatch, workgroup, releaser,
                                                /*source_complete=*/false);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(incomplete, releaser),
            ConSanMoiInlineCausalSnapshotStatus::SourceIncomplete);
}

TEST(ConSanMoi, InlineCausalSnapshotRejectsMalformedDuplicateCycleAndHiddenState) {
  constexpr uint64_t dispatch = 0xD150000000000001ull;
  constexpr uint32_t workgroup = 19;
  constexpr uint32_t releaser = 7;
  constexpr auto token = [=](uint32_t producer, uint32_t epoch_plus_one, uint32_t version = 2u) {
    return ConSanMoiInlineCausalTokenView{
        .version_before = version,
        .version_after = version,
        .dispatch_id = dispatch,
        .workgroup_key = workgroup,
        .consumer_owner_id = releaser,
        .producer_owner_id = producer,
        .producer_epoch_plus_one = epoch_plus_one,
        .kind = ConSanMoiInlineTokenEvidenceKind::Direct,
        .source_release_address = 0x4000,
        .source_release_version = 2,
        .consumer_epoch_plus_one = 1,
    };
  };
  constexpr std::array duplicate = {token(3, 11), token(3, 12)};
  auto snapshot =
      consan_moi_inline_capture_causal_snapshot(duplicate, dispatch, workgroup, releaser);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(snapshot, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);

  constexpr std::array self_cycle = {token(releaser, 11)};
  snapshot = consan_moi_inline_capture_causal_snapshot(self_cycle, dispatch, workgroup, releaser);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(snapshot, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);

  constexpr std::array publishing_source = {token(3, 11, /*version=*/3)};
  snapshot =
      consan_moi_inline_capture_causal_snapshot(publishing_source, dispatch, workgroup, releaser);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(snapshot, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);

  auto changed_source = token(3, 11);
  changed_source.version_after = 4;
  snapshot = consan_moi_inline_capture_causal_snapshot(
      std::span<const ConSanMoiInlineCausalTokenView>(&changed_source, 1), dispatch, workgroup,
      releaser);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(snapshot, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);
  auto unbound_source = token(3, 11);
  unbound_source.source_release_address = 0;
  snapshot = consan_moi_inline_capture_causal_snapshot(
      std::span<const ConSanMoiInlineCausalTokenView>(&unbound_source, 1), dispatch, workgroup,
      releaser);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(snapshot, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);
  auto zero_version_with_payload = token(3, 11);
  zero_version_with_payload.version_before = zero_version_with_payload.version_after = 0;
  auto unknown_kind = token(3, 11);
  unknown_kind.kind = static_cast<ConSanMoiInlineTokenEvidenceKind>(99);
  auto zero_source_version = token(3, 11);
  zero_source_version.source_release_version = 0;
  auto odd_source_version = token(3, 11);
  odd_source_version.source_release_version = 3;
  auto reserved_source = token(3, 11);
  reserved_source.reservation_version = 1;
  for (const auto &bad_source : {zero_version_with_payload, unknown_kind, zero_source_version,
                                 odd_source_version, reserved_source}) {
    snapshot = consan_moi_inline_capture_causal_snapshot(
        std::span<const ConSanMoiInlineCausalTokenView>(&bad_source, 1), dispatch, workgroup,
        releaser);
    EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(snapshot, releaser),
              ConSanMoiInlineCausalSnapshotStatus::Malformed);
  }
  ConSanMoiInlineCausalTokenView hidden_empty_source;
  hidden_empty_source.kind = ConSanMoiInlineTokenEvidenceKind::Inherited;
  snapshot = consan_moi_inline_capture_causal_snapshot(
      std::span<const ConSanMoiInlineCausalTokenView>(&hidden_empty_source, 1), dispatch, workgroup,
      releaser);
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(snapshot, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);

  ConSanMoiInlineCausalSnapshot boundary;
  boundary.entry_count = 1;
  boundary.entries[0] = {/*ancestor_owner_id=*/3,
                         /*ancestor_epoch_plus_one=*/1023};
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(boundary, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Usable);
  boundary.entries[0].ancestor_epoch_plus_one = 1024;
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(boundary, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);
  boundary.entries[0].ancestor_epoch_plus_one = 1023;
  boundary.flags = 1u << 31u;
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(boundary, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);
  boundary.flags = 0;
  boundary.entry_count = kConSanMoiInlineCausalSnapshotEntryCapacity + 1u;
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(boundary, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);
  boundary.entry_count = 1;
  boundary.entries[2] = {/*ancestor_owner_id=*/9,
                         /*ancestor_epoch_plus_one=*/1};
  EXPECT_EQ(consan_moi_inline_validate_causal_snapshot(boundary, releaser),
            ConSanMoiInlineCausalSnapshotStatus::Malformed);
}

TEST(ConSanMoi, InlineCausalImportClosesNoMiddleAccessAndTwoHopChains) {
  constexpr uint64_t dispatch = 0xD150000000000001ull;
  constexpr uint32_t workgroup = 19;
  constexpr ConSanMoiInlineVersionedReleaseIdentity p_identity{dispatch, 0x4000, workgroup};
  constexpr auto token = [=](uint32_t consumer, uint32_t producer, uint32_t epoch_plus_one,
                             uint64_t source_address, uint32_t source_version) {
    return ConSanMoiInlineCausalTokenView{
        .version_before = 2,
        .version_after = 2,
        .dispatch_id = dispatch,
        .workgroup_key = workgroup,
        .consumer_owner_id = consumer,
        .producer_owner_id = producer,
        .producer_epoch_plus_one = epoch_plus_one,
        .kind = ConSanMoiInlineTokenEvidenceKind::Direct,
        .source_release_address = source_address,
        .source_release_version = source_version,
        .consumer_epoch_plus_one = 1,
    };
  };
  constexpr ConSanMoiInlineStableReleaseSnapshot p_release{/*version_before=*/2,
                                                           /*version_after=*/2,
                                                           p_identity,
                                                           /*releaser_owner_id=*/3,
                                                           /*releaser_epoch_plus_one=*/12,
                                                           /*snapshot=*/{}};
  const auto middle_import =
      consan_moi_inline_plan_causal_import(p_release, p_identity, /*consumer_owner_id=*/7);
  ASSERT_TRUE(middle_import.authoritative());
  ASSERT_EQ(middle_import.entry_count, 1u);
  EXPECT_EQ(middle_import.entries[0].producer_owner_id, 3u);

  // M need not touch the protected LDS payload. Its acquired token alone is
  // snapshotted at M's release and supplies the missing P -> M -> C closure.
  const std::array middle_tokens = {token(/*consumer=*/7,
                                          middle_import.entries[0].producer_owner_id,
                                          middle_import.entries[0].producer_epoch_plus_one,
                                          p_identity.atomic_address, p_release.version_after)};
  const auto middle_snapshot =
      consan_moi_inline_capture_causal_snapshot(middle_tokens, dispatch, workgroup,
                                                /*releaser_owner_id=*/7);
  const ConSanMoiInlineVersionedReleaseIdentity m_identity{dispatch, 0x5000, workgroup};
  const ConSanMoiInlineStableReleaseSnapshot m_release{/*version_before=*/4,
                                                       /*version_after=*/4,
                                                       m_identity,
                                                       /*releaser_owner_id=*/7,
                                                       /*releaser_epoch_plus_one=*/20,
                                                       middle_snapshot};
  const auto consumer_import =
      consan_moi_inline_plan_causal_import(m_release, m_identity, /*consumer_owner_id=*/9);
  ASSERT_TRUE(consumer_import.authoritative());
  ASSERT_EQ(consumer_import.entry_count, 2u);
  EXPECT_EQ(consumer_import.entries[0].producer_owner_id, 7u);
  EXPECT_EQ(consumer_import.entries[1].producer_owner_id, 3u);
  EXPECT_EQ(consumer_import.entries[1].producer_epoch_plus_one, 12u);

  const std::array next_tokens = {
      token(/*consumer=*/9, consumer_import.entries[0].producer_owner_id,
            consumer_import.entries[0].producer_epoch_plus_one, m_identity.atomic_address,
            m_release.version_after),
      token(/*consumer=*/9, consumer_import.entries[1].producer_owner_id,
            consumer_import.entries[1].producer_epoch_plus_one, m_identity.atomic_address,
            m_release.version_after),
  };
  const auto next_snapshot = consan_moi_inline_capture_causal_snapshot(
      next_tokens, dispatch, workgroup, /*releaser_owner_id=*/9);
  const ConSanMoiInlineVersionedReleaseIdentity n_identity{dispatch, 0x6000, workgroup};
  const ConSanMoiInlineStableReleaseSnapshot n_release{/*version_before=*/6,
                                                       /*version_after=*/6,
                                                       n_identity,
                                                       /*releaser_owner_id=*/9,
                                                       /*releaser_epoch_plus_one=*/30,
                                                       next_snapshot};
  const auto two_hop =
      consan_moi_inline_plan_causal_import(n_release, n_identity, /*consumer_owner_id=*/11);
  ASSERT_TRUE(two_hop.authoritative());
  ASSERT_EQ(two_hop.entry_count, 3u);
  EXPECT_EQ(two_hop.entries[0].producer_owner_id, 9u);
  EXPECT_EQ(two_hop.entries[1].producer_owner_id, 3u);
  EXPECT_EQ(two_hop.entries[2].producer_owner_id, 7u);
}

TEST(ConSanMoi, InlineCausalImportUsesImmutableReleaseTimeSnapshotAndFailsClosed) {
  constexpr uint64_t dispatch = 0xD150000000000001ull;
  constexpr uint32_t workgroup = 19;
  constexpr uint32_t releaser = 7;
  std::array tokens = {ConSanMoiInlineCausalTokenView{
      .version_before = 2,
      .version_after = 2,
      .dispatch_id = dispatch,
      .workgroup_key = workgroup,
      .consumer_owner_id = releaser,
      .producer_owner_id = 3,
      .producer_epoch_plus_one = 12,
      .kind = ConSanMoiInlineTokenEvidenceKind::Direct,
      .source_release_address = 0x4000,
      .source_release_version = 2,
      .consumer_epoch_plus_one = 1,
  }};
  const auto before =
      consan_moi_inline_capture_causal_snapshot(tokens, dispatch, workgroup, releaser);
  tokens[0].producer_epoch_plus_one = 22;
  const auto after =
      consan_moi_inline_capture_causal_snapshot(tokens, dispatch, workgroup, releaser);
  EXPECT_EQ(before.entries[0].ancestor_epoch_plus_one, 12u);
  EXPECT_EQ(after.entries[0].ancestor_epoch_plus_one, 22u);

  const ConSanMoiInlineVersionedReleaseIdentity identity{dispatch, 0x4000, workgroup};
  ConSanMoiInlineStableReleaseSnapshot release{/*version_before=*/2,
                                               /*version_after=*/2,
                                               identity,
                                               /*releaser_owner_id=*/releaser,
                                               /*releaser_epoch_plus_one=*/30,
                                               before};
  auto plan = consan_moi_inline_plan_causal_import(release, identity, /*consumer_owner_id=*/9);
  ASSERT_TRUE(plan.authoritative());
  ASSERT_EQ(plan.entry_count, 2u);
  EXPECT_EQ(plan.entries[1].producer_epoch_plus_one, 12u);

  release.version_after = 4;
  EXPECT_EQ(consan_moi_inline_plan_causal_import(release, identity, 9).status,
            ConSanMoiInlineCausalImportStatus::UnstableRelease);
  release.version_before = release.version_after = 3;
  EXPECT_EQ(consan_moi_inline_plan_causal_import(release, identity, 9).status,
            ConSanMoiInlineCausalImportStatus::UnstableRelease);
  release.version_before = release.version_after = 2;
  auto other_dispatch = identity;
  other_dispatch.dispatch_id += 1u;
  EXPECT_EQ(consan_moi_inline_plan_causal_import(release, other_dispatch, 9).status,
            ConSanMoiInlineCausalImportStatus::IdentityMismatch);
  auto other_cell = identity;
  other_cell.atomic_address += 4u;
  EXPECT_EQ(consan_moi_inline_plan_causal_import(release, other_cell, 9).status,
            ConSanMoiInlineCausalImportStatus::IdentityMismatch);

  // Two producers may publish independent snapshots in two cells, but an
  // acquire selected for one address cannot import the other's ancestry.
  auto second_release = release;
  second_release.identity = other_cell;
  second_release.releaser_owner_id = 11;
  second_release.releaser_epoch_plus_one = 41;
  second_release.snapshot = after;
  EXPECT_TRUE(consan_moi_inline_plan_causal_import(second_release, other_cell, 13).authoritative());
  EXPECT_EQ(consan_moi_inline_plan_causal_import(second_release, identity, 13).status,
            ConSanMoiInlineCausalImportStatus::IdentityMismatch);
  EXPECT_EQ(
      consan_moi_inline_plan_causal_import(release, identity, 9, /*destination_collision=*/true)
          .status,
      ConSanMoiInlineCausalImportStatus::DestinationCollision);
  EXPECT_EQ(consan_moi_inline_plan_causal_import(release, identity, 9,
                                                 /*destination_collision=*/false,
                                                 /*destination_capacity_exhausted=*/true)
                .status,
            ConSanMoiInlineCausalImportStatus::DestinationCapacityExhausted);

  release.snapshot.entries[0].ancestor_owner_id = 9;
  EXPECT_EQ(consan_moi_inline_plan_causal_import(release, identity, 9).status,
            ConSanMoiInlineCausalImportStatus::MalformedSnapshot);
  release.snapshot = before;
  release.snapshot.flags =
      consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::CapacityOverflow);
  EXPECT_EQ(consan_moi_inline_plan_causal_import(release, identity, 9).status,
            ConSanMoiInlineCausalImportStatus::CapacityOverflow);
}

TEST(ConSanMoi, InlineTokenEvidenceAcceptsOnlyTwoSidedOrderedTransitiveProof) {
  constexpr uint64_t dispatch = 0xD150000000000001ull;
  constexpr uint32_t workgroup = 19;
  constexpr ConSanMoiInlineVersionedReleaseIdentity identity{dispatch, 0x5000, workgroup};
  ConSanMoiInlineCausalSnapshot snapshot;
  snapshot.entry_count = 1;
  snapshot.entries[0] = {/*ancestor_owner_id=*/3,
                         /*ancestor_epoch_plus_one=*/12};
  const std::array releases = {ConSanMoiInlineStableReleaseEvidence{
      identity, /*version_before=*/4, /*version_after=*/4,
      /*releaser_owner_id=*/7, /*releaser_epoch_plus_one=*/20, snapshot}};
  const std::array tokens = {
      ConSanMoiInlineTokenEvidence{dispatch, workgroup,
                                   /*consumer_owner_id=*/9,
                                   /*producer_owner_id=*/7,
                                   /*producer_epoch_plus_one=*/20,
                                   ConSanMoiInlineTokenEvidenceKind::Direct, identity,
                                   /*source_release_version=*/4},
      ConSanMoiInlineTokenEvidence{dispatch, workgroup,
                                   /*consumer_owner_id=*/9,
                                   /*producer_owner_id=*/3,
                                   /*producer_epoch_plus_one=*/12,
                                   ConSanMoiInlineTokenEvidenceKind::Inherited, identity,
                                   /*source_release_version=*/4},
  };
  const std::array accesses = {
      ConSanMoiInlineAccessEvidence{dispatch, workgroup, /*owner_id=*/7, /*access_count=*/2},
      ConSanMoiInlineAccessEvidence{dispatch, workgroup, /*owner_id=*/9, /*access_count=*/1},
  };
  constexpr ConSanMoiInlineQualificationExpectation expected{
      ConSanMoiInlineQualificationSemantics::Ordered,
      identity,
      /*producer_owner_id=*/7,
      /*producer_epoch_plus_one=*/20,
      /*consumer_owner_id=*/9,
      /*required_ancestor_owner_id=*/3,
      /*required_ancestor_epoch_plus_one=*/12};
  constexpr ConSanMoiInlineEvidenceCounters counters{dispatch,
                                                     /*dispatch_coverage_count=*/1,
                                                     /*undercoverage_count=*/0,
                                                     /*overflow_count=*/0,
                                                     /*unsupported_count=*/0,
                                                     /*consan_diagnostic_count=*/0,
                                                     /*matched_source_diagnostic_count=*/0};
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, tokens, accesses, counters)
          .accepted());

  auto missing_inherited = tokens;
  missing_inherited[1].producer_owner_id = 5;
  EXPECT_TRUE(consan_moi_inline_qualify_token_evidence(expected, releases, missing_inherited,
                                                       accesses, counters)
                  .has(ConSanMoiInlineQualificationFailure::InheritedToken));
  auto wrong_version = tokens;
  wrong_version[0].source_release_version = 2;
  EXPECT_TRUE(consan_moi_inline_qualify_token_evidence(expected, releases, wrong_version, accesses,
                                                       counters)
                  .has(ConSanMoiInlineQualificationFailure::DirectToken));
  std::array<ConSanMoiInlineTokenEvidence, 3> duplicate_direct = {tokens[0], tokens[1], tokens[0]};
  EXPECT_TRUE(consan_moi_inline_qualify_token_evidence(expected, releases, duplicate_direct,
                                                       accesses, counters)
                  .has(ConSanMoiInlineQualificationFailure::DirectToken));
  auto one_sided = accesses;
  one_sided[0].access_count = 0;
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, tokens, one_sided, counters)
          .has(ConSanMoiInlineQualificationFailure::ProducerAccess));
  one_sided = accesses;
  one_sided[1].dispatch_id += 1u;
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, tokens, one_sided, counters)
          .has(ConSanMoiInlineQualificationFailure::ConsumerAccess));
}

TEST(ConSanMoi, InlineTokenEvidenceRejectsCoverageAndMalformedProofFailures) {
  constexpr uint64_t dispatch = 0xD150000000000001ull;
  constexpr uint32_t workgroup = 19;
  constexpr ConSanMoiInlineVersionedReleaseIdentity identity{dispatch, 0x4000, workgroup};
  const std::array releases = {ConSanMoiInlineStableReleaseEvidence{
      identity, /*version_before=*/2, /*version_after=*/2,
      /*releaser_owner_id=*/3, /*releaser_epoch_plus_one=*/12, /*snapshot=*/{}}};
  const std::array tokens = {ConSanMoiInlineTokenEvidence{
      dispatch, workgroup, /*consumer_owner_id=*/7, /*producer_owner_id=*/3,
      /*producer_epoch_plus_one=*/12, ConSanMoiInlineTokenEvidenceKind::Direct, identity,
      /*source_release_version=*/2}};
  const std::array accesses = {
      ConSanMoiInlineAccessEvidence{dispatch, workgroup, /*owner_id=*/3, /*access_count=*/1},
      ConSanMoiInlineAccessEvidence{dispatch, workgroup, /*owner_id=*/7, /*access_count=*/1},
  };
  constexpr ConSanMoiInlineQualificationExpectation expected{
      ConSanMoiInlineQualificationSemantics::Ordered,
      identity,
      /*producer_owner_id=*/3,
      /*producer_epoch_plus_one=*/12,
      /*consumer_owner_id=*/7,
      /*required_ancestor_owner_id=*/0,
      /*required_ancestor_epoch_plus_one=*/0};
  ConSanMoiInlineEvidenceCounters counters{dispatch, 1, 0, 0, 0, 0, 0};
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, tokens, accesses, counters)
          .accepted());

  counters.dispatch_coverage_count = 0;
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, tokens, accesses, counters)
          .has(ConSanMoiInlineQualificationFailure::DispatchCoverage));
  counters = {dispatch, 1, /*undercoverage_count=*/1, 0, 0, 0, 0};
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, tokens, accesses, counters)
          .has(ConSanMoiInlineQualificationFailure::Undercoverage));
  counters = {dispatch, 1, 0, /*overflow_count=*/1, 0, 0, 0};
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, tokens, accesses, counters)
          .has(ConSanMoiInlineQualificationFailure::Overflow));
  counters = {dispatch, 1, 0, 0, /*unsupported_count=*/1, 0, 0};
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, tokens, accesses, counters)
          .has(ConSanMoiInlineQualificationFailure::Unsupported));
  counters = {dispatch, 1, 0, 0, 0, /*consan_diagnostic_count=*/1, 0};
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, tokens, accesses, counters)
          .has(ConSanMoiInlineQualificationFailure::UnexpectedDiagnostic));

  counters = {dispatch, 1, 0, 0, 0, 0, 0};
  auto unstable = releases;
  unstable[0].version_after = 4;
  const auto unstable_result =
      consan_moi_inline_qualify_token_evidence(expected, unstable, tokens, accesses, counters);
  EXPECT_TRUE(unstable_result.has(ConSanMoiInlineQualificationFailure::MalformedEvidence));
  EXPECT_TRUE(unstable_result.has(ConSanMoiInlineQualificationFailure::StableRelease));
  auto malformed_token = tokens;
  malformed_token[0].producer_epoch_plus_one = 0;
  EXPECT_TRUE(consan_moi_inline_qualify_token_evidence(expected, releases, malformed_token,
                                                       accesses, counters)
                  .has(ConSanMoiInlineQualificationFailure::MalformedEvidence));
  malformed_token = tokens;
  malformed_token[0].kind = static_cast<ConSanMoiInlineTokenEvidenceKind>(99);
  EXPECT_TRUE(consan_moi_inline_qualify_token_evidence(expected, releases, malformed_token,
                                                       accesses, counters)
                  .has(ConSanMoiInlineQualificationFailure::MalformedEvidence));

  // Evidence from another dispatch cannot satisfy this dispatch's release,
  // token, access, or explicit coverage requirements.
  auto other_release = releases;
  other_release[0].identity.dispatch_id += 1u;
  auto other_tokens = tokens;
  other_tokens[0].dispatch_id += 1u;
  other_tokens[0].source_release.dispatch_id += 1u;
  auto other_accesses = accesses;
  for (auto &access : other_accesses)
    access.dispatch_id += 1u;
  counters.dispatch_id += 1u;
  const auto cross_dispatch = consan_moi_inline_qualify_token_evidence(
      expected, other_release, other_tokens, other_accesses, counters);
  EXPECT_TRUE(cross_dispatch.has(ConSanMoiInlineQualificationFailure::DispatchCoverage));
  EXPECT_TRUE(cross_dispatch.has(ConSanMoiInlineQualificationFailure::StableRelease));
  EXPECT_TRUE(cross_dispatch.has(ConSanMoiInlineQualificationFailure::DirectToken));
  EXPECT_TRUE(cross_dispatch.has(ConSanMoiInlineQualificationFailure::ProducerAccess));
  EXPECT_TRUE(cross_dispatch.has(ConSanMoiInlineQualificationFailure::ConsumerAccess));
}

TEST(ConSanMoi, InlineTokenEvidenceRelaxedControlForbidsAuthorizationAndRetainsSourceDiagnostic) {
  constexpr uint64_t dispatch = 0xD150000000000001ull;
  constexpr uint32_t workgroup = 19;
  constexpr ConSanMoiInlineVersionedReleaseIdentity identity{dispatch, 0x4000, workgroup};
  constexpr ConSanMoiInlineQualificationExpectation expected{
      ConSanMoiInlineQualificationSemantics::NativeRelaxed,
      identity,
      /*producer_owner_id=*/3,
      /*producer_epoch_plus_one=*/12,
      /*consumer_owner_id=*/7,
      /*required_ancestor_owner_id=*/0,
      /*required_ancestor_epoch_plus_one=*/0};
  const std::array accesses = {
      ConSanMoiInlineAccessEvidence{dispatch, workgroup, /*owner_id=*/3, /*access_count=*/1},
      ConSanMoiInlineAccessEvidence{dispatch, workgroup, /*owner_id=*/7, /*access_count=*/1},
  };
  ConSanMoiInlineEvidenceCounters counters{dispatch,
                                           1,
                                           0,
                                           0,
                                           0,
                                           /*consan_diagnostic_count=*/1,
                                           /*matched_source_diagnostic_count=*/1};
  const std::array<ConSanMoiInlineStableReleaseEvidence, 0> no_releases{};
  const std::array<ConSanMoiInlineTokenEvidence, 0> no_tokens{};
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, no_releases, no_tokens, accesses, counters)
          .accepted());

  counters.matched_source_diagnostic_count = 0;
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, no_releases, no_tokens, accesses, counters)
          .has(ConSanMoiInlineQualificationFailure::MissingSourceDiagnostic));
  counters.matched_source_diagnostic_count = 1;
  counters.consan_diagnostic_count = 2;
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, no_releases, no_tokens, accesses, counters)
          .has(ConSanMoiInlineQualificationFailure::UnexpectedDiagnostic));

  counters.consan_diagnostic_count = 1;
  const std::array releases = {ConSanMoiInlineStableReleaseEvidence{
      identity, /*version_before=*/2, /*version_after=*/2,
      /*releaser_owner_id=*/3, /*releaser_epoch_plus_one=*/12, /*snapshot=*/{}}};
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, releases, no_tokens, accesses, counters)
          .has(ConSanMoiInlineQualificationFailure::UnexpectedAuthorization));
  const std::array tokens = {ConSanMoiInlineTokenEvidence{
      dispatch, workgroup, /*consumer_owner_id=*/7, /*producer_owner_id=*/3,
      /*producer_epoch_plus_one=*/12, ConSanMoiInlineTokenEvidenceKind::Direct, identity,
      /*source_release_version=*/2}};
  EXPECT_TRUE(
      consan_moi_inline_qualify_token_evidence(expected, no_releases, tokens, accesses, counters)
          .has(ConSanMoiInlineQualificationFailure::UnexpectedAuthorization));
}

TEST(ConSanMoi, FinalValidationPinsVersionedCausalReleaseTransaction) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_global_cas_code_object(
      /*return_old_value=*/true, /*vector_only_address=*/false);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult valid = try_patch_consan(bytes, options);

  ASSERT_TRUE(valid.errors.empty()) << testing::PrintToString(valid.errors);
  ASSERT_TRUE(valid.modified) << testing::PrintToString(valid.warnings);
  ASSERT_FALSE(valid.text_sections.empty());
  const auto patch = std::ranges::find_if(valid.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
  });
  ASSERT_NE(patch, valid.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, valid).empty());

  const size_t body_file_offset =
      valid.text_sections.front().file_offset + patch->trampoline_offset;
  ASSERT_LE(body_file_offset + patch->trampoline_size, valid.elf_bytes.size());
  std::vector<uint32_t> body(patch->trampoline_size / sizeof(uint32_t));
  std::memcpy(body.data(), valid.elf_bytes.data() + body_file_offset, patch->trampoline_size);
  const uint16_t scratch = *patch->scratch_vgpr;
  const auto version_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 5u), static_cast<uint16_t>(scratch + 5u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(version_cas);
  const auto cas_position =
      std::search(body.begin(), body.end(), version_cas->begin(), version_cas->end());
  ASSERT_NE(cas_position, body.end());

  ConSanResult wrong_claim = valid;
  const auto atomic_add = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 5u), static_cast<uint16_t>(scratch + 5u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_add);
  ASSERT_EQ(atomic_add->size(), version_cas->size());
  const size_t claim_byte_offset =
      static_cast<size_t>(std::distance(body.begin(), cas_position)) * sizeof(uint32_t);
  std::memcpy(wrong_claim.elf_bytes.data() + body_file_offset + claim_byte_offset,
              atomic_add->data(), atomic_add->size() * sizeof(uint32_t));
  const std::vector<std::string> claim_errors = validate_consan_modified_elf(bytes, wrong_claim);
  EXPECT_TRUE(std::ranges::any_of(claim_errors, [](const std::string &error) {
    return error.find("versioned release transaction semantics") != std::string::npos;
  }));

  const auto snapshot_flags = build_flat_store_b32_vaddr_vsrc(
      static_cast<uint16_t>(scratch + 5u), static_cast<uint16_t>(scratch + 8u),
      ROCJITSU_CODE_ARCH_RDNA4, offsetof(ConSanMoiInlineCausalSnapshot, flags));
  ASSERT_TRUE(snapshot_flags);
  const auto flags_position =
      std::search(body.begin(), body.end(), snapshot_flags->begin(), snapshot_flags->end());
  ASSERT_NE(flags_position, body.end());
  const auto wrong_flags = build_flat_store_b32_vaddr_vsrc(
      static_cast<uint16_t>(scratch + 5u), static_cast<uint16_t>(scratch + 7u),
      ROCJITSU_CODE_ARCH_RDNA4, offsetof(ConSanMoiInlineCausalSnapshot, flags));
  ASSERT_TRUE(wrong_flags);
  ASSERT_EQ(wrong_flags->size(), snapshot_flags->size());
  ConSanResult incomplete_snapshot = valid;
  const size_t flags_byte_offset =
      static_cast<size_t>(std::distance(body.begin(), flags_position)) * sizeof(uint32_t);
  std::memcpy(incomplete_snapshot.elf_bytes.data() + body_file_offset + flags_byte_offset,
              wrong_flags->data(), wrong_flags->size() * sizeof(uint32_t));
  const std::vector<std::string> snapshot_errors =
      validate_consan_modified_elf(bytes, incomplete_snapshot);
  EXPECT_TRUE(std::ranges::any_of(snapshot_errors, [](const std::string &error) {
    return error.find("versioned release transaction semantics") != std::string::npos;
  }));
}

TEST(ConSanMoi, InlineAcquiredEpochTokenLookupIsPairAndWorkgroupScoped) {
  constexpr uint32_t workgroup_key = 19;
  constexpr uint32_t consumer = 7;
  constexpr uint32_t producer = 3;
  ConSanMoiInlineAcquiredEpochTokenSlot slot;
  EXPECT_EQ(consan_moi_inline_acquired_epoch_token_lookup(slot, workgroup_key, consumer, producer),
            ConSanMoiInlineAcquiredEpochTokenLookup::Empty);

  slot = {.version = 2,
          .consumer_owner_id = consumer,
          .producer_owner_id = producer,
          .producer_epoch_plus_one = 12,
          .workgroup_key = workgroup_key,
          .kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct),
          .dispatch_id = 0xd150000000000001ull,
          .source_release_address = 0x4000,
          .source_release_version = 2,
          .consumer_epoch_plus_one = 1};
  EXPECT_EQ(consan_moi_inline_acquired_epoch_token_lookup(slot, workgroup_key, consumer, producer),
            ConSanMoiInlineAcquiredEpochTokenLookup::Exact);
  auto incomplete = slot;
  incomplete.dispatch_id = 0;
  EXPECT_EQ(
      consan_moi_inline_acquired_epoch_token_lookup(incomplete, workgroup_key, consumer, producer),
      ConSanMoiInlineAcquiredEpochTokenLookup::Collision);
  ConSanMoiInlineAcquiredEpochTokenSlot hidden_empty;
  hidden_empty.producer_owner_id = producer;
  EXPECT_EQ(consan_moi_inline_acquired_epoch_token_lookup(hidden_empty, workgroup_key, consumer,
                                                          producer),
            ConSanMoiInlineAcquiredEpochTokenLookup::Collision);
  EXPECT_EQ(
      consan_moi_inline_acquired_epoch_token_lookup(slot, workgroup_key, consumer, producer + 1u),
      ConSanMoiInlineAcquiredEpochTokenLookup::Collision);
  EXPECT_EQ(
      consan_moi_inline_acquired_epoch_token_lookup(slot, workgroup_key + 1u, consumer, producer),
      ConSanMoiInlineAcquiredEpochTokenLookup::Collision);

  constexpr uint32_t capacity = 64;
  const uint32_t index = consan_moi_inline_acquired_epoch_token_slot_index(workgroup_key, consumer,
                                                                           producer, capacity);
  EXPECT_LT(index, capacity);
  EXPECT_EQ(consan_moi_inline_acquired_epoch_token_slot_index(workgroup_key, consumer, producer, 0),
            0u);
  EXPECT_EQ(
      consan_moi_inline_acquired_epoch_token_slot_index(workgroup_key, consumer, producer, 48), 0u);
}

TEST(ConSanMoi, InlineFullTokenClassifierRequiresOneStableCompleteIdentity) {
  ConSanMoiInlineAcquiredEpochTokenSlot token{
      .version = 6,
      .consumer_owner_id = 7,
      .producer_owner_id = 3,
      .producer_epoch_plus_one = 12,
      .workgroup_key = 31,
      .kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct),
      .dispatch_id = 0xd150000000000001ull,
      .source_release_address = 0x4000,
      .source_release_version = 4,
      .consumer_epoch_plus_one = 1};
  EXPECT_EQ(consan_moi_inline_classify_acquired_token({6, token, 6}).state,
            ConSanMoiInlineAcquiredTokenState::Stable);
  token.kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence);
  EXPECT_EQ(consan_moi_inline_classify_acquired_token({6, token, 6}).state,
            ConSanMoiInlineAcquiredTokenState::Stable);
  token.kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct);
  EXPECT_EQ(consan_moi_inline_classify_acquired_token({6, token, 8}).state,
            ConSanMoiInlineAcquiredTokenState::Changed);
  token.version = 7;
  EXPECT_EQ(consan_moi_inline_classify_acquired_token({7, token, 7}).state,
            ConSanMoiInlineAcquiredTokenState::Publishing);
  token = {};
  EXPECT_EQ(consan_moi_inline_classify_acquired_token({0, token, 0}).state,
            ConSanMoiInlineAcquiredTokenState::Empty);
  token.producer_owner_id = 3;
  EXPECT_EQ(consan_moi_inline_classify_acquired_token({0, token, 0}).state,
            ConSanMoiInlineAcquiredTokenState::Malformed);
}

TEST(ConSanMoi, InlineFullTokenTransactionRollsBackEveryReservation) {
  std::array<ConSanMoiInlineAcquiredEpochTokenSlot, 64> table{};
  auto make_token = [](uint32_t producer, ConSanMoiInlineTokenEvidenceKind kind) {
    return ConSanMoiInlineAcquiredEpochTokenSlot{.consumer_owner_id = 7,
                                                 .producer_owner_id = producer,
                                                 .producer_epoch_plus_one = 12,
                                                 .workgroup_key = 31,
                                                 .kind = static_cast<uint32_t>(kind),
                                                 .dispatch_id = 0xd150000000000001ull,
                                                 .source_release_address = 0x4000,
                                                 .source_release_version = 4,
                                                 .consumer_epoch_plus_one = 1};
  };
  std::array desired{make_token(3, ConSanMoiInlineTokenEvidenceKind::Direct),
                     make_token(5, ConSanMoiInlineTokenEvidenceKind::Inherited),
                     make_token(3, ConSanMoiInlineTokenEvidenceKind::ReleaseSequence)};
  ASSERT_NE(consan_moi_inline_acquired_epoch_token_slot_index(31, 7, 3, table.size()),
            consan_moi_inline_acquired_epoch_token_slot_index(31, 7, 5, table.size()));

  const std::array<bool, 3> lose_second{true, false, true};
  auto result = consan_moi_inline_publish_acquired_token_transaction(table, desired, lose_second);
  EXPECT_TRUE(result.claim_failed);
  EXPECT_FALSE(result.committed);
  for (const auto &slot : table)
    EXPECT_EQ(slot.version, 0u);

  result = consan_moi_inline_publish_acquired_token_transaction(table, desired);
  ASSERT_TRUE(result.committed);
  for (const auto &wanted : desired) {
    const auto &actual = table[consan_moi_inline_acquired_epoch_token_slot_index(
        wanted.workgroup_key, wanted.consumer_owner_id, wanted.producer_owner_id, table.size(),
        wanted.kind == static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence))];
    EXPECT_EQ(
        consan_moi_inline_classify_acquired_token({actual.version, actual, actual.version}).state,
        ConSanMoiInlineAcquiredTokenState::Stable);
    EXPECT_TRUE(consan_moi_inline_acquired_token_identity_matches(actual, wanted));
  }

  std::array duplicate{desired[0], desired[0]};
  duplicate[1].producer_epoch_plus_one = 13;
  const auto before = table;
  result = consan_moi_inline_publish_acquired_token_transaction(table, duplicate);
  EXPECT_TRUE(result.duplicate_destination);
  EXPECT_EQ(table, before);
}

TEST(ConSanMoi, InlineAcquiredEpochTokenOrdersOnlyItsExactPair) {
  constexpr uint32_t workgroup_key = 31;
  const ConSanMoiExactShadowEntry prior{ConSanMoiShadowAccessKind::Write,
                                        /*owner_id=*/3,
                                        /*epoch=*/11,
                                        /*generation=*/workgroup_key,
                                        /*instruction_offset=*/0x100};
  const ConSanMoiExactShadowEntry current{ConSanMoiShadowAccessKind::Read,
                                          /*owner_id=*/7,
                                          /*epoch=*/11,
                                          /*generation=*/workgroup_key,
                                          /*instruction_offset=*/0x200};
  ConSanMoiInlineAcquiredEpochTokenSlot token{
      .version = 2,
      .consumer_owner_id = 7,
      .producer_owner_id = 3,
      .producer_epoch_plus_one = 12,
      .workgroup_key = workgroup_key,
      .kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct),
      .dispatch_id = 0xd150000000000001ull,
      .source_release_address = 0x4000,
      .source_release_version = 2,
      .consumer_epoch_plus_one = 12};
  EXPECT_TRUE(consan_moi_inline_acquired_epoch_orders(token, current, prior));

  auto unrelated_prior = prior;
  unrelated_prior.owner_id = 4;
  EXPECT_FALSE(consan_moi_inline_acquired_epoch_orders(token, current, unrelated_prior));
  auto other_workgroup = current;
  other_workgroup.generation = workgroup_key + 1u;
  EXPECT_FALSE(consan_moi_inline_acquired_epoch_orders(token, other_workgroup, prior));
  token.producer_epoch_plus_one = 11;
  EXPECT_FALSE(consan_moi_inline_acquired_epoch_orders(token, current, prior));
  token.producer_epoch_plus_one = 12;
  token.source_release_version = 3;
  EXPECT_FALSE(consan_moi_inline_acquired_epoch_orders(token, current, prior));
}

TEST(ConSanMoi, InlineAcquiredEpochTokenOrdersObservedPairInEitherDirection) {
  constexpr uint32_t workgroup_key = 31;
  const ConSanMoiExactShadowEntry producer{ConSanMoiShadowAccessKind::Write,
                                           /*owner_id=*/3,
                                           /*epoch=*/11,
                                           /*generation=*/workgroup_key,
                                           /*instruction_offset=*/0x100};
  const ConSanMoiExactShadowEntry consumer{ConSanMoiShadowAccessKind::Read,
                                           /*owner_id=*/7,
                                           /*epoch=*/11,
                                           /*generation=*/workgroup_key,
                                           /*instruction_offset=*/0x200};
  ConSanMoiInlineAcquiredEpochTokenSlot empty;
  const ConSanMoiInlineAcquiredEpochTokenSlot consumer_after_producer{
      .version = 2,
      .consumer_owner_id = consumer.owner_id,
      .producer_owner_id = producer.owner_id,
      .producer_epoch_plus_one = 12,
      .workgroup_key = workgroup_key,
      .kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct),
      .dispatch_id = 0xd150000000000001ull,
      .source_release_address = 0x4000,
      .source_release_version = 2,
      .consumer_epoch_plus_one = 12};

  EXPECT_TRUE(consan_moi_inline_acquired_epoch_orders_pair(consumer_after_producer, empty, consumer,
                                                           producer));
  EXPECT_TRUE(consan_moi_inline_acquired_epoch_orders_pair(empty, consumer_after_producer, producer,
                                                           consumer));
  EXPECT_FALSE(consan_moi_inline_acquired_epoch_orders_pair(empty, empty, consumer, producer));
}

TEST(ConSanMoi, InlineStableDirectAndInheritedTokensSurviveSourceReplacement) {
  constexpr uint64_t dispatch = 0xd150000000000001ull;
  constexpr uint32_t workgroup = 31;
  const ConSanMoiExactShadowEntry prior{ConSanMoiShadowAccessKind::Write, 3, 11, workgroup, 0x100};
  const ConSanMoiExactShadowEntry current{ConSanMoiShadowAccessKind::Read, 7, 11, workgroup, 0x200};
  ConSanMoiInlineAcquiredEpochTokenSlot token{
      .version = 6,
      .consumer_owner_id = current.owner_id,
      .producer_owner_id = prior.owner_id,
      .producer_epoch_plus_one = 12,
      .workgroup_key = workgroup,
      .kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct),
      .dispatch_id = dispatch,
      .source_release_address = 0x4000,
      .source_release_version = 4,
      .consumer_epoch_plus_one = 12};
  ConSanMoiInlineReleaseSnapshotWords source{.version_before = 4,
                                             .slot = {.version = 4,
                                                      .owner_id = prior.owner_id,
                                                      .epoch_plus_one = 12,
                                                      .workgroup_key = workgroup,
                                                      .atomic_address = 0x4000,
                                                      .dispatch_id = dispatch},
                                             .snapshot = {},
                                             .version_after = 4};
  EXPECT_TRUE(
      consan_moi_inline_stable_token_orders({6, token, 6}, source, dispatch, current, prior));

  // A later source-slot publication cannot revoke an acquire fact already
  // committed in a stable token.
  source.version_before = 8;
  source.version_after = 8;
  source.slot.version = 8;
  source.slot.owner_id = 5;
  source.slot.epoch_plus_one = 20;
  EXPECT_TRUE(
      consan_moi_inline_stable_token_orders({6, token, 6}, source, dispatch, current, prior));

  token.kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited);
  source.snapshot.entry_count = 1;
  source.snapshot.entries[0] = {prior.owner_id, 12};
  EXPECT_TRUE(
      consan_moi_inline_stable_token_orders({6, token, 6}, source, dispatch, current, prior));
  // The inherited fact was validated before token publication. A later
  // replacement or transient rewrite of the direct-mapped source snapshot is
  // irrelevant to the durable token.
  source.snapshot.entries[0].ancestor_epoch_plus_one = 11;
  EXPECT_TRUE(
      consan_moi_inline_stable_token_orders({6, token, 6}, source, dispatch, current, prior));
  source.snapshot.entries[0].ancestor_epoch_plus_one = 12;
  token.source_release_version = 8;
  EXPECT_TRUE(
      consan_moi_inline_stable_token_orders({6, token, 6}, source, dispatch, current, prior));
  token.source_release_version = 10;
  EXPECT_TRUE(
      consan_moi_inline_stable_token_orders({6, token, 6}, source, dispatch, current, prior));
  token.source_release_version = 4;
  token.kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence);
  EXPECT_FALSE(
      consan_moi_inline_stable_token_orders({6, token, 6}, source, dispatch, current, prior));
  token.kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited);
  token.dispatch_id ^= 1u;
  EXPECT_FALSE(
      consan_moi_inline_stable_token_orders({6, token, 6}, source, dispatch, current, prior));
  token.dispatch_id = dispatch;
  token.reservation_version = 1;
  EXPECT_FALSE(
      consan_moi_inline_stable_token_orders({6, token, 6}, source, dispatch, current, prior));
}

TEST(ConSanMoi, InlineDeferredQualificationRequiresExactStableCausalIdentity) {
  constexpr uint64_t dispatch = 0xd150000000000001ull;
  constexpr uint32_t workgroup = 31;
  constexpr uint32_t consumer = 7;
  constexpr uint32_t producer = 3;
  constexpr uint32_t consumer_epoch = 17;
  constexpr uint32_t producer_epoch = 11;
  ConSanMoiInlineAcquiredEpochTokenSlot token{
      .version = 2,
      .consumer_owner_id = consumer,
      .producer_owner_id = producer,
      .producer_epoch_plus_one = producer_epoch + 1u,
      .workgroup_key = workgroup,
      .kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct),
      .dispatch_id = dispatch,
      .source_release_address = 0x4000,
      .source_release_version = 2,
      .consumer_epoch_plus_one = consumer_epoch + 1u};

  EXPECT_TRUE(consan_moi_inline_stable_token_orders_deferred(
      token, dispatch, workgroup, consumer, producer, consumer_epoch, producer_epoch));
  token.kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited);
  EXPECT_TRUE(consan_moi_inline_stable_token_orders_deferred(
      token, dispatch, workgroup, consumer, producer, consumer_epoch, producer_epoch));

  token.kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence);
  EXPECT_FALSE(consan_moi_inline_stable_token_orders_deferred(
      token, dispatch, workgroup, consumer, producer, consumer_epoch, producer_epoch));
  token.kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct);
  EXPECT_FALSE(consan_moi_inline_stable_token_orders_deferred(
      token, dispatch ^ 1u, workgroup, consumer, producer, consumer_epoch, producer_epoch));
  EXPECT_FALSE(consan_moi_inline_stable_token_orders_deferred(
      token, dispatch, workgroup + 1u, consumer, producer, consumer_epoch, producer_epoch));
  EXPECT_FALSE(consan_moi_inline_stable_token_orders_deferred(
      token, dispatch, workgroup, producer, consumer, consumer_epoch, producer_epoch));
  EXPECT_FALSE(consan_moi_inline_stable_token_orders_deferred(
      token, dispatch, workgroup, consumer, producer, consumer_epoch, producer_epoch + 1u));
  EXPECT_FALSE(consan_moi_inline_stable_token_orders_deferred(
      token, dispatch, workgroup, consumer, producer, consumer_epoch - 1u, producer_epoch))
      << "a later acquire may not retroactively authorize an earlier consumer access";
  EXPECT_FALSE(consan_moi_inline_stable_token_orders_deferred(
      token, dispatch, workgroup, consumer, producer, consan_moi_exact_shadow::max_epoch,
      producer_epoch));
}

TEST(ConSanMoi, DeferredInlineDiagnosticFilterIsFailClosedAndPreservesDroppedCount) {
  constexpr uint64_t dispatch = 0xd150000000000001ull;
  constexpr uint32_t workgroup = 31;
  ConSanMoiInlineAcquiredEpochTokenSlot token{
      .version = 2,
      .consumer_owner_id = 7,
      .producer_owner_id = 3,
      .producer_epoch_plus_one = 12,
      .workgroup_key = workgroup,
      .kind = static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct),
      .dispatch_id = dispatch,
      .source_release_address = 0x4000,
      .source_release_version = 2,
      .consumer_epoch_plus_one = 18};
  ConSanMoiDiagnosticRecord ordered{
      .kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict),
      .backend = static_cast<uint32_t>(ConSanMoiEngine::InlineShadow),
      .generation = dispatch,
      .epoch = 17,
      .first_epoch = 11,
      .first_owner_id = 3,
      .second_owner_id = 7,
      .reserved = workgroup};
  auto reverse_replacement = ordered;
  reverse_replacement.epoch = ordered.first_epoch;
  reverse_replacement.first_epoch = ordered.epoch;
  reverse_replacement.first_owner_id = ordered.second_owner_id;
  reverse_replacement.second_owner_id = ordered.first_owner_id;
  auto unrelated = ordered;
  unrelated.first_owner_id = 5;
  const std::array diagnostics = {ordered, reverse_replacement, unrelated};
  const std::array tokens = {token};

  auto filtered = consan_moi_filter_deferred_inline_diagnostics(
      diagnostics, /*total_diagnostic_count=*/5, tokens, /*token_evidence_complete=*/true);
  EXPECT_EQ(filtered.qualified_count, 2u);
  EXPECT_EQ(filtered.visible_indices, std::vector<uint32_t>({2}));
  EXPECT_EQ(filtered.effective_diagnostic_count, 3u)
      << "three dropped diagnostics remain part of the effective total";

  filtered = consan_moi_filter_deferred_inline_diagnostics(
      diagnostics, /*total_diagnostic_count=*/5, tokens, /*token_evidence_complete=*/false);
  EXPECT_EQ(filtered.qualified_count, 0u);
  EXPECT_EQ(filtered.visible_indices, (std::vector<uint32_t>{0, 1, 2}));
  EXPECT_EQ(filtered.effective_diagnostic_count, 5u);

  auto later_acquire = token;
  later_acquire.consumer_epoch_plus_one = ordered.epoch + 2u;
  filtered = consan_moi_filter_deferred_inline_diagnostics(
      std::span<const ConSanMoiDiagnosticRecord>(&ordered, 1), /*total_diagnostic_count=*/1,
      std::span<const ConSanMoiInlineAcquiredEpochTokenSlot>(&later_acquire, 1),
      /*token_evidence_complete=*/true);
  EXPECT_EQ(filtered.qualified_count, 0u);
  EXPECT_EQ(filtered.visible_indices, std::vector<uint32_t>({0}));

  auto wrong_backend = ordered;
  wrong_backend.backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
  filtered = consan_moi_filter_deferred_inline_diagnostics(
      std::span<const ConSanMoiDiagnosticRecord>(&wrong_backend, 1),
      /*total_diagnostic_count=*/1, tokens, /*token_evidence_complete=*/true);
  EXPECT_EQ(filtered.qualified_count, 0u);
  EXPECT_EQ(filtered.visible_indices, std::vector<uint32_t>({0}));
}

TEST(ConSanMoi, InlineAcquiredEpochTokenPublicationIsMonotonicAndFailClosed) {
  ConSanMoiInlineAcquiredEpochTokenSlot slot;
  auto result = consan_moi_inline_publish_acquired_epoch_token(
      slot, /*workgroup_key=*/31, /*consumer_owner_id=*/7, /*producer_owner_id=*/3,
      /*producer_epoch=*/11, /*consumer_epoch=*/17,
      /*dispatch_id=*/0xd150000000000001ull, ConSanMoiInlineTokenEvidenceKind::Direct,
      /*source_release_address=*/0x4000,
      /*source_release_version=*/2);
  EXPECT_TRUE(result.updated);
  EXPECT_EQ(slot.producer_epoch_plus_one, 12u);
  EXPECT_EQ(slot.consumer_epoch_plus_one, 18u);

  result = consan_moi_inline_publish_acquired_epoch_token(
      slot, /*workgroup_key=*/31, /*consumer_owner_id=*/7, /*producer_owner_id=*/3,
      /*producer_epoch=*/5, /*consumer_epoch=*/17,
      /*dispatch_id=*/0xd150000000000001ull, ConSanMoiInlineTokenEvidenceKind::Direct,
      /*source_release_address=*/0x4000,
      /*source_release_version=*/2);
  EXPECT_FALSE(result.updated);
  EXPECT_FALSE(result.collision);
  EXPECT_EQ(slot.producer_epoch_plus_one, 12u);

  const auto unchanged = slot;
  result = consan_moi_inline_publish_acquired_epoch_token(
      slot, /*workgroup_key=*/31, /*consumer_owner_id=*/7, /*producer_owner_id=*/4,
      /*producer_epoch=*/20, /*consumer_epoch=*/17,
      /*dispatch_id=*/0xd150000000000001ull, ConSanMoiInlineTokenEvidenceKind::Direct,
      /*source_release_address=*/0x5000,
      /*source_release_version=*/4);
  EXPECT_TRUE(result.collision);
  EXPECT_EQ(slot.producer_owner_id, unchanged.producer_owner_id);
  EXPECT_EQ(slot.producer_epoch_plus_one, unchanged.producer_epoch_plus_one);

  result = consan_moi_inline_publish_acquired_epoch_token(
      slot, /*workgroup_key=*/0, /*consumer_owner_id=*/7, /*producer_owner_id=*/3,
      /*producer_epoch=*/20, /*consumer_epoch=*/17,
      /*dispatch_id=*/0xd150000000000001ull, ConSanMoiInlineTokenEvidenceKind::Direct,
      /*source_release_address=*/0x4000,
      /*source_release_version=*/2);
  EXPECT_TRUE(result.invalid_identity);
  EXPECT_EQ(slot.producer_epoch_plus_one, unchanged.producer_epoch_plus_one);
}

TEST(ConSanMoi, InlineAcquiredEpochTokenSaturatesAtPackedEpochBoundary) {
  EXPECT_EQ(consan_moi_inline_acquired_epoch_token_value(0), 1u);
  EXPECT_EQ(consan_moi_inline_acquired_epoch_token_value(consan_moi_exact_shadow::max_epoch - 1u),
            consan_moi_exact_shadow::max_epoch);
  EXPECT_EQ(consan_moi_inline_acquired_epoch_token_value(consan_moi_exact_shadow::max_epoch),
            consan_moi_exact_shadow::max_epoch);
  EXPECT_EQ(consan_moi_inline_acquired_epoch_token_value(consan_moi_exact_shadow::max_epoch + 1u),
            consan_moi_exact_shadow::max_epoch);
}

TEST(ConSanMoi, InlineAtomicSupportInventoryPinsAdmittedAndDeferredClasses) {
  ConSanAtomicSite site;
  site.mnemonic = "flat_atomic_add_u32";
  site.size = 3u * sizeof(uint32_t);
  site.width_bits = 32;
  site.addr_vgpr = 2;
  site.data_vgpr = 4;
  site.raw_saddr = rdna4::OPR_SREG_NULL;
  site.raw_ioffset = 0;
  site.raw_scope = 2;
  site.raw_th = 0;
  site.returns_old_value = false;

  EXPECT_EQ(classify_consan_moi_inline_atomic_support(site, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::Supported);
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(site, ConSanMoiAtomicEventKind::Acquire),
            ConSanMoiInlineAtomicSupport::Supported);
  EXPECT_EQ(
      classify_consan_moi_inline_atomic_support(site, ConSanMoiAtomicEventKind::AcquireRelease),
      ConSanMoiInlineAtomicSupport::Supported);

  ConSanAtomicSite changed = site;
  changed.mnemonic = "flat_atomic_cmpswap_u32";
  changed.dst_vgpr = 6;
  changed.returns_old_value = true;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::Supported);
  changed.returns_old_value = false;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::CompareExchangeOutcomeUnavailable);
  changed = site;
  changed.mnemonic = "global_atomic_add_u32";
  changed.raw_saddr = 4;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::Supported);
  changed.size = 2u * sizeof(uint32_t);
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::UnsupportedEncoding);
  EXPECT_EQ(consan_capability_disposition(ROCJITSU_CODE_TARGET_GFX950,
                                          ConSanCapabilityEngine::InlineShadow,
                                          ConSanCapabilityForm::OrderedVglobalAtomic),
            ConSanCapabilityDisposition::Unsupported);
  changed.size = 3u * sizeof(uint32_t);
  changed.mnemonic = "global_atomic_cmpswap_b32";
  changed.dst_vgpr = 6;
  changed.returns_old_value = true;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::Supported);
  changed.returns_old_value = false;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::CompareExchangeOutcomeUnavailable);
  changed = site;
  changed.width_bits = 64;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::UnsupportedWidth);
  changed = site;
  changed.raw_ioffset = 4;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::NonzeroOffset);
  changed = site;
  changed.raw_scope = 0;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::UnsupportedScope);
  changed.raw_scope = 1;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::Supported);
  changed.raw_scope = 3;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::Supported);
  changed.raw_scope = 4;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::UnsupportedScope);
  changed = site;
  changed.raw_saddr = 4;
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::UnsupportedEncoding);
  changed = site;
  changed.addr_vgpr.reset();
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::MissingOperands);
  changed = site;
  changed.raw_scope.reset();
  EXPECT_EQ(classify_consan_moi_inline_atomic_support(changed, ConSanMoiAtomicEventKind::Release),
            ConSanMoiInlineAtomicSupport::MissingOrderingMetadata);
}

TEST(ConSanMoi, InlineAtomicSupportReasonNamesAreStable) {
  EXPECT_EQ(consan_moi_inline_atomic_support_name(ConSanMoiInlineAtomicSupport::Supported),
            "supported");
  EXPECT_EQ(consan_moi_inline_atomic_support_name(
                ConSanMoiInlineAtomicSupport::CompareExchangeOutcomeUnavailable),
            "compare-exchange-outcome-unavailable");
}

TEST(ConSanMoi, SharedAtomicAddressPlanAliasesFlatAndMaterializesVglobal) {
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::Moi;
  const ConSanResult global_inventory =
      try_patch_consan(make_rdna4_global_atomic_code_object(), inventory_options);
  ASSERT_TRUE(global_inventory.errors.empty()) << testing::PrintToString(global_inventory.errors);
  ASSERT_EQ(global_inventory.kernels.size(), 1u);
  ASSERT_EQ(global_inventory.kernels.front().atomic_sites.size(), 1u);
  const ConSanAtomicSite &global_site = global_inventory.kernels.front().atomic_sites.front();

  const ConSanMoiAtomicAddressPlan global_plan = plan_consan_moi_atomic_address(
      global_site, /*scratch_vgpr=*/8, /*scratch_vgpr_count=*/5,
      ConSanRegisterAllocationSource::Explicit, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(global_plan.supported())
      << consan_moi_atomic_address_support_name(global_plan.support);
  EXPECT_EQ(global_plan.kind, ConSanMoiAtomicAddressKind::VglobalMaterialized);
  EXPECT_EQ(global_plan.input_address_vgpr, 2u);
  EXPECT_EQ(global_plan.input_address_vgpr_count, 1u);
  ASSERT_TRUE(global_plan.scalar_base_sgpr);
  EXPECT_EQ(*global_plan.scalar_base_sgpr, 4u);
  EXPECT_EQ(global_plan.result_address_vgpr, 11u);
  EXPECT_EQ(global_plan.result_address_vgpr_count, 2u);

  ConSanAtomicSite vector_only_global = global_site;
  vector_only_global.raw_saddr = rdna4::OPR_SREG_NULL;
  vector_only_global.saddr_sgpr = rdna4::OPR_SREG_NULL;
  vector_only_global.addr_vgpr = 2u;
  vector_only_global.raw_vaddr = 2u;
  const ConSanMoiAtomicAddressPlan vector_only_plan = plan_consan_moi_atomic_address(
      vector_only_global, /*scratch_vgpr=*/8, /*scratch_vgpr_count=*/3,
      ConSanRegisterAllocationSource::LivenessDead, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(vector_only_plan.supported())
      << consan_moi_atomic_address_support_name(vector_only_plan.support);
  EXPECT_EQ(vector_only_plan.kind, ConSanMoiAtomicAddressKind::VglobalGuestPair);
  EXPECT_EQ(vector_only_plan.input_address_vgpr_count, 2u);
  EXPECT_EQ(vector_only_plan.result_address_vgpr, 2u);
  EXPECT_FALSE(vector_only_plan.requires_materialization());

  const ConSanResult flat_inventory =
      try_patch_consan(make_rdna4_flat_atomic_code_object(), inventory_options);
  ASSERT_TRUE(flat_inventory.errors.empty()) << testing::PrintToString(flat_inventory.errors);
  ASSERT_EQ(flat_inventory.kernels.size(), 1u);
  ASSERT_EQ(flat_inventory.kernels.front().atomic_sites.size(), 1u);
  const ConSanMoiAtomicAddressPlan flat_plan = plan_consan_moi_atomic_address(
      flat_inventory.kernels.front().atomic_sites.front(), /*scratch_vgpr=*/8,
      /*scratch_vgpr_count=*/3, ConSanRegisterAllocationSource::LivenessDead,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(flat_plan.supported()) << consan_moi_atomic_address_support_name(flat_plan.support);
  EXPECT_EQ(flat_plan.kind, ConSanMoiAtomicAddressKind::FlatGuestPair);
  EXPECT_EQ(flat_plan.input_address_vgpr_count, 2u);
  EXPECT_EQ(flat_plan.result_address_vgpr, flat_plan.input_address_vgpr);
  EXPECT_FALSE(flat_plan.requires_materialization());
  const auto flat_words = build_consan_moi_atomic_address_materialization(
      flat_plan, /*vcc_save_sgpr=*/80, /*scc_save_sgpr=*/82, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(flat_words);
  EXPECT_TRUE(flat_words->empty());
}

TEST(ConSanMoi, SharedAddressPlanMaterializesGfx1250BufferResourceAddress) {
  ConSanAtomicSite site;
  site.text_offset = 0;
  site.file_offset = 0;
  site.size = 3u * sizeof(uint32_t);
  site.width_bits = 128u;
  site.addr_vgpr = 4u;
  site.data_vgpr = 24u;
  site.saddr_sgpr = 72u;
  site.raw_vaddr = 4u;
  site.raw_rsrc = 72u;
  site.raw_soffset = 10u;
  site.raw_offen = true;
  site.raw_idxen = false;
  site.raw_ioffset = 16;
  site.raw_scope = 2u;
  site.raw_th = 0u;
  site.returns_old_value = false;
  site.mnemonic = "buffer_store_b128";

  const ConSanMoiAtomicAddressPlan plan = plan_consan_moi_atomic_address(
      site, /*scratch_vgpr=*/0, /*scratch_vgpr_count=*/5,
      ConSanRegisterAllocationSource::LivenessDead, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(plan.supported()) << consan_moi_atomic_address_support_name(plan.support);
  EXPECT_EQ(plan.kind, ConSanMoiAtomicAddressKind::BufferResourceMaterialized);
  EXPECT_EQ(plan.input_address_vgpr, 4u);
  EXPECT_EQ(plan.result_address_vgpr, 3u);
  ASSERT_TRUE(plan.scalar_base_sgpr);
  EXPECT_EQ(*plan.scalar_base_sgpr, 72u);
  ASSERT_TRUE(plan.scalar_offset_sgpr);
  EXPECT_EQ(*plan.scalar_offset_sgpr, 10u);
  EXPECT_EQ(plan.signed_byte_offset, 16);

  const auto words = build_consan_moi_atomic_address_materialization(
      plan, /*vcc_save_sgpr=*/80, /*scc_save_sgpr=*/82, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(words);
  EXPECT_GT(words->size(), 10u);

  EXPECT_EQ(plan_consan_moi_atomic_address(site, 0, 5, ConSanRegisterAllocationSource::LivenessDead,
                                           ROCJITSU_CODE_ARCH_RDNA4)
                .support,
            ConSanMoiAtomicAddressSupport::UnsupportedArchitecture);
  site.raw_idxen = true;
  EXPECT_EQ(plan_consan_moi_atomic_address(site, 0, 5, ConSanRegisterAllocationSource::LivenessDead,
                                           ROCJITSU_CODE_ARCH_GFX1250)
                .support,
            ConSanMoiAtomicAddressSupport::UnsupportedEncoding);
}

TEST(ConSanMoi, SharedAddressPlanMaterializesGfx1250LdsTokenWithByteOffset) {
  ConSanAtomicSite site;
  site.size = 2u * sizeof(uint32_t);
  site.width_bits = 32u;
  site.addr_vgpr = 4u;
  site.data_vgpr = 6u;
  site.raw_addr = 4u;
  site.raw_data0 = 6u;
  site.raw_ioffset = 12;
  site.raw_scope = 1u;
  site.returns_old_value = false;
  site.mnemonic = "ds_add_u32";

  const ConSanMoiAtomicAddressPlan plan = plan_consan_moi_atomic_address(
      site, /*scratch_vgpr=*/8, /*scratch_vgpr_count=*/5,
      ConSanRegisterAllocationSource::LivenessDead, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(plan.supported()) << consan_moi_atomic_address_support_name(plan.support);
  EXPECT_EQ(plan.kind, ConSanMoiAtomicAddressKind::LdsByteOffsetToken);
  EXPECT_EQ(plan.input_address_vgpr, 4u);
  EXPECT_EQ(plan.input_address_vgpr_count, 1u);
  EXPECT_EQ(plan.signed_byte_offset, 12);
  EXPECT_EQ(plan.result_address_vgpr, 11u);
  EXPECT_EQ(plan.result_address_vgpr_count, 2u);

  const auto words = build_consan_moi_atomic_address_materialization(
      plan, /*vcc_save_sgpr=*/80, /*scc_save_sgpr=*/82, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(words);
  EXPECT_EQ(words->front(),
            build_v_mov_b32_e32(/*vdst=*/11, vector_source_vgpr(4), ROCJITSU_CODE_ARCH_GFX1250));
  const auto tag = build_v_mov_b32_e64_literal(
      /*vdst=*/12, kConSanMoiLdsAddressTokenTag, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(tag);
  EXPECT_GT(words->size(), tag->size() + 1u);
  EXPECT_TRUE(contains_subsequence(*words, *tag));

  EXPECT_EQ(plan_consan_moi_atomic_address(site, 8, 5, ConSanRegisterAllocationSource::LivenessDead,
                                           ROCJITSU_CODE_ARCH_RDNA4)
                .support,
            ConSanMoiAtomicAddressSupport::UnsupportedArchitecture);
}

TEST(ConSanMoi, VglobalAddressMaterializationPreservesSpecialStateAndSignedOffset) {
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::Moi;
  const ConSanResult inventory =
      try_patch_consan(make_rdna4_global_atomic_code_object(), inventory_options);
  ASSERT_EQ(inventory.kernels.size(), 1u);
  ASSERT_EQ(inventory.kernels.front().atomic_sites.size(), 1u);
  ConSanAtomicSite site = inventory.kernels.front().atomic_sites.front();
  site.raw_ioffset = -4;

  constexpr uint16_t kVccSave = 80;
  constexpr uint16_t kSccSave = 82;
  struct RdnaFamilyTarget {
    rj_code_arch_t arch;
    std::string_view label;
  };
  constexpr std::array<RdnaFamilyTarget, 2> kTargets = {{
      {ROCJITSU_CODE_ARCH_RDNA4, "gfx1201/rdna4"},
      {ROCJITSU_CODE_ARCH_GFX1250, "gfx1250"},
  }};
  for (const RdnaFamilyTarget &target : kTargets) {
    SCOPED_TRACE(target.label);
    const ConSanMoiAtomicAddressPlan plan = plan_consan_moi_atomic_address(
        site, /*scratch_vgpr=*/8, /*scratch_vgpr_count=*/5,
        ConSanRegisterAllocationSource::DescriptorGrowth, target.arch);
    ASSERT_TRUE(plan.supported()) << consan_moi_atomic_address_support_name(plan.support);
    EXPECT_EQ(plan.signed_byte_offset, -4);

    const auto words =
        build_consan_moi_atomic_address_materialization(plan, kVccSave, kSccSave, target.arch);
    const auto save_scc = build_rdna4_s_cselect_b32(kSccSave, scalar_positive_inline_u32(1),
                                                    scalar_positive_inline_u32(0), target.arch);
    const auto save_vcc = build_s_mov_b64(kVccSave, kRdna4VccLo, target.arch);
    const auto add_vaddr =
        build_v_add_u64_vgpr_offset(plan.result_address_vgpr, plan.input_address_vgpr, target.arch);
    const auto add_negative = build_v_add_u64_signed_i24(plan.result_address_vgpr, -4, target.arch);
    const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, kVccSave, target.arch);
    const auto restore_scc =
        build_rdna4_s_cmp_lg_u32(kSccSave, scalar_positive_inline_u32(0), target.arch);
    ASSERT_TRUE(words);
    ASSERT_TRUE(save_scc);
    ASSERT_TRUE(save_vcc);
    ASSERT_TRUE(add_vaddr);
    ASSERT_TRUE(add_negative);
    ASSERT_TRUE(restore_vcc);
    ASSERT_TRUE(restore_scc);

    std::vector<uint32_t> expected = {
        *save_scc,
        *save_vcc,
        build_v_mov_b32_e32(plan.result_address_vgpr, 4u, target.arch),
        build_v_mov_b32_e32(plan.result_address_vgpr + 1u, 5u, target.arch),
    };
    expected.insert(expected.end(), add_vaddr->begin(), add_vaddr->end());
    expected.insert(expected.end(), add_negative->begin(), add_negative->end());
    expected.push_back(*restore_vcc);
    expected.push_back(*restore_scc);
    EXPECT_EQ(*words, expected);
  }
}

TEST(ConSanMoi, DisplacedVectorOnlyVglobalMaterializesGuestPairAndSignedOffset) {
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::Moi;
  const ConSanResult inventory = try_patch_consan(
      make_rdna4_displaced_vglobal_atomic_release_acquire_code_object(), inventory_options);
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);
  ASSERT_EQ(inventory.kernels.size(), 1u);
  ASSERT_EQ(inventory.kernels.front().atomic_sites.size(), 2u);
  const ConSanAtomicSite &site = inventory.kernels.front().atomic_sites[1];
  ASSERT_EQ(site.raw_saddr, rdna4::OPR_SREG_NULL);
  ASSERT_EQ(site.raw_ioffset, 20);

  const ConSanMoiAtomicAddressPlan plan = plan_consan_moi_atomic_address(
      site, /*scratch_vgpr=*/12, /*scratch_vgpr_count=*/5,
      ConSanRegisterAllocationSource::LivenessDead, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(plan.supported()) << consan_moi_atomic_address_support_name(plan.support);
  EXPECT_EQ(plan.kind, ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized);
  EXPECT_EQ(plan.input_address_vgpr, 8u);
  EXPECT_EQ(plan.input_address_vgpr_count, 2u);
  EXPECT_FALSE(plan.scalar_base_sgpr);
  EXPECT_EQ(plan.signed_byte_offset, 20);
  EXPECT_EQ(plan.result_address_vgpr, 15u);
  EXPECT_TRUE(plan.requires_materialization());

  constexpr uint16_t kVccSave = 80;
  constexpr uint16_t kSccSave = 82;
  const auto words = build_consan_moi_atomic_address_materialization(plan, kVccSave, kSccSave,
                                                                     ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(words);
  const auto add_offset =
      build_v_add_u64_signed_i24(/*address_vgpr=*/15, 20, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(add_offset);
  ASSERT_EQ(words->size(), 12u);
  EXPECT_EQ((*words)[2],
            build_v_mov_b32_e32(/*vdst=*/15, vector_source_vgpr(8), ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ((*words)[3],
            build_v_mov_b32_e32(/*vdst=*/16, vector_source_vgpr(9), ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(std::equal(add_offset->begin(), add_offset->end(), words->begin() + 4));

  ConSanAtomicSite boundary = site;
  boundary.raw_ioffset = -(1 << 23);
  EXPECT_TRUE(plan_consan_moi_atomic_address(boundary, 12, 5,
                                             ConSanRegisterAllocationSource::SpillRequired,
                                             ROCJITSU_CODE_ARCH_RDNA4)
                  .supported());
  boundary.raw_ioffset = (1 << 23) - 1;
  EXPECT_TRUE(plan_consan_moi_atomic_address(boundary, 12, 5,
                                             ConSanRegisterAllocationSource::SpillRequired,
                                             ROCJITSU_CODE_ARCH_RDNA4)
                  .supported());
  boundary.raw_ioffset = 1 << 23;
  EXPECT_EQ(plan_consan_moi_atomic_address(boundary, 12, 5,
                                           ConSanRegisterAllocationSource::SpillRequired,
                                           ROCJITSU_CODE_ARCH_RDNA4)
                .support,
            ConSanMoiAtomicAddressSupport::UnsupportedOffset);
  EXPECT_EQ(plan_consan_moi_atomic_address(
                site, 12, 3, ConSanRegisterAllocationSource::LivenessDead, ROCJITSU_CODE_ARCH_RDNA4)
                .support,
            ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
}

TEST(ConSanMoi, VglobalAddressPlanAcceptsSpillResourcesAndPinsPrivatePair) {
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::Moi;
  const ConSanResult inventory =
      try_patch_consan(make_rdna4_global_atomic_code_object(), inventory_options);
  ASSERT_EQ(inventory.kernels.size(), 1u);
  ASSERT_EQ(inventory.kernels.front().atomic_sites.size(), 1u);
  const ConSanAtomicSite &site = inventory.kernels.front().atomic_sites.front();

  RegisterSet live;
  expand_all_vgprs(live);
  ConSanRegisterRequest request = vgpr_request(/*count=*/5, /*current_allocation_count=*/256,
                                               /*max_referenced_count=*/256);
  request.forbidden.expand({RegClass::VGPR, 0, 4});
  const ConSanRegisterPlan resources = plan_consan_registers(request, live);
  ASSERT_EQ(resources.source, ConSanRegisterAllocationSource::SpillRequired);
  ASSERT_TRUE(resources.base);
  const ConSanMoiAtomicAddressPlan plan = plan_consan_moi_atomic_address(
      site, *resources.base, /*scratch_vgpr_count=*/5, resources.source, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(plan.supported()) << consan_moi_atomic_address_support_name(plan.support);
  EXPECT_EQ(plan.resource_source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.result_address_vgpr, *resources.base + 3u);
}

TEST(ConSanMoi, AtomicAddressPlanFailsClosedForUnsupportedShapesAndAliases) {
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::Moi;
  const ConSanResult inventory =
      try_patch_consan(make_rdna4_global_atomic_code_object(), inventory_options);
  ASSERT_EQ(inventory.kernels.size(), 1u);
  ASSERT_EQ(inventory.kernels.front().atomic_sites.size(), 1u);
  const ConSanAtomicSite base = inventory.kernels.front().atomic_sites.front();
  const auto classify = [&](const ConSanAtomicSite &site, uint16_t scratch = 8u,
                            uint16_t count = 5u,
                            ConSanRegisterAllocationSource source =
                                ConSanRegisterAllocationSource::Explicit) {
    return plan_consan_moi_atomic_address(site, scratch, count, source, ROCJITSU_CODE_ARCH_RDNA4)
        .support;
  };

  ConSanAtomicSite changed = base;
  changed.raw_scope = 0u;
  EXPECT_EQ(classify(changed), ConSanMoiAtomicAddressSupport::UnsupportedScope);
  changed.raw_scope = 1u;
  EXPECT_EQ(classify(changed), ConSanMoiAtomicAddressSupport::Supported);
  changed.raw_scope = 3u;
  EXPECT_EQ(classify(changed), ConSanMoiAtomicAddressSupport::Supported);
  changed = base;
  changed.width_bits = 64u;
  EXPECT_EQ(classify(changed), ConSanMoiAtomicAddressSupport::Supported);
  changed.width_bits = 128u;
  EXPECT_EQ(classify(changed), ConSanMoiAtomicAddressSupport::UnsupportedWidth);
  changed = base;
  changed.raw_saddr = rdna4::OPR_SREG_NULL;
  changed.saddr_sgpr.reset();
  EXPECT_EQ(classify(changed), ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
  changed = base;
  changed.addr_vgpr = 255u;
  changed.raw_vaddr = 255u;
  EXPECT_EQ(classify(changed), ConSanMoiAtomicAddressSupport::Supported);
  changed = base;
  changed.raw_ioffset = 1 << 23;
  EXPECT_EQ(classify(changed), ConSanMoiAtomicAddressSupport::UnsupportedOffset);
  changed = base;
  changed.addr_vgpr = 11u;
  changed.raw_vaddr = 11u;
  EXPECT_EQ(classify(changed), ConSanMoiAtomicAddressSupport::ResultAddressAlias);
  EXPECT_EQ(classify(base, /*scratch=*/8, /*count=*/4),
            ConSanMoiAtomicAddressSupport::UnsupportedScratchShape);
  EXPECT_EQ(classify(base, /*scratch=*/0, /*count=*/5),
            ConSanMoiAtomicAddressSupport::ScratchOperandAlias);
  EXPECT_EQ(classify(base, /*scratch=*/8, /*count=*/5, ConSanRegisterAllocationSource::Unsupported),
            ConSanMoiAtomicAddressSupport::UnsupportedResourcePlan);

  const ConSanMoiAtomicAddressPlan plan = plan_consan_moi_atomic_address(
      base, /*scratch_vgpr=*/8, /*scratch_vgpr_count=*/5, ConSanRegisterAllocationSource::Explicit,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(plan.supported());
  // Save SGPRs may not overwrite the guest scalar address pair.
  EXPECT_FALSE(build_consan_moi_atomic_address_materialization(
      plan, /*vcc_save_sgpr=*/4, /*scc_save_sgpr=*/82, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(
      consan_moi_atomic_address_support_name(ConSanMoiAtomicAddressSupport::ScratchOperandAlias),
      "scratch-operand-alias");
}

TEST(ConSanMoi, SampledAtomicTrackingRequiresSelectedReadyCausalWindow) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(1);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.patches.empty());
  const ConSanMoiAutoReportInventory inventory =
      inventory_consan_moi_auto_report(result, options, bytes);
  EXPECT_EQ(inventory.atomic_event_count, 0u);
  EXPECT_EQ(inventory.access_range_count, 0u);
  EXPECT_EQ(inventory.sampled_range_bank_count, 0u);
  EXPECT_EQ(inventory.sampled_watchpoint_count, 0u);
  EXPECT_EQ(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                              &ConSanPatchInfo::kind),
            result.patches.end());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no selected LDS access candidates") != std::string::npos;
  }));
  EXPECT_EQ(std::ranges::count(result.site_dispositions, ConSanResourceSiteKind::Atomic,
                               &ConSanSiteDispositionRecord::site_kind),
            0u);
}

TEST(ConSanMoi, SampledAccessAndAtomicShareSelectedCausalSlot) {
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic);
  std::array<uint32_t, 520> words{};
  words[0] = 0xD8340000u;
  words[1] = 0; // ds_store_b32 v0, v0
  words[2] = 0xEE0B0000u;
  words[3] = 0;
  words[4] = 0; // global_wb
  words[5] = (*atomic)[0];
  words[6] = (*atomic)[1];
  words[7] = (*atomic)[2];
  for (size_t i = 8; i + 1u < words.size(); ++i)
    words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  constexpr uint64_t slot_bytes = direct_sampled_report_bytes(1) - sizeof(ConSanMoiReportHeader);
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 2u * slot_bytes;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto sampled_access =
      std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
               patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
      });
  ASSERT_NE(sampled_access, result.patches.end());
  EXPECT_EQ(sampled_access->sampled_access_kind, ConSanLdsAccessKind::Write);
  EXPECT_EQ(sampled_access->sampled_access_range_count, 1u);
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                              &ConSanPatchInfo::kind),
            result.patches.end());
}

TEST(ConSanMoi, FenceRecordPatchesCarryExactAtomicAddressIntoAbiV4Input) {
  constexpr uint16_t kExecSaveSgpr = 90u;
  constexpr uint16_t kExecSaveSgprCount = 7u;
  std::vector<uint16_t> live_sgprs;
  for (uint16_t sgpr = 0; sgpr <= 105u; ++sgpr) {
    if (sgpr < kExecSaveSgpr || sgpr >= kExecSaveSgpr + kExecSaveSgprCount)
      live_sgprs.push_back(sgpr);
  }
  const std::vector<uint8_t> bytes = make_rdna4_atomic_fence_sequence_code_object(live_sgprs);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.max_patches = 3;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = kExecSaveSgpr;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(3, 0, 0, 0, 0, 3, 3);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_sgpr);
  ASSERT_EQ(result.moi_fence_candidates.size(), 2u);
  EXPECT_TRUE(std::ranges::all_of(result.moi_fence_candidates, &ConSanMoiFenceCandidate::eligible));
  std::vector<const ConSanPatchInfo *> fences;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiFenceRecord)
      fences.push_back(&patch);
  }
  ASSERT_EQ(fences.size(), 2u);
  EXPECT_EQ(fences[0]->anchor_offset, result.moi_fence_candidates[0].text_offset);
  EXPECT_EQ(fences[1]->anchor_offset, result.moi_fence_candidates[1].text_offset);
  EXPECT_EQ(std::ranges::count(result.resource_plans, ConSanResourceSiteKind::Fence,
                               &ConSanCandidateResourcePlan::site_kind),
            2);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const ConSanMoiReportBufferLayout layout = consan_moi_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size, /*include_barriers=*/false,
      /*include_atomics=*/true, /*include_fences=*/true);
  ASSERT_EQ(layout.fence_record_capacity, 3u);
  for (size_t index = 0; index < fences.size(); ++index) {
    const ConSanPatchInfo &patch = *fences[index];
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    const uint64_t record = *options.moi_report_buffer_address + layout.fence_records_offset +
                            index * sizeof(ConSanMoiFenceRecord);
    const auto materialize_record = build_v_mov_b32_e64_literal(
        *options.scratch_vgpr, static_cast<uint32_t>(record), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(materialize_record);
    EXPECT_TRUE(contains_subsequence(words, *materialize_record));
    const auto expect_vgpr = [&](uint32_t offset, uint16_t value_vgpr) {
      EXPECT_TRUE(contains_subsequence(
          words, make_expected_offset_store_words(offset, value_vgpr, *options.scratch_vgpr)));
    };
    const auto expect_literal = [&](uint32_t offset, uint32_t value) {
      EXPECT_TRUE(
          contains_subsequence(words, make_expected_literal_offset_store_words(
                                          offset, value, *options.scratch_vgpr,
                                          static_cast<uint16_t>(*options.scratch_vgpr + 2u))));
    };
    expect_vgpr(offsetof(ConSanMoiFenceRecord, communication_token), 2);
    expect_vgpr(offsetof(ConSanMoiFenceRecord, communication_token) + sizeof(uint32_t), 3);
    expect_literal(offsetof(ConSanMoiFenceRecord, generation),
                   static_cast<uint32_t>(options.moi_report_dispatch_id));
    expect_literal(offsetof(ConSanMoiFenceRecord, generation) + sizeof(uint32_t),
                   static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u));
    EXPECT_TRUE(contains_subsequence(
        words,
        make_expected_literal_store_words(*options.moi_report_buffer_address +
                                              offsetof(ConSanMoiReportHeader, fence_record_count),
                                          2u, *options.scratch_vgpr)));
    expect_literal(offsetof(ConSanMoiFenceRecord, kind),
                   static_cast<uint32_t>(index == 0 ? ConSanMoiFenceEventKind::Release
                                                    : ConSanMoiFenceEventKind::Acquire));
    expect_literal(offsetof(ConSanMoiFenceRecord, scope), 2u);
  }
}

TEST(ConSanMoi, AtomicRecordCapturesVglobalCasThroughSharedAddressPlan) {
  for (const bool vector_only_address : {false, true}) {
    SCOPED_TRACE(vector_only_address ? "vector-only VGLOBAL address"
                                     : "scalar-plus-vector VGLOBAL address");
    const std::vector<uint8_t> bytes =
        make_rdna4_ordered_global_cas_code_object(/*return_old_value=*/true, vector_only_address);
    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.moi_track_atomics = true;
    options.scratch_vgpr = 8;
    options.moi_owner_vgpr = 15;
    options.moi_epoch_vgpr = 16;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
      return item.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
    });
    ASSERT_NE(patch, result.patches.end());
    const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
      return item.site_kind == ConSanResourceSiteKind::Atomic;
    });
    ASSERT_NE(plan, result.resource_plans.end());
    EXPECT_EQ(plan->scratch_vgpr_count, 7u);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
    const auto compare = build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(/*compare_vgpr=*/5),
                                                    /*old_value_vgpr=*/0, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(compare);
    EXPECT_NE(std::find(words.begin(), words.end(), *compare), words.end());
  }
}

TEST(ConSanMoi, InlineAtomicMixedTablePublishesReleaseAndPairScopedAcquireToken) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 2u);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
                          }),
            0);
  ASSERT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
                          }),
            2);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();

  const auto release_patch_it =
      std::find_if(result.patches.begin(), result.patches.end(), [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
               patch.anchor_offset == 3u * sizeof(uint32_t);
      });
  const auto acquire_patch_it =
      std::find_if(result.patches.begin(), result.patches.end(), [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
               patch.anchor_offset == 6u * sizeof(uint32_t);
      });
  ASSERT_NE(release_patch_it, result.patches.end());
  ASSERT_NE(acquire_patch_it, result.patches.end());
  ASSERT_TRUE(release_patch_it->relocated_guest_instruction_offset);
  ASSERT_TRUE(acquire_patch_it->relocated_guest_instruction_offset);

  std::vector<uint32_t> release_words(release_patch_it->trampoline_size / sizeof(uint32_t));
  std::memcpy(release_words.data(), text_section->data() + release_patch_it->trampoline_offset,
              release_patch_it->trampoline_size);
  std::vector<uint32_t> acquire_words(acquire_patch_it->trampoline_size / sizeof(uint32_t));
  std::memcpy(acquire_words.data(), text_section->data() + acquire_patch_it->trampoline_offset,
              acquire_patch_it->trampoline_size);

  const auto original_release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto original_acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(original_release);
  ASSERT_TRUE(original_acquire);
  ASSERT_GE(release_words.size(), original_release->size() + 1u);
  ASSERT_GE(acquire_words.size(), original_acquire->size() + 1u);
  const size_t release_guest_word = (*release_patch_it->relocated_guest_instruction_offset -
                                     release_patch_it->trampoline_offset) /
                                    sizeof(uint32_t);
  const size_t acquire_guest_word = (*acquire_patch_it->relocated_guest_instruction_offset -
                                     acquire_patch_it->trampoline_offset) /
                                    sizeof(uint32_t);
  ASSERT_GE(release_words.size(), release_guest_word + original_release->size() + 2u);
  ASSERT_GE(acquire_words.size(), acquire_guest_word + original_acquire->size() + 1u);
  EXPECT_TRUE(std::equal(original_release->begin(), original_release->end(),
                         release_words.begin() + release_guest_word));
  EXPECT_TRUE(std::equal(original_acquire->begin(), original_acquire->end(),
                         acquire_words.begin() + acquire_guest_word));
  const auto wait_loadcnt0 = instrumentation::build_s_wait_flat_load0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_loadcnt0);
  EXPECT_EQ(release_words[release_guest_word + original_release->size()], *wait_loadcnt0);
  const auto wait_storecnt0 = instrumentation::build_s_wait_flat_store0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_storecnt0);
  EXPECT_EQ(release_words[release_guest_word + original_release->size() + 1u], *wait_storecnt0);
  EXPECT_EQ(acquire_words[acquire_guest_word + original_acquire->size()], *wait_loadcnt0);
  const auto wait_global_load0 =
      instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto wait_global_store0 =
      instrumentation::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_global_load0);
  ASSERT_TRUE(wait_global_store0);
  EXPECT_GT(*release_patch_it->relocated_guest_instruction_offset,
            release_patch_it->trampoline_offset);
  EXPECT_GT(*acquire_patch_it->relocated_guest_instruction_offset,
            acquire_patch_it->trampoline_offset)
      << "stable release version capture must bracket the guest acquire";

  const uint16_t stable_release_address = static_cast<uint16_t>(*options.scratch_vgpr + 24u);
  const std::array<uint32_t, 2> retain_release_address = {
      build_v_mov_b32_e32(stable_release_address, vector_source_vgpr(/*guest low=*/2),
                          ROCJITSU_CODE_ARCH_RDNA4),
      build_v_mov_b32_e32(static_cast<uint16_t>(stable_release_address + 1u),
                          vector_source_vgpr(/*guest high=*/3), ROCJITSU_CODE_ARCH_RDNA4),
  };
  EXPECT_TRUE(contains_subsequence(release_words, retain_release_address))
      << "release publication must retain its object identity across predecessor import";

  const auto slot_stride = build_v_lshlrev_b32_e32(
      /*vdst=*/10, scalar_positive_inline_u32(5), /*vsrc1=*/10, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(slot_stride);
  EXPECT_EQ(std::count(release_words.begin(), release_words.end(), *slot_stride), 2)
      << "release publication imports and then replaces its 32-byte predecessor slot";
  EXPECT_EQ(std::count(acquire_words.begin(), acquire_words.end(), *slot_stride), 2)
      << "acquire pre-reads and validates its 32-byte release slot";
  const auto token_stride_eight = build_v_lshlrev_b32_e32(
      /*vdst=*/8, scalar_positive_inline_u32(3), /*vsrc1=*/29, ROCJITSU_CODE_ARCH_RDNA4);
  const auto token_stride_sixteen = build_v_lshlrev_b32_e32(
      /*vdst=*/27, scalar_positive_inline_u32(4), /*vsrc1=*/29, ROCJITSU_CODE_ARCH_RDNA4);
  const auto token_stride_thirty_two = build_v_lshlrev_b32_e32(
      /*vdst=*/29, scalar_positive_inline_u32(5), /*vsrc1=*/29, ROCJITSU_CODE_ARCH_RDNA4);
  const auto token_stride_forty_eight = build_v_add_nc_u32_e32(
      /*vdst=*/29, vector_source_vgpr(/*vsrc0=*/27), /*vsrc1=*/29, ROCJITSU_CODE_ARCH_RDNA4);
  const auto token_stride_fifty_six = build_v_add_nc_u32_e32(
      /*vdst=*/29, vector_source_vgpr(/*vsrc0=*/8), /*vsrc1=*/29, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(token_stride_eight);
  ASSERT_TRUE(token_stride_sixteen);
  ASSERT_TRUE(token_stride_thirty_two);
  ASSERT_TRUE(token_stride_forty_eight);
  ASSERT_TRUE(token_stride_fifty_six);
  EXPECT_TRUE(contains_subsequence(
      acquire_words,
      std::vector<uint32_t>{*token_stride_eight, *token_stride_sixteen, *token_stride_thirty_two,
                            *token_stride_forty_eight, *token_stride_fifty_six}))
      << "56-byte token slots require index*8 + index*16 + index*32";

  const ConSanMoiReportBufferLayout layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
  EXPECT_TRUE(contains_subsequence(
      release_words,
      make_expected_offset_store_words(offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                                       *options.moi_owner_vgpr, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      release_words,
      make_expected_offset_store_words(offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch_plus_one),
                                       /*value_vgpr=*/12, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      release_words,
      make_expected_offset_store_words(offsetof(ConSanMoiInlineAtomicReleaseSlot, workgroup_key),
                                       /*value_vgpr=*/11, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      release_words,
      make_expected_offset_store_words(offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                                       stable_release_address, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      release_words,
      make_expected_offset_store_words(
          offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) + sizeof(uint32_t),
          static_cast<uint16_t>(stable_release_address + 1u), *options.scratch_vgpr)));
  const auto release_claim_and_commit = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/13, /*vdst=*/13, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(release_claim_and_commit);
  EXPECT_EQ(count_subsequence(release_words, *release_claim_and_commit), 2u)
      << "release publication must use odd claim then even commit CAS";
  std::vector<uint32_t> drained_release_transaction(release_claim_and_commit->begin(),
                                                    release_claim_and_commit->end());
  drained_release_transaction.push_back(*wait_global_load0);
  drained_release_transaction.push_back(*wait_global_store0);
  EXPECT_EQ(count_subsequence(release_words, drained_release_transaction), 2u)
      << "every gfx12 release claim and commit must drain both sides of the returning CAS";
  const uint16_t release_retry_count_sgpr = 100u;
  const std::array<uint32_t, 2> initialize_release_retries = {
      build_s_mov_b32(release_retry_count_sgpr, /*literal source=*/255u, ROCJITSU_CODE_ARCH_RDNA4),
      4096u,
  };
  const auto decrement_release_retry =
      build_s_sub_u32(release_retry_count_sgpr, release_retry_count_sgpr,
                      scalar_positive_inline_u32(1), ROCJITSU_CODE_ARCH_RDNA4);
  const auto release_retries_remain = build_rdna4_s_cmp_lg_u32(
      release_retry_count_sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(decrement_release_retry);
  ASSERT_TRUE(release_retries_remain);
  EXPECT_TRUE(contains_subsequence(release_words, initialize_release_retries));
  EXPECT_NE(std::find(release_words.begin(), release_words.end(), *decrement_release_retry),
            release_words.end());
  EXPECT_NE(std::find(release_words.begin(), release_words.end(), *release_retries_remain),
            release_words.end());
  EXPECT_NE(std::find(release_words.begin(), release_words.end(),
                      build_s_sleep(/*delay=*/1, ROCJITSU_CODE_ARCH_RDNA4)),
            release_words.end());
  const auto coherent_version_load = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(coherent_version_load);
  std::vector<uint32_t> coherent_version_load_words = {
      build_v_mov_b32_e32(/*vdst=*/10, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4)};
  coherent_version_load_words.insert(coherent_version_load_words.end(),
                                     coherent_version_load->begin(), coherent_version_load->end());
  coherent_version_load_words.push_back(*wait_global_load0);
  EXPECT_TRUE(contains_subsequence(acquire_words, coherent_version_load_words));
  EXPECT_TRUE(contains_subsequence(
      acquire_words,
      make_expected_offset_load_words(offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                                      /*value_vgpr=*/10, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words,
      make_expected_offset_load_words(offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) +
                                          sizeof(uint32_t),
                                      /*value_vgpr=*/10, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words,
      make_expected_offset_load_words(offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                                      /*value_vgpr=*/12, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words,
      make_expected_offset_load_words(offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch_plus_one),
                                      /*value_vgpr=*/13, *options.scratch_vgpr)));
  EXPECT_EQ(layout.inline_atomic_release_capacity, 64u);
  EXPECT_EQ(layout.inline_acquired_epoch_token_capacity, 64u);
  EXPECT_TRUE(contains_subsequence(
      acquire_words, make_expected_offset_store_words(
                         offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
                         /*value_vgpr=*/11, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words, make_expected_offset_store_words(
                         offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
                         /*value_vgpr=*/40, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words, make_expected_offset_store_words(
                         offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
                         /*value_vgpr=*/12, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words, make_expected_offset_store_words(
                         offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one),
                         /*value_vgpr=*/13, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      acquire_words, make_expected_offset_store_words(
                         offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_epoch_plus_one),
                         /*value_vgpr=*/27, *options.scratch_vgpr)));
  const auto advance_consumer_segment =
      instrumentation::build_v_add_u32(*options.moi_epoch_vgpr, scalar_positive_inline_u32(1),
                                       *options.moi_epoch_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  const auto saturate_consumer_segment = instrumentation::build_v_min_u32_literal(
      *options.moi_epoch_vgpr, consan_moi_exact_shadow::max_epoch, *options.moi_epoch_vgpr,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(advance_consumer_segment);
  ASSERT_TRUE(saturate_consumer_segment);
  std::vector<uint32_t> consumer_segment_advance(advance_consumer_segment->begin(),
                                                 advance_consumer_segment->end());
  consumer_segment_advance.insert(consumer_segment_advance.end(),
                                  saturate_consumer_segment->begin(),
                                  saturate_consumer_segment->end());
  EXPECT_TRUE(contains_subsequence(acquire_words, consumer_segment_advance))
      << "a validated acquire must establish its consumer segment before token publication";
  const auto token_claim_or_update = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/27, /*vdst=*/27, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(token_claim_or_update);
  EXPECT_EQ(count_subsequence(acquire_words, *token_claim_or_update), 15u)
      << "five destinations each have reserve, rollback, and commit CAS bodies";

  const auto forbidden_epoch_import =
      build_v_add_nc_u32_e32(*options.moi_epoch_vgpr, scalar_positive_inline_u32(1), /*vsrc1=*/10,
                             ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_original_exec =
      build_s_mov_b64(kRdna4ExecLo, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 12u),
                      ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_scc = build_rdna4_s_cselect_b32(
      /*sdst=*/static_cast<uint16_t>(*options.moi_exec_save_sgpr + 10u),
      scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(
      /*sdst=*/static_cast<uint16_t>(*options.moi_exec_save_sgpr + 8u), kRdna4VccLo,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(
      kRdna4VccLo, /*ssrc0=*/static_cast<uint16_t>(*options.moi_exec_save_sgpr + 8u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_rdna4_s_cmp_lg_u32(
      /*ssrc0=*/static_cast<uint16_t>(*options.moi_exec_save_sgpr + 10u),
      scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(forbidden_epoch_import);
  ASSERT_TRUE(restore_original_exec);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  EXPECT_EQ(std::find(acquire_words.begin(), acquire_words.end(), *forbidden_epoch_import),
            acquire_words.end());
  EXPECT_NE(std::find(acquire_words.begin(), acquire_words.end(), *restore_original_exec),
            acquire_words.end());
  EXPECT_TRUE(contains_subsequence(acquire_words, std::array<uint32_t, 2>{*save_scc, *save_vcc}));
  EXPECT_TRUE(contains_subsequence(
      acquire_words, std::array<uint32_t, 3>{*restore_original_exec, *restore_vcc, *restore_scc}));
}

TEST(ConSanMoi, InlineShadowExactConflictUsesStableFullAcquiredToken) {
  const std::vector<uint8_t> bytes = make_rdna4_lds_and_ordered_flat_atomic_handoff_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 16;
  options.moi_owner_vgpr = 48;
  options.moi_epoch_vgpr = 49;
  options.moi_exec_save_sgpr = 40;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_report_generation = 0x123456789abcdef0ull;
  options.max_patches = 16;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            2u);
  const auto load_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore &&
           patch.anchor_offset == 14u * sizeof(uint32_t);
  });
  ASSERT_NE(load_patch, result.patches.end());
  EXPECT_EQ(load_patch->scratch_vgpr, options.scratch_vgpr);
  const auto load_plan = std::ranges::find_if(result.resource_plans, [](const auto &plan) {
    return plan.site_kind == ConSanResourceSiteKind::Access &&
           plan.text_offset == 14u * sizeof(uint32_t);
  });
  ASSERT_NE(load_plan, result.resource_plans.end());
  EXPECT_EQ(load_plan->scratch_vgpr_count, 24u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  const auto acquire_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
           patch.anchor_offset == 8u * sizeof(uint32_t);
  });
  ASSERT_NE(acquire_patch, result.patches.end());
  std::vector<uint32_t> acquire_words(acquire_patch->trampoline_size / sizeof(uint32_t));
  std::memcpy(acquire_words.data(), text_section->data() + acquire_patch->trampoline_offset,
              acquire_patch->trampoline_size);
  constexpr uint16_t kScalarInlineNegativeOneOperand = 193u;
  const auto widen_consumer_segment = instrumentation::build_s_mov_b64(
      kRdna4ExecLo, kScalarInlineNegativeOneOperand, ROCJITSU_CODE_ARCH_RDNA4);
  const auto advance_consumer_segment =
      instrumentation::build_v_add_u32(*options.moi_epoch_vgpr, scalar_positive_inline_u32(1),
                                       *options.moi_epoch_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  const auto saturate_consumer_segment = instrumentation::build_v_min_u32_literal(
      *options.moi_epoch_vgpr, consan_moi_exact_shadow::max_epoch, *options.moi_epoch_vgpr,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_validated_acquire_exec =
      instrumentation::build_s_mov_b64(static_cast<uint16_t>(*options.moi_exec_save_sgpr + 16u),
                                       kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_validated_acquire_exec = instrumentation::build_s_mov_b64(
      kRdna4ExecLo, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 16u),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(widen_consumer_segment && advance_consumer_segment && saturate_consumer_segment &&
              save_validated_acquire_exec && restore_validated_acquire_exec);
  std::vector<uint32_t> owner_wide_consumer_segment = {*widen_consumer_segment};
  owner_wide_consumer_segment.insert(owner_wide_consumer_segment.end(),
                                     advance_consumer_segment->begin(),
                                     advance_consumer_segment->end());
  owner_wide_consumer_segment.insert(owner_wide_consumer_segment.end(),
                                     saturate_consumer_segment->begin(),
                                     saturate_consumer_segment->end());
  owner_wide_consumer_segment.push_back(*restore_validated_acquire_exec);
  const auto skip_failed_acquire = instrumentation::build_s_cbranch_execz(
      static_cast<int16_t>(owner_wide_consumer_segment.size()), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(skip_failed_acquire);
  std::vector<uint32_t> guarded_owner_wide_consumer_segment = {*save_validated_acquire_exec,
                                                               *skip_failed_acquire};
  guarded_owner_wide_consumer_segment.insert(guarded_owner_wide_consumer_segment.end(),
                                             owner_wide_consumer_segment.begin(),
                                             owner_wide_consumer_segment.end());
  EXPECT_TRUE(contains_subsequence(acquire_words, guarded_owner_wide_consumer_segment))
      << "only a validated acquire may advance every lane of the wave owner";
  std::vector<uint32_t> words(load_patch->trampoline_size / sizeof(uint32_t));
  std::memcpy(words.data(), text_section->data() + load_patch->trampoline_offset,
              load_patch->trampoline_size);

  constexpr uint16_t scratch = 16;
  constexpr uint16_t current_field = scratch + 3u;
  constexpr uint16_t temporary = scratch + 4u;
  constexpr uint16_t exec = 40;
  const auto same_workgroup = build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(current_field),
                                                         temporary, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_same_workgroup =
      build_s_and_saveexec_b64(exec + 2u, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(same_workgroup);
  ASSERT_TRUE(narrow_same_workgroup);
  EXPECT_TRUE(contains_subsequence(
      words, std::array<uint32_t, 2>{*same_workgroup, *narrow_same_workgroup}));

  constexpr uint32_t low_generation_bits = 32u - consan_moi_exact_shadow::generation_shift;
  constexpr uint32_t high_generation_bits =
      consan_moi_exact_shadow::generation_bits - low_generation_bits;
  constexpr uint16_t retained_workgroup_key = scratch + 19u;
  const auto extract_high_first =
      build_v_and_b32_e32_literal(temporary, (uint32_t{1} << high_generation_bits) - 1u,
                                  current_field, ROCJITSU_CODE_ARCH_RDNA4);
  const auto extract_low_after = build_v_lshrrev_b32_e32(
      retained_workgroup_key, scalar_positive_inline_u32(consan_moi_exact_shadow::generation_shift),
      /*current low=*/scratch + 2u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto shift_high =
      build_v_lshlrev_b32_e32(temporary, scalar_positive_inline_u32(low_generation_bits), temporary,
                              ROCJITSU_CODE_ARCH_RDNA4);
  const auto combine_key =
      build_v_add_nc_u32_e32(retained_workgroup_key, vector_source_vgpr(retained_workgroup_key),
                             temporary, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(extract_high_first);
  ASSERT_TRUE(extract_low_after);
  ASSERT_TRUE(shift_high);
  ASSERT_TRUE(combine_key);
  std::vector<uint32_t> retained_key_extract(extract_high_first->begin(),
                                             extract_high_first->end());
  retained_key_extract.push_back(*extract_low_after);
  retained_key_extract.push_back(*shift_high);
  retained_key_extract.push_back(*combine_key);
  const auto retained_key_position = std::search(
      words.begin(), words.end(), retained_key_extract.begin(), retained_key_extract.end());
  const auto same_workgroup_position = std::ranges::find(words, *same_workgroup);
  ASSERT_NE(retained_key_position, words.end());
  ASSERT_NE(same_workgroup_position, words.end());
  EXPECT_LT(retained_key_position, same_workgroup_position)
      << "token lookup must retain the workgroup key after packed metadata is reused";

  const auto save_valid_exec = build_s_mov_b64(exec, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto select_unsupported = build_s_and_not1_b64(
      kRdna4ExecLo, exec + kConSanMoiInlineOriginalExecSaveOffset, exec, ROCJITSU_CODE_ARCH_RDNA4);
  const uint64_t unsupported_count_address =
      *options.moi_report_buffer_address +
      offsetof(ConSanMoiReportHeader, inline_unsupported_count);
  const auto unsupported_address_lo = build_v_mov_b32_e64_literal(
      scratch, static_cast<uint32_t>(unsupported_count_address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto unsupported_address_hi = build_v_mov_b32_e64_literal(
      scratch + 1u, static_cast<uint32_t>(unsupported_count_address >> 32u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto unsupported_one = build_v_mov_b32_e64_literal(temporary, 1u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_unsupported = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      scratch, temporary, temporary, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto unsupported_wait =
      instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto unsupported_store_wait =
      instrumentation::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_valid_exec = build_s_mov_b64(kRdna4ExecLo, exec, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_valid_exec);
  ASSERT_TRUE(select_unsupported);
  ASSERT_TRUE(unsupported_address_lo);
  ASSERT_TRUE(unsupported_address_hi);
  ASSERT_TRUE(unsupported_one);
  ASSERT_TRUE(count_unsupported);
  ASSERT_TRUE(unsupported_wait);
  ASSERT_TRUE(unsupported_store_wait);
  ASSERT_TRUE(restore_valid_exec);
  std::vector<uint32_t> expected_unsupported_accounting = {*save_valid_exec, *select_unsupported};
  expected_unsupported_accounting.insert(expected_unsupported_accounting.end(),
                                         unsupported_address_lo->begin(),
                                         unsupported_address_lo->end());
  expected_unsupported_accounting.insert(expected_unsupported_accounting.end(),
                                         unsupported_address_hi->begin(),
                                         unsupported_address_hi->end());
  expected_unsupported_accounting.insert(expected_unsupported_accounting.end(),
                                         unsupported_one->begin(), unsupported_one->end());
  expected_unsupported_accounting.insert(expected_unsupported_accounting.end(),
                                         count_unsupported->begin(), count_unsupported->end());
  expected_unsupported_accounting.push_back(*unsupported_wait);
  expected_unsupported_accounting.push_back(*unsupported_store_wait);
  expected_unsupported_accounting.push_back(*restore_valid_exec);
  EXPECT_TRUE(contains_subsequence(words, expected_unsupported_accounting));

  const auto disable_legacy_token_authority =
      build_s_mov_b64(kRdna4ExecLo, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(disable_legacy_token_authority);
  EXPECT_EQ(std::find(words.begin(), words.end(), *disable_legacy_token_authority), words.end())
      << "the stable full-token reader replaces unconditional legacy suppression disablement";

  // The access-time token reader runs inside the wave-coalesced address
  // partition loop. scratch+7..scratch+12 are the loop's live saved address,
  // packed current value, and exact-byte provenance, so token fields must use
  // the diagnostic-only tail instead of corrupting the next partition.
  for (const auto &[offset, destination] : std::array<std::pair<size_t, uint16_t>, 12>{
           {{offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key), temporary},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id), temporary},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id), temporary},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one),
             scratch + 17u},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind), scratch + 16u},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id), temporary},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) + sizeof(uint32_t),
             temporary},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address),
             scratch + 21u},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address) +
                 sizeof(uint32_t),
             scratch + 22u},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version),
             scratch + 18u},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_epoch_plus_one),
             scratch + 14u},
            {offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version), temporary}}}) {
    EXPECT_TRUE(
        contains_subsequence(words, make_expected_offset_load_words(offset, destination, scratch)))
        << "missing acquired-token field at offset " << offset;
  }

  const auto coherent_token_version_load = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/scratch, /*vsrc=*/scratch + 15u, /*vdst=*/scratch + 15u,
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto coherent_token_version_reread = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/scratch, /*vsrc=*/scratch + 14u, /*vdst=*/scratch + 14u,
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto token_version_wait =
      instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(coherent_token_version_load);
  ASSERT_TRUE(coherent_token_version_reread);
  ASSERT_TRUE(token_version_wait);
  for (const auto &[load, destination] :
       std::array<std::pair<std::array<uint32_t, 3>, uint16_t>, 2>{
           {{*coherent_token_version_load, scratch + 15u},
            {*coherent_token_version_reread, scratch + 14u}}}) {
    std::vector<uint32_t> expected = {
        build_v_mov_b32_e32(destination, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4)};
    expected.insert(expected.end(), load.begin(), load.end());
    expected.push_back(*token_version_wait);
    EXPECT_TRUE(contains_subsequence(words, expected))
        << "acquired-token version snapshots must use coherent device-scope loads";
    EXPECT_EQ(count_subsequence(words, expected), 4u)
        << "both possible cells and both happens-before directions require stable token snapshots";
  }

  const auto authorize_stable_access_token =
      build_v_cmp_gt_u32_e32_vcc(scalar_positive_inline_u32(static_cast<uint32_t>(
                                     ConSanMoiInlineTokenEvidenceKind::ReleaseSequence)),
                                 scratch + 16u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(authorize_stable_access_token);
  EXPECT_NE(std::ranges::find(words.begin(), words.end(), *authorize_stable_access_token),
            words.end())
      << "stable direct and inherited tokens must authorize ordinary accesses";

  const auto remove_insufficient =
      build_s_and_not1_b64(kRdna4ExecLo, kRdna4ExecLo, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto remove_ordered =
      build_s_and_not1_b64(kRdna4ExecLo, exec + 2u, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_original_exec = build_s_mov_b64(
      kRdna4ExecLo, exec + kConSanMoiInlineOriginalExecSaveOffset, ROCJITSU_CODE_ARCH_RDNA4);
  const auto forbidden_vcc_clobber =
      build_s_and_saveexec_b64(exec + 8u, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, exec + 8u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc =
      build_rdna4_s_cmp_lg_u32(exec + 10u, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(remove_insufficient);
  ASSERT_TRUE(remove_ordered);
  ASSERT_TRUE(restore_original_exec);
  ASSERT_TRUE(forbidden_vcc_clobber);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  EXPECT_NE(std::ranges::find(words, *remove_insufficient), words.end());
  EXPECT_NE(std::ranges::find(words, *remove_ordered), words.end());
  EXPECT_NE(std::ranges::find(words, *restore_original_exec), words.end());
  EXPECT_EQ(std::ranges::find(words, *forbidden_vcc_clobber), words.end());
  EXPECT_TRUE(contains_subsequence(words, std::array<uint32_t, 2>{*restore_vcc, *restore_scc}));
}

TEST(ConSanMoi, InlineAtomicScalarPersistentAcquireGuardsEpochAdvanceAndPersist) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_persistent_owner_sgpr = 70;
  options.moi_persistent_epoch_sgpr = 71;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "warnings=" << testing::PrintToString(result.warnings)
      << " errors=" << testing::PrintToString(result.errors)
      << " resources=" << testing::PrintToString(result.resource_plans);
  ASSERT_TRUE(result.modified);
  const auto acquire_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
           patch.anchor_offset == 6u * sizeof(uint32_t);
  });
  ASSERT_NE(acquire_patch, result.patches.end());
  ASSERT_TRUE(acquire_patch->scratch_vgpr);
  const auto acquire_plan = std::ranges::find_if(result.resource_plans, [&](const auto &plan) {
    return plan.site_kind == ConSanResourceSiteKind::Atomic &&
           plan.text_offset == acquire_patch->anchor_offset &&
           plan.scratch_vgpr == acquire_patch->scratch_vgpr;
  });
  ASSERT_NE(acquire_plan, result.resource_plans.end());
  ASSERT_GE(acquire_plan->scratch_vgpr_count, 4u);
  const uint16_t materialized_epoch =
      static_cast<uint16_t>(*acquire_patch->scratch_vgpr + acquire_plan->scratch_vgpr_count - 3u);
  EXPECT_EQ(result.resolved_moi_persistent_owner_sgpr, options.moi_persistent_owner_sgpr);
  EXPECT_EQ(result.resolved_moi_persistent_epoch_sgpr, options.moi_persistent_epoch_sgpr);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> acquire_words = text_words_at_offset(
      patched, acquire_patch->trampoline_offset, acquire_patch->trampoline_size);

  const auto advance_consumer_segment =
      instrumentation::build_v_add_u32(materialized_epoch, scalar_positive_inline_u32(1),
                                       materialized_epoch, ROCJITSU_CODE_ARCH_RDNA4);
  const auto saturate_consumer_segment = instrumentation::build_v_min_u32_literal(
      materialized_epoch, consan_moi_exact_shadow::max_epoch, materialized_epoch,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto persist_consumer_segment = instrumentation::build_v_readfirstlane_b32(
      *options.moi_persistent_epoch_sgpr, materialized_epoch, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(advance_consumer_segment && saturate_consumer_segment && persist_consumer_segment);
  std::vector<uint32_t> scalar_consumer_segment(advance_consumer_segment->begin(),
                                                advance_consumer_segment->end());
  scalar_consumer_segment.insert(scalar_consumer_segment.end(), saturate_consumer_segment->begin(),
                                 saturate_consumer_segment->end());
  scalar_consumer_segment.push_back(*persist_consumer_segment);
  const auto skip_failed_acquire = instrumentation::build_s_cbranch_execz(
      static_cast<int16_t>(scalar_consumer_segment.size()), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(skip_failed_acquire);
  scalar_consumer_segment.insert(scalar_consumer_segment.begin(), *skip_failed_acquire);
  EXPECT_TRUE(contains_subsequence(acquire_words, scalar_consumer_segment))
      << "an empty validated acquire mask must skip the scalar epoch advance and persist";
}

TEST(ConSanMoi, InlineAtomicRetainsDisplacedVglobalAcquireAndPublishesToken) {
  const std::vector<uint8_t> bytes =
      make_rdna4_displaced_vglobal_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 12;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("pruned isolated no-return release metadata") != std::string::npos;
  }));
  ASSERT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            2)
      << testing::PrintToString(result.warnings)
      << " patches=" << testing::PrintToString(result.patches)
      << " resources=" << testing::PrintToString(result.resource_plans);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  EXPECT_EQ(result.resource_plans[0].scratch_vgpr_count, 26u);
  EXPECT_EQ(result.resource_plans[1].scratch_vgpr_count, 26u);

  const auto acquire_patch = std::ranges::find_if(result.patches, [](const auto &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
           patch.anchor_offset == 4u * sizeof(uint32_t);
  });
  ASSERT_NE(acquire_patch, result.patches.end());
  ASSERT_TRUE(acquire_patch->relocated_guest_instruction_offset);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> acquire_words = text_words_at_offset(
      patched, acquire_patch->trampoline_offset, acquire_patch->trampoline_size);

  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 2u);
  const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
      result.kernels.front().atomic_sites[1], /*scratch_vgpr=*/12,
      /*scratch_vgpr_count=*/26, ConSanRegisterAllocationSource::Explicit,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(address_plan.supported());
  const auto address_words = build_consan_moi_atomic_address_materialization(
      address_plan, /*vcc_save_sgpr=*/88, /*scc_save_sgpr=*/90, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(address_words);
  ASSERT_FALSE(address_words->empty());
  EXPECT_TRUE(std::equal(address_words->begin(), address_words->end(), acquire_words.begin()));
  EXPECT_GT(*acquire_patch->relocated_guest_instruction_offset,
            acquire_patch->trampoline_offset + address_words->size() * sizeof(uint32_t))
      << "pre-guest release version capture follows address materialization";

  const auto forbidden_epoch_import = build_v_add_nc_u32_e32(
      /*vdst=*/29, scalar_positive_inline_u32(1), /*vsrc1=*/14, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(forbidden_epoch_import);
  EXPECT_EQ(std::find(acquire_words.begin(), acquire_words.end(), *forbidden_epoch_import),
            acquire_words.end());
  EXPECT_TRUE(contains_subsequence(
      acquire_words, make_expected_offset_store_words(
                         offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
                         /*value_vgpr=*/40, /*address_vgpr=*/12)));

  const auto token_transaction_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/12, /*vsrc=*/31, /*vdst=*/31, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto token_load_wait = instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto token_store_wait =
      instrumentation::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(token_transaction_cas);
  ASSERT_TRUE(token_load_wait);
  ASSERT_TRUE(token_store_wait);
  std::vector<uint32_t> drained_token_transaction(token_transaction_cas->begin(),
                                                  token_transaction_cas->end());
  drained_token_transaction.push_back(*token_load_wait);
  drained_token_transaction.push_back(*token_store_wait);
  const size_t token_transaction_count = count_subsequence(acquire_words, *token_transaction_cas);
  ASSERT_GT(token_transaction_count, 0u);
  EXPECT_EQ(count_subsequence(acquire_words, drained_token_transaction), token_transaction_count)
      << "every gfx12 token claim, rollback, and commit must drain both sides of "
         "the returning FLAT atomic";
}

TEST(ConSanMoi, InlineAtomicRetainsIsolatedNoReturnReleaseAndExactShadowAccess) {
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_store_and_release_wait_no_return_bitwise_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            1);
  const auto exact_shadow = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  ASSERT_NE(exact_shadow, result.patches.end());
  EXPECT_EQ(exact_shadow->anchor_offset, 0u);
  EXPECT_EQ(exact_shadow->original_size, 2u * sizeof(uint32_t));
  EXPECT_GT(exact_shadow->trampoline_size, 0u);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("pruned isolated no-return release metadata") != std::string::npos;
  }));
}

TEST(ConSanMoi, InlineVglobalReleaseMaterializesAddressWithTransactionPlan) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_global_atomic_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->scratch_vgpr, 8u);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->scratch_vgpr_count, 26u);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
      result.kernels.front().atomic_sites.front(), /*scratch_vgpr=*/8,
      /*scratch_vgpr_count=*/26, ConSanRegisterAllocationSource::Explicit,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(address_plan.supported());
  EXPECT_EQ(address_plan.kind, ConSanMoiAtomicAddressKind::VglobalMaterialized);
  EXPECT_EQ(address_plan.result_address_vgpr, 32u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto address_words = build_consan_moi_atomic_address_materialization(
      address_plan, /*vcc_save_sgpr=*/88, /*scc_save_sgpr=*/90, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(address_words);
  ASSERT_FALSE(address_words->empty());
  EXPECT_TRUE(std::equal(address_words->begin(), address_words->end(), cave_words.begin()));
}

TEST(ConSanMoi, InlineAtomicReturningCasPublishesOnlyOnSuccess) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_cas_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  EXPECT_EQ(result.sync_sequences.front().operation, ConSanSyncOperation::AtomicCompareExchange);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
  });
  ASSERT_NE(patch, result.patches.end());
  const auto resource_plan =
      std::ranges::find_if(result.resource_plans, [](const ConSanCandidateResourcePlan &item) {
        return item.site_kind == ConSanResourceSiteKind::Atomic;
      });
  ASSERT_NE(resource_plan, result.resource_plans.end());
  EXPECT_EQ(resource_plan->scratch_vgpr_count, 26u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto success = build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(/*compare_vgpr=*/2),
                                                  /*old_value_vgpr=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow = build_s_and_saveexec_b64(/*sdst=*/80, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(success);
  ASSERT_TRUE(narrow);
  const std::array<uint32_t, 2> success_sequence = {*success, *narrow};
  EXPECT_TRUE(contains_subsequence(cave_words, success_sequence));
  const auto release_claim_and_commit = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/13, /*vdst=*/13, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(release_claim_and_commit);
  EXPECT_EQ(count_subsequence(cave_words, *release_claim_and_commit), 2u)
      << "successful release CAS lanes must claim odd and commit even";
  const auto success_position = std::search(cave_words.begin(), cave_words.end(),
                                            success_sequence.begin(), success_sequence.end());
  const auto claim_position =
      std::search(cave_words.begin(), cave_words.end(), release_claim_and_commit->begin(),
                  release_claim_and_commit->end());
  ASSERT_NE(success_position, cave_words.end());
  ASSERT_NE(claim_position, cave_words.end());
  EXPECT_LT(success_position, claim_position)
      << "dynamic CAS success must narrow EXEC before the release claim";
}

TEST(ConSanMoi, InlineVglobalReturningCasPublishesTokenBeforeSuccessPredicatedRelease) {
  for (const bool vector_only_address : {false, true}) {
    SCOPED_TRACE(vector_only_address ? "vector-only VGLOBAL address"
                                     : "scalar-plus-vector VGLOBAL address");
    const std::vector<uint8_t> bytes = make_rdna4_ordered_global_cas_code_object(
        /*return_old_value=*/true, vector_only_address);
    ASSERT_FALSE(bytes.empty());
    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_track_atomics = true;
    options.scratch_vgpr = 8;
    options.moi_exec_save_sgpr = 80;
    options.moi_owner_vgpr = 40;
    options.moi_epoch_vgpr = 41;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.sync_sequences.size(), 1u);
    EXPECT_EQ(result.sync_sequences.front().operation, ConSanSyncOperation::AtomicCompareExchange);
    EXPECT_EQ(result.sync_sequences.front().memory_role, ConSanSyncMemoryRole::AcquireRelease);
    ASSERT_EQ(result.kernels.size(), 1u);
    ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
    const ConSanAtomicSite &site = result.kernels.front().atomic_sites.front();
    EXPECT_EQ(
        classify_consan_moi_inline_atomic_support(site, ConSanMoiAtomicEventKind::AcquireRelease),
        ConSanMoiInlineAtomicSupport::Supported);

    const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
      return item.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
    });
    ASSERT_NE(patch, result.patches.end());
    const auto resource_plan =
        std::ranges::find_if(result.resource_plans, [](const ConSanCandidateResourcePlan &item) {
          return item.site_kind == ConSanResourceSiteKind::Atomic;
        });
    ASSERT_NE(resource_plan, result.resource_plans.end());
    EXPECT_EQ(resource_plan->scratch_vgpr_count, 26u);

    const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
        site, /*scratch_vgpr=*/8, resource_plan->scratch_vgpr_count,
        ConSanRegisterAllocationSource::Explicit, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(address_plan.supported())
        << consan_moi_atomic_address_support_name(address_plan.support);
    EXPECT_EQ(address_plan.kind, vector_only_address
                                     ? ConSanMoiAtomicAddressKind::VglobalGuestPair
                                     : ConSanMoiAtomicAddressKind::VglobalMaterialized);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    const std::vector<uint32_t> cave_words =
        text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);

    const std::vector<uint32_t> token_epoch_store = make_expected_offset_store_words(
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one),
        /*value_vgpr=*/13, /*address_vgpr=*/8);
    const auto restore_acquire_exec =
        build_s_mov_b64(kRdna4ExecLo, /*ssrc0=*/92, ROCJITSU_CODE_ARCH_RDNA4);
    const auto success = build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(/*compare_vgpr=*/5),
                                                    /*old_value_vgpr=*/0, ROCJITSU_CODE_ARCH_RDNA4);
    const auto narrow_release =
        build_s_and_saveexec_b64(/*sdst=*/80, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(restore_acquire_exec);
    ASSERT_TRUE(success);
    ASSERT_TRUE(narrow_release);
    const std::array<uint32_t, 2> success_sequence = {*success, *narrow_release};
    const auto token_position = std::search(cave_words.begin(), cave_words.end(),
                                            token_epoch_store.begin(), token_epoch_store.end());
    const auto restore_position = std::find(token_position + token_epoch_store.size(),
                                            cave_words.end(), *restore_acquire_exec);
    const auto success_position = std::search(cave_words.begin(), cave_words.end(),
                                              success_sequence.begin(), success_sequence.end());
    ASSERT_NE(token_position, cave_words.end());
    ASSERT_NE(restore_position, cave_words.end());
    ASSERT_NE(success_position, cave_words.end());
    EXPECT_LT(token_position, restore_position);
    EXPECT_LT(restore_position, success_position)
        << "acquire token publication must complete before CAS success narrows release "
           "publication";

    const std::vector<uint32_t> owner_store = make_expected_offset_store_words(
        offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id), /*value_vgpr=*/40,
        /*address_vgpr=*/8);
    const auto publication_position =
        std::search(success_position, cave_words.end(), owner_store.begin(), owner_store.end());
    ASSERT_NE(publication_position, cave_words.end());
    EXPECT_LT(success_position, publication_position);
    const auto restore_release_exec = std::find(publication_position + owner_store.size(),
                                                cave_words.end(), *restore_acquire_exec);
    ASSERT_NE(restore_release_exec, cave_words.end());
    EXPECT_LT(publication_position, restore_release_exec)
        << "failed CAS lanes must resume after success-predicated publication";

    const auto release_claim_and_commit = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
        /*vaddr=*/8, /*vsrc=*/13, /*vdst=*/13, /*return_old_value=*/true,
        /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(release_claim_and_commit);
    EXPECT_EQ(count_subsequence(cave_words, *release_claim_and_commit), 2u)
        << "AcquireRelease CAS must use one odd/even release transaction";
    const auto claim_position =
        std::search(success_position, cave_words.end(), release_claim_and_commit->begin(),
                    release_claim_and_commit->end());
    ASSERT_NE(claim_position, cave_words.end());
    EXPECT_LT(success_position, claim_position);
    EXPECT_LT(claim_position, publication_position)
        << "the version claim must precede causal snapshot and release metadata staging";
  }
}

TEST(ConSanMoi, InlineVglobalNoReturnCasFailsClosedWithoutOutcome) {
  for (const bool vector_only_address : {false, true}) {
    SCOPED_TRACE(vector_only_address ? "vector-only VGLOBAL address"
                                     : "scalar-plus-vector VGLOBAL address");
    const std::vector<uint8_t> bytes = make_rdna4_ordered_global_cas_code_object(
        /*return_old_value=*/false, vector_only_address);
    ASSERT_FALSE(bytes.empty());
    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_track_atomics = true;
    options.scratch_vgpr = 8;
    options.moi_exec_save_sgpr = 80;
    options.moi_owner_vgpr = 13;
    options.moi_epoch_vgpr = 14;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    EXPECT_FALSE(result.modified);
    EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
      return warning.find("compare-exchange-outcome-unavailable") != std::string::npos;
    })) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.patches.empty());
  }
}

TEST(ConSanMoi, InlineAtomicNoReturnCasFailsClosedWithoutOutcome) {
  const std::vector<uint8_t> bytes =
      make_rdna4_ordered_flat_cas_code_object(/*return_old_value=*/false);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("compare-exchange-outcome-unavailable") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.patches.empty());
}

TEST(ConSanMoi, InlineAtomicOrderingAutomaticallyPlansAllRegisterState) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::TrampolineMoiInlineAtomicOrdering;
                                  }),
            2u);
  EXPECT_EQ(std::ranges::count_if(result.resource_plans,
                                  [](const auto &plan) {
                                    return plan.site_kind == ConSanResourceSiteKind::Atomic &&
                                           plan.scratch_vgpr;
                                  }),
            2u);
}

TEST(ConSanMoi, InlineAtomicUsesAutomaticScalarSpillAtFullScalarPressure) {
  const auto release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(release && acquire);
  std::vector<uint32_t> text_words(800, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  size_t cursor = 0;
  // Keep sparse scalar values live from the access through both atomics. No
  // complete Inline scalar window is dead, so the router must spill its state.
  text_words[cursor++] = build_s_mov_b32(0, 105u, ROCJITSU_CODE_ARCH_RDNA4);
  constexpr std::array<uint16_t, 6> kLiveSgprs = {15u, 31u, 47u, 63u, 79u, 95u};
  for (uint16_t sgpr : kLiveSgprs)
    text_words[cursor++] =
        build_s_mov_b32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const size_t lds_access_word = cursor;
  text_words[cursor++] = 0xD8340000u;
  text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0
  constexpr std::array<uint32_t, 3> kGlobalWb = {0xEE0B0000u, 0x00000000u, 0x00000000u};
  std::copy(kGlobalWb.begin(), kGlobalWb.end(), text_words.begin() + cursor);
  cursor += kGlobalWb.size();
  std::copy(release->begin(), release->end(), text_words.begin() + cursor);
  cursor += release->size();
  std::copy(acquire->begin(), acquire->end(), text_words.begin() + cursor);
  cursor += acquire->size();
  constexpr std::array<uint32_t, 3> kGlobalInv = {0xEE0AC000u, 0x00000000u, 0x00000000u};
  std::copy(kGlobalInv.begin(), kGlobalInv.end(), text_words.begin() + cursor);
  cursor += kGlobalInv.size();
  for (uint16_t sgpr : kLiveSgprs)
    text_words[cursor++] = build_s_mov_b32(0, sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "inline_atomic_scalar_spill");

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 3u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("automatically assigned spill-backed Inline SGPRs") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            2u);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiInlineAtomicOrdering)
      continue;
    ASSERT_TRUE(patch.scratch_vgpr);
    EXPECT_NE(patch.required_private_segment_size, 0u);
    const std::vector<uint32_t> cave =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    // This fixture starts with no private segment, so the shared scalar spill
    // allocation begins at byte zero for both mutually exclusive trampolines.
    const auto first_store = instrumentation::build_private_store_b32(
        *patch.scratch_vgpr, /*byte_offset=*/0u, ROCJITSU_CODE_ARCH_RDNA4);
    const auto first_load = instrumentation::build_private_load_b32(
        *patch.scratch_vgpr, /*byte_offset=*/0u, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(first_store && first_load);
    EXPECT_TRUE(contains_subsequence(cave, *first_store));
    EXPECT_TRUE(contains_subsequence(cave, *first_load));
    EXPECT_NE(std::ranges::find(cave, build_v_mov_b32_e32(*patch.scratch_vgpr,
                                                          *result.resolved_moi_exec_save_sgpr,
                                                          ROCJITSU_CODE_ARCH_RDNA4)),
              cave.end());
  }

  std::vector<uint32_t> atomic_only_words = text_words;
  atomic_only_words[lds_access_word] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  atomic_only_words[lds_access_word + 1u] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions atomic_only_options = options;
  atomic_only_options.scratch_vgpr = 82u;
  atomic_only_options.moi_owner_vgpr = 80u;
  atomic_only_options.moi_epoch_vgpr = 81u;
  atomic_only_options.moi_exec_save_sgpr = 4u;
  atomic_only_options.automatic_moi_exec_save_sgprs = true;
  atomic_only_options.automatic_moi_inline_sgpr_spill = true;
  atomic_only_options.moi_inline_visible_evidence_sgpr = 40u;
  atomic_only_options.moi_inline_dispatch_key_sgpr = 41u;
  atomic_only_options.moi_inline_indirect_pc_sgpr = 42u;
  atomic_only_options.moi_inline_call_return_sgpr = 44u;
  atomic_only_options.moi_inline_indirect_scc_sgpr = 46u;
  const ConSanResult atomic_only = try_patch_consan(
      make_rdna4_lds_code_object(atomic_only_words, "inline_atomic_only_scalar_spill"),
      atomic_only_options);

  ASSERT_TRUE(consan_patch_succeeded(atomic_only)) << testing::PrintToString(atomic_only.errors);
  ASSERT_TRUE(atomic_only.modified) << testing::PrintToString(atomic_only.warnings);
  EXPECT_TRUE(atomic_only.final_validation_passed);
  ASSERT_EQ(std::ranges::count(atomic_only.patches,
                               ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            2u);
  for (const ConSanPatchInfo &patch : atomic_only.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiInlineAtomicOrdering)
      continue;
    // Atomic-only Inline probes use scalar slots +0..+21. The access layout
    // through +29 must not inflate their private spill window.
    EXPECT_EQ(patch.required_private_segment_size, 22u * sizeof(uint32_t));
  }
}

TEST(ConSanMoi, InlineAtomicScalarSpillRejectsAliasedGuestScalarAddress) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_global_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 82u;
  options.moi_owner_vgpr = 80u;
  options.moi_epoch_vgpr = 81u;
  // VGLOBAL reads its address from s[4:5]. An automatic scalar spill may
  // surround a vector-only FLAT atomic, but it must not clobber this pair.
  options.moi_exec_save_sgpr = 4u;
  options.automatic_moi_exec_save_sgprs = true;
  options.automatic_moi_inline_sgpr_spill = true;
  options.moi_inline_visible_evidence_sgpr = 40u;
  options.moi_inline_dispatch_key_sgpr = 41u;
  options.moi_inline_indirect_pc_sgpr = 42u;
  options.moi_inline_call_return_sgpr = 44u;
  options.moi_inline_indirect_scc_sgpr = 46u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2u;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_EQ(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                              &ConSanPatchInfo::kind),
            result.patches.end());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("spill-backed scalar window aliases a guest atomic address input") !=
           std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Gfx1250InlineAtomicOrdersReleaseAndAcquire) {
  const auto release = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_GFX1250);
  const auto acquire = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/4, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(release && acquire);
  const std::array<uint32_t, 13> text_words = {
      0xEE0B0000u,   0x00000000u,   0x00000000u, // global_wb
      (*release)[0], (*release)[1], (*release)[2], (*acquire)[0], (*acquire)[1],
      (*acquire)[2], 0xEE0AC000u,   0x00000000u,   0x00000000u, // global_inv
      0xBFB00000u,                                              // s_endpgm
  };
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "gfx1250_inline_atomic"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 2u);
  EXPECT_TRUE(
      std::ranges::all_of(result.kernels.front().atomic_sites, [](const ConSanAtomicSite &site) {
        return site.raw_saddr == static_cast<uint32_t>(gfx1250::OPR_SREG_NULL) &&
               site.raw_scope == 2u && site.raw_ioffset == 0;
      }));
  EXPECT_TRUE(std::ranges::any_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind == ConSanResourceSiteKind::Atomic && plan.scratch_vgpr_count == 26u;
  }));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            2u);
}

TEST(ConSanMoi, InlineAtomicOnlyObjectOmitsUnusedWorkgroupFilterSgprs) {
  const std::vector<uint8_t> bytes =
      make_rdna4_ordered_flat_atomic_high_sgpr_pressure_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_EQ(*result.resolved_moi_dispatch_id_sgpr, 82u);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  EXPECT_EQ(*result.resolved_moi_exec_save_sgpr, 84u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_NE(std::ranges::find(
                result.warnings,
                "ConSan MOI inline shadow omitted unused access-only workgroup-filter state"),
            result.warnings.end());
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("could not place a fresh automatic EXEC-save SGPR window") !=
           std::string::npos;
  }));
}

TEST(ConSanMoi, InlineAtomicFitsAboveMetadataOwnedOddSgprCount) {
  std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_high_sgpr_pressure_code_object();
  ASSERT_FALSE(bytes.empty());
  append_kernel_metadata_note(bytes, "atomic_high_sgpr_pressure",
                              /*uses_dynamic_stack=*/true, /*sgpr_count=*/81u);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.kernels().size(), 1u);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_EQ(*result.resolved_moi_dispatch_id_sgpr, 82u);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  EXPECT_EQ(*result.resolved_moi_exec_save_sgpr, 84u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            2u);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->original_size, 3u * sizeof(uint32_t));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD original_descriptor{};
  KD patched_descriptor{};
  std::memcpy(&original_descriptor,
              bytes.data() + original.kernels().front().descriptor_file_offset,
              sizeof(original_descriptor));
  std::memcpy(&patched_descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(patched_descriptor));
  EXPECT_EQ(static_cast<int64_t>(patched.kernel_descriptor_offset("atomic_high_sgpr_pressure")) +
                patched_descriptor.kernel_code_entry_byte_offset,
            static_cast<int64_t>(original.kernel_descriptor_offset("atomic_high_sgpr_pressure")) +
                original_descriptor.kernel_code_entry_byte_offset);
  uint32_t entry_word = 0;
  std::memcpy(&entry_word,
              patched.text_sections().front()->data() + patched.kernels().front().entry_text_offset,
              sizeof(entry_word));
  EXPECT_NE(entry_word, 0xEE0B0000u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("automatically assigned EXEC-save SGPRs s84:s105") != std::string::npos;
  }));
}

TEST(ConSanMoi, InlineAtomicUsesIndirectIslandsForFarAppendedHelpers) {
  constexpr size_t kLargeTextWords = 33000u;
  const auto release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(release);
  ASSERT_TRUE(acquire);
  std::vector<uint32_t> text_words = {
      0xEE0B0000u,
      0x00000000u,
      0x00000000u, // global_wb
      (*release)[0],
      (*release)[1],
      (*release)[2],
      (*acquire)[0],
      (*acquire)[1],
      (*acquire)[2],
      0xEE0AC000u,
      0x00000000u,
      0x00000000u, // global_inv
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  constexpr uint64_t kReleaseOffset = 3u * sizeof(uint32_t);
  constexpr uint64_t kAcquireOffset = 6u * sizeof(uint32_t);
  constexpr uint64_t kOwnerCodeBytes = 13u * sizeof(uint32_t);
  // Three local indirect islands: one for the dynamic-stack entry prologue
  // and two for the far release/acquire helpers.
  text_words.resize(37u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.resize(kLargeTextWords - 1u, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  ASSERT_FALSE(compute_sopp_branch_simm16(kReleaseOffset, original_text_size));
  ASSERT_FALSE(compute_sopp_branch_simm16(kAcquireOffset, original_text_size));
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = kOwnerCodeBytes; });
  append_kernel_metadata_note(bytes, "lds_probe", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/0u);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_inline_workgroup_shadow = false;
  options.scratch_vgpr = 32;
  options.moi_exec_save_sgpr = 80;
  options.moi_dispatch_id_sgpr = 60;
  options.moi_owner_vgpr = 14;
  options.moi_epoch_vgpr = 15;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            3u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());
}

TEST(ConSanMoi, InlineAtomicPersistentDispatchIdCoversEveryAcquireReleaseComparison) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.helper_atomic_acquire_release = true;
  fixture.second_continuation_live_sgprs = {0u};
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  options.moi_track_atomics = true;
  options.moi_track_barriers = false;
  options.moi_inline_workgroup_shadow = false;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.max_patches = 2u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("proven-unused RDNA4 SGPR pair") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const uint16_t scratch = *patch->scratch_vgpr;
  struct DispatchComparison {
    uint16_t address_offset;
    uint16_t value_offset;
    size_t field_offset;
    size_t expected_count;
    std::string_view label;
  };
  constexpr std::array kDispatchComparisons = {
      // append_inline_atomic_acquire_import: the release-slot value is loaded
      // into `value` at +2.
      DispatchComparison{0u, 2u, offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id), 1u,
                         "release-slot acquire"},
      // append_inline_acquired_epoch_token_publications: the acquired-token
      // value is loaded into `value` at +19 for each of five unrolled edges.
      DispatchComparison{0u, 19u, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id), 5u,
                         "acquired-token publication"},
      // append_inline_release_causal_snapshot_capture: the token-table value
      // is loaded through `temporary` at +20 from the address at +5.
      DispatchComparison{5u, 20u, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id), 1u,
                         "causal-snapshot scan"},
      // append_inline_versioned_release_transaction: the release-record value
      // is loaded through `temporary` at +4.
      DispatchComparison{0u, 4u, offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id), 1u,
                         "release-record verification"},
  };
  for (const DispatchComparison &comparison : kDispatchComparisons) {
    SCOPED_TRACE(comparison.label);
    for (const bool high_word : {false, true}) {
      SCOPED_TRACE(high_word ? "high" : "low");
      const uint16_t address = static_cast<uint16_t>(scratch + comparison.address_offset);
      const uint16_t value = static_cast<uint16_t>(scratch + comparison.value_offset);
      const uint16_t expected_sgpr =
          static_cast<uint16_t>(*result.resolved_moi_dispatch_id_sgpr + (high_word ? 1u : 0u));
      const auto load = instrumentation::build_flat_load_b32(
          address, value, ROCJITSU_CODE_ARCH_RDNA4,
          static_cast<uint32_t>(comparison.field_offset + (high_word ? sizeof(uint32_t) : 0u)));
      const auto wait = instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
      const auto compare =
          instrumentation::build_v_cmp_eq_u32_vcc(expected_sgpr, value, ROCJITSU_CODE_ARCH_RDNA4);
      ASSERT_TRUE(load && wait && compare);
      std::vector<uint32_t> expected = *load;
      expected.push_back(*wait);
      expected.push_back(*compare);
      EXPECT_EQ(count_subsequence(cave, expected), comparison.expected_count);
    }
  }
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, InlineAtomicLiteralDispatchIdCoversEveryAcquireReleaseComparison) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.helper_atomic_acquire_release = true;
  // The explicit 22-register atomic EXEC window occupies s84:s105. Keep every
  // lower ordinary SGPR live across both calls so no lifetime-safe persistent
  // pair exists and the public planner must select its literal representation.
  for (uint16_t sgpr = 0u; sgpr < 84u; ++sgpr) {
    fixture.first_continuation_live_sgprs.push_back(sgpr);
    fixture.second_continuation_live_sgprs.push_back(sgpr);
  }
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  options.moi_track_atomics = true;
  options.moi_track_barriers = false;
  options.moi_inline_workgroup_shadow = false;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_exec_save_sgpr = 84u;
  options.max_patches = 2u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_vgpr);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("literal report dispatch ID") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const uint16_t scratch = *patch->scratch_vgpr;
  struct DispatchComparison {
    uint16_t literal_temporary_offset;
    uint16_t value_offset;
    std::string_view label;
  };
  constexpr std::array kDispatchComparisons = {
      // append_inline_atomic_acquire_import: version-transaction `temporary`
      // at +4 compares the release-slot value loaded at +2.
      DispatchComparison{4u, 2u, "release-slot acquire"},
      // append_inline_acquired_epoch_token_publications: `hash` at +21
      // compares the acquired-token value loaded at +19.
      DispatchComparison{21u, 19u, "acquired-token publication"},
      // append_inline_release_causal_snapshot_capture: `version_before` at
      // +17 compares the token-table value loaded at +20.
      DispatchComparison{17u, 20u, "causal-snapshot scan"},
      // append_inline_versioned_release_transaction: `cas_new` at +5 compares
      // the release-record value loaded through `temporary` at +4.
      DispatchComparison{5u, 4u, "release-record verification"},
  };
  for (const DispatchComparison &comparison : kDispatchComparisons) {
    SCOPED_TRACE(comparison.label);
    for (const bool high_word : {false, true}) {
      SCOPED_TRACE(high_word ? "high" : "low");
      const uint16_t literal_temporary =
          static_cast<uint16_t>(scratch + comparison.literal_temporary_offset);
      const uint16_t value = static_cast<uint16_t>(scratch + comparison.value_offset);
      const uint32_t literal = high_word
                                   ? static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u)
                                   : static_cast<uint32_t>(options.moi_report_dispatch_id);
      const auto materialize = instrumentation::build_v_mov_b32_literal(literal_temporary, literal,
                                                                        ROCJITSU_CODE_ARCH_RDNA4);
      const auto compare = instrumentation::build_v_cmp_eq_u32_vcc(
          vector_source_vgpr(literal_temporary), value, ROCJITSU_CODE_ARCH_RDNA4);
      ASSERT_TRUE(materialize && compare);
      std::vector<uint32_t> expected = *materialize;
      expected.push_back(*compare);
      EXPECT_TRUE(contains_subsequence(cave, expected));
    }
  }
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, InlineAtomicDynamicStackSpillPreservesEverySharedOwnerFrame) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.helper_atomic_acquire_release = true;
  fixture.first_continuation_uses_v1 = true;
  fixture.second_continuation_live_sgprs = {0u};
  // Leave the seven-word in-place entry relay relocatable; the shared helper
  // call begins immediately after it.
  fixture.entry_nop_words = 7u;
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());
  constexpr std::array<std::string_view, 1> kAdditionalOwners = {"shared_owner_1"};
  // Every owner of the shared atomic helper is dynamic.
  append_kernel_metadata_note(bytes, "shared_owner_0", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/0u, std::nullopt, std::nullopt,
                              /*has_dynamic_lds=*/false, kAdditionalOwners);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  options.moi_track_atomics = true;
  options.moi_track_barriers = false;
  options.moi_inline_workgroup_shadow = false;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_dispatch_id_sgpr = 80u;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.max_patches = 2u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  EXPECT_EQ(result.resolved_moi_dispatch_id_sgpr, options.moi_dispatch_id_sgpr);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan->owner_descriptor_file_offsets.size(), 2u);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_GT(patch->spilled_vgpr_count, 0u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  // Atomic-only Inline state normally ends at +21; dynamic spill extends the
  // same window through the frame-base save slot at +24.
  const uint16_t saved_frame_sgpr =
      static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 24u);
  EXPECT_NE(std::ranges::find(cave, build_s_mov_b32(saved_frame_sgpr, /*frame base=*/33u,
                                                    ROCJITSU_CODE_ARCH_RDNA4)),
            cave.end());
  EXPECT_NE(std::ranges::find(cave, build_s_mov_b32(/*frame base=*/33u, /*stack top=*/32u,
                                                    ROCJITSU_CODE_ARCH_RDNA4)),
            cave.end());
  EXPECT_NE(std::ranges::find(cave, build_s_mov_b32(/*stack top=*/32u, /*frame base=*/33u,
                                                    ROCJITSU_CODE_ARCH_RDNA4)),
            cave.end());
  EXPECT_NE(std::ranges::find(cave, build_s_mov_b32(/*frame base=*/33u, saved_frame_sgpr,
                                                    ROCJITSU_CODE_ARCH_RDNA4)),
            cave.end());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, InlineAtomicDynamicStackRejectsExplicitExecWindowWithoutFrameSlot) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.helper_atomic_acquire_release = true;
  fixture.second_continuation_live_sgprs = {0u};
  fixture.entry_nop_words = 7u;
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());
  constexpr std::array<std::string_view, 1> kAdditionalOwners = {"shared_owner_1"};
  append_kernel_metadata_note(bytes, "shared_owner_0", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/0u, std::nullopt, std::nullopt,
                              /*has_dynamic_lds=*/false, kAdditionalOwners);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  options.moi_track_atomics = true;
  options.moi_track_barriers = false;
  options.moi_inline_workgroup_shadow = false;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_dispatch_id_sgpr = 80u;
  // Atomic-only state normally permits s234:s255. The dynamic frame save at
  // +24 makes that explicit window two registers too short.
  options.moi_exec_save_sgpr = 234u;
  options.max_patches = 2u;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("aliases an architectural special SGPR") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
                              &ConSanPatchInfo::kind),
            result.patches.end());
}

TEST(ConSanMoi, SampledAtomicAttachmentRejectsEveryCausalIdentityMismatch) {
  const ConSanMoiSampledAtomicAttachmentKey key{
      .generation = 0x1122334455667788ull,
      .dispatch_id = 0x8877665544332211ull,
      .workgroup_x = 3,
      .workgroup_y = 4,
      .workgroup_z = 5,
      .epoch = 7,
      .owner_id = 11,
  };
  const ConSanMoiSampledCausalWindow window{
      key.generation,
      key.dispatch_id,
      key.workgroup_x,
      key.workgroup_y,
      key.workgroup_z,
      key.epoch,
      /*first_entry=*/2,
      /*entry_count=*/1,
      static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready),
      /*cluster_workgroup_id=*/0,
  };
  const uint64_t watchpoint = pack_consan_moi_sampled_watchpoint_entry(
      ConSanMoiShadowAccessKind::Write, key.owner_id, key.epoch,
      static_cast<uint32_t>(key.generation), /*start_cell=*/4, /*cell_count=*/1);
  ASSERT_TRUE(consan_moi_sampled_atomic_attachment_matches(window, watchpoint, 2, key));

  const auto reject_window = [&](auto mutate) {
    ConSanMoiSampledCausalWindow changed = window;
    mutate(changed);
    EXPECT_FALSE(consan_moi_sampled_atomic_attachment_matches(changed, watchpoint, 2, key));
  };
  reject_window([](auto &value) { value.publication_state = 1; });
  reject_window([](auto &value) { ++value.generation; });
  reject_window([](auto &value) { ++value.dispatch_id; });
  reject_window([](auto &value) { ++value.workgroup_x; });
  reject_window([](auto &value) { ++value.workgroup_y; });
  reject_window([](auto &value) { ++value.workgroup_z; });
  reject_window([](auto &value) { ++value.epoch; });
  reject_window([](auto &value) { ++value.first_entry; });
  reject_window([](auto &value) { value.entry_count = 0; });
  reject_window([](auto &value) { value.cluster_workgroup_id = 1; });
  EXPECT_FALSE(consan_moi_sampled_atomic_attachment_matches(window, watchpoint, 1, key));

  auto changed_key = key;
  ++changed_key.owner_id;
  EXPECT_FALSE(consan_moi_sampled_atomic_attachment_matches(window, watchpoint, 2, changed_key));
  const uint64_t consumed = pack_consan_moi_sampled_watchpoint_entry(
      ConSanMoiShadowAccessKind::Write, key.owner_id, key.epoch,
      static_cast<uint32_t>(key.generation), 4, 1, /*consumed=*/true);
  EXPECT_FALSE(consan_moi_sampled_atomic_attachment_matches(window, consumed, 2, key));
}

TEST(ConSanMoi, SampledCausalSelectionUsesWholeWorkgroupEpochKey) {
  uint32_t selected_offset = 4;
  for (uint32_t offset = 0; offset < 4; ++offset) {
    if (consan_moi_sampled_causal_window_selected(7, 9, 3, 4, 5, 6, 4, offset)) {
      ASSERT_EQ(selected_offset, 4u);
      selected_offset = offset;
    }
  }
  ASSERT_LT(selected_offset, 4u);
  EXPECT_TRUE(consan_moi_sampled_causal_window_selected(7, 9, 3, 4, 5, 6, 4, selected_offset));
  EXPECT_FALSE(
      consan_moi_sampled_causal_window_selected(7, 9, 3, 4, 5, 6, 4, (selected_offset + 1u) & 3u));
  EXPECT_FALSE(consan_moi_sampled_causal_window_selected(7, 9, 3, 4, 5, 6, 3, 0));
  EXPECT_FALSE(consan_moi_sampled_causal_window_selected(7, 9, 3, 4, 5, 6, 4, 4));
}

TEST(ConSanMoi, SampledCausalClaimSerializesInterleavedPublishers) {
  std::array<ConSanMoiSampledCausalWindow, 2> windows{};
  const ConSanMoiSampledCausalKey key{7, 9, 1, 2, 3, 4};
  const ConSanMoiSampledClaimResult winner = consan_moi_sampled_begin_causal_claim(windows, key);
  ASSERT_EQ(winner.outcome, ConSanMoiSampledClaimOutcome::Claimed);
  EXPECT_EQ(windows[winner.slot].publication_state,
            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing));

  const ConSanMoiSampledClaimResult interleaved =
      consan_moi_sampled_begin_causal_claim(windows, key);
  EXPECT_EQ(interleaved.outcome, ConSanMoiSampledClaimOutcome::Busy);
  EXPECT_TRUE(consan_moi_sampled_commit_causal_claim(windows[winner.slot], key, 5, 2));

  const ConSanMoiSampledClaimResult existing = consan_moi_sampled_begin_causal_claim(windows, key);
  EXPECT_EQ(existing.outcome, ConSanMoiSampledClaimOutcome::Existing);
  EXPECT_EQ(existing.slot, winner.slot);
  EXPECT_EQ(windows[winner.slot].first_entry, 5u);
  EXPECT_EQ(windows[winner.slot].entry_count, 2u);
  EXPECT_EQ(windows[winner.slot].cluster_workgroup_id, 0u);
}

TEST(ConSanMoi, SampledCausalClaimTypesCapacityAndMalformedSlots) {
  std::array<ConSanMoiSampledCausalWindow, 2> windows{};
  const ConSanMoiSampledCausalKey first_key{7, 9, 0, 0, 0, 1};
  const ConSanMoiSampledCausalKey second_key{7, 9, 1, 0, 0, 1};
  const ConSanMoiSampledCausalKey third_key{7, 9, 2, 0, 0, 1};
  for (const auto &[key, first_entry] :
       std::array{std::pair{first_key, 0u}, std::pair{second_key, 1u}}) {
    const ConSanMoiSampledClaimResult claim = consan_moi_sampled_begin_causal_claim(windows, key);
    ASSERT_EQ(claim.outcome, ConSanMoiSampledClaimOutcome::Claimed);
    ASSERT_TRUE(consan_moi_sampled_commit_causal_claim(windows[claim.slot], key, first_entry, 1));
  }
  const ConSanMoiSampledClaimResult full =
      consan_moi_sampled_begin_causal_claim(windows, third_key);
  EXPECT_EQ(full.outcome, ConSanMoiSampledClaimOutcome::CapacityExhausted);
  EXPECT_EQ(full.collision_count, 2u);

  std::array<ConSanMoiSampledCausalWindow, 1> malformed{};
  malformed[0].publication_state =
      static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Malformed);
  const ConSanMoiSampledClaimResult rejected =
      consan_moi_sampled_begin_causal_claim(malformed, first_key);
  EXPECT_EQ(rejected.outcome, ConSanMoiSampledClaimOutcome::CapacityExhausted);
  EXPECT_EQ(rejected.malformed_slot_count, 1u);

  std::array<ConSanMoiSampledCausalWindow, 1> aborted{};
  const ConSanMoiSampledClaimResult claim =
      consan_moi_sampled_begin_causal_claim(aborted, first_key);
  ASSERT_EQ(claim.outcome, ConSanMoiSampledClaimOutcome::Claimed);
  EXPECT_TRUE(consan_moi_sampled_abort_causal_claim(aborted[claim.slot]));
  EXPECT_EQ(aborted[claim.slot].publication_state,
            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Malformed));
  EXPECT_FALSE(consan_moi_sampled_commit_causal_claim(aborted[claim.slot], first_key, 0, 1));
}

TEST(ConSanMoi, SampledCausalPublicationKeepsWindowsWholeAndBounded) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      7, 9, /*access_record_capacity=*/4, /*diagnostic_capacity=*/0,
      /*exact_shadow_entry_capacity=*/0, /*sampled_watchpoint_capacity=*/4);
  header.access_record_count = 4;
  auto access = [](uint32_t workgroup, uint32_t owner, ConSanMoiShadowAccessKind kind) {
    ConSanMoiAccessRecord record;
    record.generation = 7;
    record.workgroup_x = workgroup;
    record.wave_id = owner;
    record.access_kind = static_cast<uint32_t>(kind);
    record.start_cell = 10;
    record.cell_count = 1;
    record.epoch = 2;
    return record;
  };
  const std::array records = {
      access(0, 1, ConSanMoiShadowAccessKind::Write),
      access(0, 2, ConSanMoiShadowAccessKind::Read),
      access(1, 3, ConSanMoiShadowAccessKind::Write),
      access(1, 4, ConSanMoiShadowAccessKind::Read),
  };
  std::array<uint64_t, 4> entries{};
  std::array<ConSanMoiSampledCausalWindow, 2> windows{};
  const ConSanMoiSampledPublishResult complete = consan_moi_sampled_publish_causal_windows(
      header, records, /*selection_stride=*/1, /*selection_offset=*/0, entries, windows);
  EXPECT_FALSE(complete.invalid_selection);
  EXPECT_EQ(complete.eligible_window_count, 2u);
  EXPECT_EQ(complete.selected_window_count, 2u);
  EXPECT_EQ(complete.published_window_count, 2u);
  EXPECT_EQ(complete.published_entry_count, 4u);
  EXPECT_EQ(windows[0].first_entry, 0u);
  EXPECT_EQ(windows[0].entry_count, 2u);
  EXPECT_EQ(windows[0].workgroup_x, 0u);
  EXPECT_EQ(windows[1].first_entry, 2u);
  EXPECT_EQ(windows[1].entry_count, 2u);
  EXPECT_EQ(windows[1].workgroup_x, 1u);

  std::array<uint64_t, 3> short_entries{};
  std::array<ConSanMoiSampledCausalWindow, 2> short_windows{};
  header.sampled_watchpoint_capacity = short_entries.size();
  const ConSanMoiSampledPublishResult bounded = consan_moi_sampled_publish_causal_windows(
      header, records, 1, 0, short_entries, short_windows);
  EXPECT_TRUE(bounded.sampled_capacity_exhausted);
  EXPECT_EQ(bounded.published_window_count, 1u);
  EXPECT_EQ(bounded.published_entry_count, 2u);
  EXPECT_EQ(short_windows[0].entry_count, 2u);

  std::array<ConSanMoiSampledCausalWindow, 1> short_window_metadata{};
  header.sampled_watchpoint_capacity = entries.size();
  const ConSanMoiSampledPublishResult metadata_bounded = consan_moi_sampled_publish_causal_windows(
      header, records, 1, 0, entries, short_window_metadata);
  EXPECT_TRUE(metadata_bounded.window_capacity_exhausted);
  EXPECT_EQ(metadata_bounded.published_window_count, 1u);
  EXPECT_EQ(metadata_bounded.published_entry_count, 2u);
  EXPECT_EQ(short_window_metadata[0].entry_count, 2u);
}

TEST(ConSanMoi, SampledCausalPublicationRejectsWholeMalformedWindow) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      7, 9, /*access_record_capacity=*/2, /*diagnostic_capacity=*/0,
      /*exact_shadow_entry_capacity=*/0, /*sampled_watchpoint_capacity=*/2);
  header.access_record_count = 2;
  std::array<ConSanMoiAccessRecord, 2> records{};
  for (uint32_t index = 0; index < records.size(); ++index) {
    records[index].generation = 7;
    records[index].workgroup_x = 1;
    records[index].wave_id = index;
    records[index].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
    records[index].start_cell = 3;
    records[index].cell_count = 1;
    records[index].epoch = 2;
  }
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Atomic);
  std::array<uint64_t, 2> entries{};
  std::array<ConSanMoiSampledCausalWindow, 1> windows{};
  const ConSanMoiSampledPublishResult result =
      consan_moi_sampled_publish_causal_windows(header, records, 1, 0, entries, windows);
  EXPECT_EQ(result.selected_window_count, 1u);
  EXPECT_EQ(result.malformed_window_count, 1u);
  EXPECT_EQ(result.published_window_count, 0u);
  EXPECT_EQ(result.published_entry_count, 0u);

  const ConSanMoiSampledPublishResult invalid = consan_moi_sampled_publish_causal_windows(
      header, records, /*selection_stride=*/3, 0, entries, windows);
  EXPECT_TRUE(invalid.invalid_selection);
  EXPECT_EQ(invalid.processed_access_count, 0u);
}

TEST(ConSanMoi, SampledCausalReplayNeverComparesDifferentWindows) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      7, 9, /*access_record_capacity=*/0, /*diagnostic_capacity=*/1,
      /*exact_shadow_entry_capacity=*/0, /*sampled_watchpoint_capacity=*/2);
  const std::array<uint64_t, 2> entries = {
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write, 1, 2, 7, 10, 1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read, 2, 2, 7, 10, 1),
  };
  const std::array separate = {
      ConSanMoiSampledCausalWindow{7, 9, 0, 0, 0, 2, 0, 1, 2, 0},
      ConSanMoiSampledCausalWindow{7, 9, 1, 0, 0, 2, 1, 1, 2, 0},
  };
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  const ConSanMoiSampledReplayResult clean =
      consan_moi_sampled_replay_causal_windows(header, entries, separate, diagnostics);
  EXPECT_FALSE(clean.invalid_causal_metadata);
  EXPECT_FALSE(clean.conflict);
  EXPECT_EQ(clean.processed_window_count, 2u);

  const std::array together = {
      ConSanMoiSampledCausalWindow{7, 9, 0, 0, 0, 2, 0, 2, 2, 0},
  };
  const ConSanMoiSampledReplayResult conflict =
      consan_moi_sampled_replay_causal_windows(header, entries, together, diagnostics);
  EXPECT_TRUE(conflict.conflict);
  EXPECT_EQ(conflict.emitted_diagnostic_count, 1u);
}

TEST(ConSanMoi, SampledCausalReplayRejectsMetadataBeforeDiagnostics) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      7, 9, /*access_record_capacity=*/0, /*diagnostic_capacity=*/1,
      /*exact_shadow_entry_capacity=*/0, /*sampled_watchpoint_capacity=*/3);
  const std::array<uint64_t, 3> entries = {
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write, 1, 2, 7, 10, 1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read, 2, 2, 7, 10, 1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read, 3, 3, 7, 20, 1),
  };
  const std::array malformed = {
      ConSanMoiSampledCausalWindow{7, 9, 0, 0, 0, 2, 0, 2, 2, 0},
      ConSanMoiSampledCausalWindow{7, 9, 1, 0, 0, 2, 2, 1, 2, 0},
  };
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  diagnostics[0].kind = 99;
  const ConSanMoiSampledReplayResult result =
      consan_moi_sampled_replay_causal_windows(header, entries, malformed, diagnostics);
  EXPECT_TRUE(result.invalid_causal_metadata);
  EXPECT_FALSE(result.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
  EXPECT_EQ(diagnostics[0].kind, 99u);
}

} // namespace
} // namespace rocjitsu
