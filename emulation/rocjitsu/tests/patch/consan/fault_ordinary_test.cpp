// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

TEST(ConSan, FaultInventoryDecodesStableOrdinaryRdna4LoadStoreSites) {
  const std::array<uint32_t, 16> text_words = {
      0xEE050004u,
      7u | (2u << 18u) | (1u << 20u),
      10u | (0xfffff0u << 8u), // global_load_b32 v7, v10, s[4:5] offset:-16
      0xEE068006u,
      1u << 18u | 9u << 23u,
      12u | (20u << 8u), // global_store_b32 v12, v9, s[6:7] offset:20
      0xEC05007Cu,
      4u | (2u << 18u),
      20u | (0xfffffcu << 8u), // flat_load_b32 v4, v[20:21] offset:-4
      0xEC068008u,
      5u << 23u,
      22u | (8u << 8u), // flat_store_b32 v22, v5, s[8:9] offset:8
      0xEE158004u,
      0x00980000u,
      0x00000002u, // global_atomic_add_f32, deliberately not ordinary memory
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "ordinary_memory_kernel");
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;

  const ConSanTransformArtifacts first = test_lower_consan(bytes, options);
  const ConSanTransformArtifacts second = test_lower_consan(bytes, options);
  const auto ordinary = [](const ConSanTransformArtifacts &result) {
    std::vector<const ConSanFaultSite *> sites;
    for (const ConSanFaultSite &site : result.fault_sites) {
      if (site.kind == ConSanFaultSiteKind::OrdinaryMemory)
        sites.push_back(&site);
    }
    return sites;
  };
  const auto first_sites = ordinary(first);
  const auto second_sites = ordinary(second);
  ASSERT_EQ(first_sites.size(), 4u);
  ASSERT_EQ(second_sites.size(), first_sites.size());
  ASSERT_EQ(first.program_inventory.kernels().size(), 1u);
  ASSERT_EQ(test_decoded_sites<ConSanOrdinaryMemorySite>(first.program_inventory,
                                                         first.program_inventory.kernels().front())
                .size(),
            4u);
  ASSERT_EQ(first.program_inventory.access_sites().size(), 2u);
  EXPECT_TRUE(std::ranges::all_of(first.program_inventory.access_sites(), [](const auto &site) {
    return site.template get_if<ConSanOrdinaryMemorySite>() != nullptr;
  }));

  for (size_t i = 0; i < first_sites.size(); ++i) {
    const ConSanFaultSiteDiagnostic diagnostic = test_fault_diagnostic(first, *first_sites[i]);
    EXPECT_EQ(first_sites[i]->identity, second_sites[i]->identity);
    EXPECT_EQ(diagnostic.ordinary_memory_support_reason,
              ConSanOrdinaryMemorySupportReason::Supported);
    EXPECT_EQ(diagnostic.width_bits, 32u);
    EXPECT_EQ(first_sites[i]->occurrence, i);
    ASSERT_EQ(diagnostic.execution_owners.size(), 1u);
    EXPECT_EQ(diagnostic.execution_owners.front().proof, ConSanOwnerProofKind::KernelLocal);
    EXPECT_NE(first_sites[i]->identity.find("|kernel=ordinary_memory_kernel|"
                                            "kind=ordinary-memory|"),
              std::string::npos);
    const ConSanSyncEvent *event = test_sync_event(first, *first_sites[i]);
    const ConSanSyncEvent *second_event = test_sync_event(second, *second_sites[i]);
    ASSERT_NE(event, nullptr);
    ASSERT_NE(second_event, nullptr);
    EXPECT_EQ(event->identity, second_event->identity);
    EXPECT_EQ(event->kind, ConSanSyncKind::OrdinaryMemory);
    EXPECT_EQ(event->operation, i % 2u == 0u ? ConSanSyncOperation::OrdinaryLoad
                                             : ConSanSyncOperation::OrdinaryStore);
    EXPECT_EQ(event->rmw_outcome, ConSanSyncRmwOutcome::NotApplicable);
    const auto event_owners = first.program_inventory.sync().execution_owners(*event);
    ASSERT_EQ(event_owners.size(), diagnostic.execution_owners.size());
    EXPECT_EQ(event_owners.front().descriptor_file_offset,
              diagnostic.execution_owners.front().descriptor_file_offset);
    EXPECT_EQ(event_owners.front().proof, diagnostic.execution_owners.front().proof);
  }

  const ConSanFaultSiteDiagnostic first_load = test_fault_diagnostic(first, *first_sites[0]);
  const ConSanFaultSiteDiagnostic first_store = test_fault_diagnostic(first, *first_sites[1]);
  const ConSanFaultSiteDiagnostic flat_load = test_fault_diagnostic(first, *first_sites[2]);
  const ConSanFaultSiteDiagnostic flat_store = test_fault_diagnostic(first, *first_sites[3]);
  EXPECT_EQ(first_load.mnemonic, "global_load_b32");
  EXPECT_EQ(first_load.semantic_role, "ordinary-load");
  EXPECT_NE(first_load.decoded_operands.find("dst_vgpr=7"), std::string::npos);
  EXPECT_NE(first_load.decoded_operands.find("addr_vgpr=10"), std::string::npos);
  EXPECT_NE(first_load.decoded_operands.find("addr_sgpr=4"), std::string::npos);
  EXPECT_NE(first_load.decoded_operands.find("raw_ioffset=-16"), std::string::npos);
  EXPECT_NE(first_load.decoded_operands.find("raw_scope=2"), std::string::npos);
  EXPECT_EQ(first_store.semantic_role, "ordinary-store");
  EXPECT_NE(first_store.decoded_operands.find("value_vgpr=9"), std::string::npos);
  EXPECT_NE(first_store.decoded_operands.find("addr_sgpr=6"), std::string::npos);
  EXPECT_EQ(flat_load.mnemonic, "flat_load_b32");
  EXPECT_EQ(flat_load.decoded_operands.find("addr_sgpr="), std::string::npos);
  EXPECT_NE(flat_load.decoded_operands.find("raw_saddr=124"), std::string::npos);
  EXPECT_EQ(flat_store.mnemonic, "flat_store_b32");
  EXPECT_NE(flat_store.decoded_operands.find("addr_sgpr=8"), std::string::npos);

  EXPECT_EQ(std::count_if(first.fault_sites.begin(), first.fault_sites.end(),
                          [](const ConSanFaultSite &site) {
                            return site.kind == ConSanFaultSiteKind::Atomic;
                          }),
            1);
}

TEST(ConSan, FaultInventoryRetainsMalformedOrdinaryMemoryAsTypedUnsupported) {
  const std::array<uint32_t, 4> text_words = {
      0xEE050104u, // reserved pad_8_13 bit is set
      7u | (2u << 18u),
      10u,
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "malformed_ordinary_memory");
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  const auto site = std::ranges::find_if(result.fault_sites, [](const ConSanFaultSite &item) {
    return item.kind == ConSanFaultSiteKind::OrdinaryMemory;
  });
  ASSERT_NE(site, result.fault_sites.end());
  const ConSanFaultSiteDiagnostic diagnostic = test_fault_diagnostic(result, *site);
  EXPECT_EQ(diagnostic.mnemonic, "global_load_b32");
  EXPECT_EQ(diagnostic.ordinary_memory_support_reason,
            ConSanOrdinaryMemorySupportReason::MalformedEncoding);
  EXPECT_EQ(test_sync_event(result, *site), nullptr);
  EXPECT_EQ(test_sync_sequence(result, *site), nullptr);
}

TEST(ConSan, FaultInventoryDecodesGfx1250VglobalMemoryForSynchronization) {
  constexpr auto load = cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                                             {.saddr = 124, .vdst = 1, .scope = 2, .vaddr = 0});
  const std::array<uint32_t, 4> text_words = {
      load[0],
      load[1],
      load[2],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "future_global_load");
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  const auto site = std::ranges::find_if(result.fault_sites, [](const ConSanFaultSite &item) {
    return item.kind == ConSanFaultSiteKind::OrdinaryMemory;
  });
  ASSERT_NE(site, result.fault_sites.end());
  const ConSanFaultSiteDiagnostic diagnostic = test_fault_diagnostic(result, *site);
  EXPECT_EQ(diagnostic.ordinary_memory_support_reason,
            ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly);
  EXPECT_EQ(diagnostic.size, 3u * sizeof(uint32_t));
  EXPECT_EQ(diagnostic.width_bits, 32u);
  EXPECT_NE(diagnostic.decoded_operands.find("dst_vgpr=1"), std::string::npos);
  EXPECT_NE(diagnostic.decoded_operands.find("addr_vgpr=0"), std::string::npos);
  EXPECT_NE(diagnostic.decoded_operands.find("raw_scope=2"), std::string::npos);
}

TEST(ConSan, FaultInventoryDecodesGfx1250BufferMemoryForSynchronization) {
  const std::array<uint32_t, 7> text_words = {
      0xC4050018u, 0x40883804u,
      0x00000005u, // buffer_load_b32 v4, v5, s[28:31], s24 offen scope:device
      0xC407400Au, 0x40889018u,
      0x00000004u, // buffer_store_b128 v[24:27], v4, s[72:75], s10 offen scope:device
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;

  const ConSanTransformArtifacts result =
      test_lower_consan(make_gfx1250_code_object(text_words, "gfx1250_buffer_sync"), options);
  std::vector<const ConSanFaultSite *> sites;
  for (const ConSanFaultSite &site : result.fault_sites) {
    if (site.kind == ConSanFaultSiteKind::OrdinaryMemory)
      sites.push_back(&site);
  }
  ASSERT_EQ(sites.size(), 2u);
  const ConSanFaultSiteDiagnostic load = test_fault_diagnostic(result, *sites[0]);
  const ConSanFaultSiteDiagnostic store = test_fault_diagnostic(result, *sites[1]);
  EXPECT_EQ(load.ordinary_memory_support_reason,
            ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly);
  EXPECT_EQ(load.width_bits, 32u);
  EXPECT_NE(load.decoded_operands.find("dst_vgpr=4"), std::string::npos);
  EXPECT_NE(load.decoded_operands.find("addr_vgpr=5"), std::string::npos);
  EXPECT_NE(load.decoded_operands.find("addr_sgpr=28"), std::string::npos);
  EXPECT_NE(load.decoded_operands.find("raw_scope=2"), std::string::npos);
  EXPECT_EQ(store.ordinary_memory_support_reason,
            ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly);
  EXPECT_EQ(store.width_bits, 128u);
  EXPECT_NE(store.decoded_operands.find("value_vgpr=24"), std::string::npos);
  EXPECT_NE(store.decoded_operands.find("addr_sgpr=72"), std::string::npos);
}

TEST(ConSan, FaultInventoryCarriesDirectOwnersForOrdinaryMemoryInSharedHelper) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordinary_memory = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  const auto site = std::ranges::find_if(result.fault_sites, [&](const ConSanFaultSite &item) {
    const ConSanProgramSite *source = test_fault_source(result, item);
    return item.kind == ConSanFaultSiteKind::OrdinaryMemory && source != nullptr &&
           source->container.name == "shared_lds_helper";
  });
  ASSERT_NE(site, result.fault_sites.end());
  const ConSanFaultSiteDiagnostic diagnostic = test_fault_diagnostic(result, *site);
  EXPECT_FALSE(diagnostic.in_kernel);
  ASSERT_EQ(diagnostic.execution_owners.size(), 2u);
  for (const ConSanExecutionOwner &owner : diagnostic.execution_owners)
    EXPECT_EQ(owner.proof, ConSanOwnerProofKind::DirectCall);
}

TEST(ConSan, AssociatesExactSameBlockOrdinaryAcquireLoadCacheSequence) {
  const auto wait_load = build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_load);
  const std::array<uint32_t, 9> text_words = {
      0xEE050004u, 7u | (2u << 18u),
      10u, // global_load_b32 v7, v10, s[4:5]
      *wait_load,
      0xBF870001u, // s_delay_alu is permitted bookkeeping after the wait
      0xEE0AC000u, 0x00000000u,
      0x00000000u, // global_inv
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanTransformArtifacts result =
      test_lower_consan(make_rdna4_lds_code_object(text_words, "ordinary_acquire"), options);

  const auto load = std::ranges::find_if(result.fault_sites, [](const ConSanFaultSite &site) {
    return site.kind == ConSanFaultSiteKind::OrdinaryMemory;
  });
  ASSERT_NE(load, result.fault_sites.end());
  ASSERT_NE(test_sync_event(result, *load), nullptr);
  const ConSanFaultSiteDiagnostic diagnostic = test_fault_diagnostic(result, *load);
  EXPECT_EQ(diagnostic.semantic_role, "ordinary-acquire-load");
  EXPECT_EQ(diagnostic.sync_memory_role, ConSanSyncMemoryRole::Acquire);
  EXPECT_EQ(diagnostic.sync_confidence, ConSanSemanticConfidence::Conservative);

  const ConSanSyncSequence *sequence = test_sync_sequence(result, *load);
  ASSERT_NE(sequence, nullptr);
  EXPECT_EQ(sequence->kind, ConSanSyncKind::OrdinaryMemory);
  EXPECT_EQ(sequence->operation, ConSanSyncOperation::OrdinaryLoad);
  EXPECT_EQ(sequence->member_event_ids.size(), 2u);
  EXPECT_EQ(sequence->begin_text_offset, 0u);
  EXPECT_EQ(sequence->end_text_offset, 8u * sizeof(uint32_t));
  EXPECT_NE(sequence->identity.find("|acquire-cache="), std::string::npos);
}

TEST(ConSan, AssociatesRetainedOrdinaryLoadSelfLoopExitAcquireSequence) {
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanTransformArtifacts result =
      test_lower_consan(make_rdna4_flag_self_loop_acquire_code_object(), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  const auto load = std::ranges::find_if(result.fault_sites, [&](const ConSanFaultSite &site) {
    const auto diagnostic = consan_fault_site_diagnostic(result.program_inventory, site);
    return site.kind == ConSanFaultSiteKind::OrdinaryMemory && diagnostic.has_value() &&
           diagnostic->semantic_role == "ordinary-acquire-load";
  });
  ASSERT_NE(load, result.fault_sites.end());
  const ConSanSyncSequence *sequence = test_sync_sequence(result, *load);
  ASSERT_NE(sequence, nullptr);
  EXPECT_EQ(sequence->memory_role, ConSanSyncMemoryRole::Acquire);
  EXPECT_EQ(sequence->memory_role_confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(sequence->member_event_ids.size(), 2u);
  EXPECT_NE(sequence->identity.find("|acquire-cache="), std::string::npos);
}

TEST(ConSan, AssociatesGfx1250BufferPollLoopWithBoundedAddressSetup) {
  const std::array<uint32_t, 12> text_words = {
      0xBE9C0132u, // bounded address setup at the polling-loop header
      0xC4050018u, 0x40883804u,
      0x00000005u, // buffer_load_b32 v4, v5, s[28:31], s24 offen scope:device
      0xBFC00000u, // s_wait_loadcnt 0
      0x7E340504u, // v_readfirstlane_b32 s26, v4
      0xBF06811Au, // s_cmp_eq_u32 s26, 1
      0xBFA1FFF8u, // s_cbranch_scc0 to the address-setup loop header
      0xEE0AC07Cu, 0x00080000u,
      0x00000000u, // global_inv scope:device
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanTransformArtifacts result =
      test_lower_consan(make_gfx1250_code_object(text_words, "gfx1250_buffer_poll"), options);

  const auto load = std::ranges::find_if(result.fault_sites, [](const ConSanFaultSite &site) {
    return site.kind == ConSanFaultSiteKind::OrdinaryMemory;
  });
  ASSERT_NE(load, result.fault_sites.end());
  const ConSanSyncSequence *sequence = test_sync_sequence(result, *load);
  ASSERT_NE(sequence, nullptr);
  EXPECT_EQ(sequence->memory_role, ConSanSyncMemoryRole::Acquire);
  EXPECT_EQ(sequence->member_event_ids.size(), 2u);
  EXPECT_NE(sequence->identity.find("|acquire-cache="), std::string::npos);
}

TEST(ConSan, AssociatesGeneratedGfx1250BufferPollLoopShape) {
  const std::array<uint32_t, 28> text_words = {
      0x84188209u, 0xBF860000u, 0x7E0A0280u, 0xBE9C0132u, 0xBE9E00FFu, 0xFFFFF000u, 0xBE9F0080u,
      0x8B05FF1Eu, 0x0000007Fu, 0xBF870009u, 0x84059905u, 0x8B1DFF1Du, 0x01FFFFFFu, 0xBF870009u,
      0x8C1D051Du, 0x851E871Eu, 0xBFC50000u, 0xC4050018u, 0x40883804u,
      0x00000005u, // buffer_load_b32 v4, v5, s[28:31], s24 offen scope:device
      0xBFC00000u, 0x7E340504u, 0xBF06811Au,
      0xBFA1FFE8u,                           // branch to the first instruction
      0xEE0AC07Cu, 0x00080000u, 0x00000000u, // global_inv scope:device
      0xBFB00000u,
  };
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanTransformArtifacts result = test_lower_consan(
      make_gfx1250_code_object(text_words, "gfx1250_generated_buffer_poll"), options);

  const auto load = std::ranges::find_if(result.fault_sites, [](const ConSanFaultSite &site) {
    return site.kind == ConSanFaultSiteKind::OrdinaryMemory;
  });
  ASSERT_NE(load, result.fault_sites.end());
  const ConSanSyncSequence *sequence = test_sync_sequence(result, *load);
  ASSERT_NE(sequence, nullptr);
  EXPECT_EQ(sequence->memory_role, ConSanSyncMemoryRole::Acquire);
  EXPECT_EQ(sequence->member_event_ids.size(), 2u);
  ASSERT_EQ(result.program_inventory.sync().moi_fence_candidates.size(), 1u);
  const ConSanMoiFenceCandidate &fence =
      result.program_inventory.sync().moi_fence_candidates.front();
  EXPECT_TRUE(fence.eligible());
  ASSERT_TRUE(fence.communication_event);
  const ConSanSyncEvent *communication =
      result.program_inventory.sync().find_event(*fence.communication_event);
  ASSERT_NE(communication, nullptr);
  EXPECT_EQ(communication->address_source, ConSanSyncAddressSource::BufferResource);
}

TEST(ConSan, OrdinaryAcquireAssociationFailsClosedOnInexactShapes) {
  const auto wait_load = build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_load);
  const auto acquire_count = [](std::span<const uint32_t> words) {
    ConSanOptions options = moi_options();
    options.fault_dry_run = true;
    const ConSanTransformArtifacts result =
        test_lower_consan(make_rdna4_lds_code_object(words, "inexact_acquire"), options);
    return std::ranges::count_if(result.program_inventory.sync().sync_sequences,
                                 [](const ConSanSyncSequence &sequence) {
                                   return sequence.kind == ConSanSyncKind::OrdinaryMemory &&
                                          sequence.memory_role == ConSanSyncMemoryRole::Acquire;
                                 });
  };
  constexpr std::array<uint32_t, 3> load = {0xEE050004u, 7u | (2u << 18u), 10u};
  constexpr std::array<uint32_t, 3> unordered_load = {0xEE050004u, 7u, 10u};
  constexpr std::array<uint32_t, 3> malformed_load = {0xEE050104u, 7u | (2u << 18u), 10u};
  constexpr std::array<uint32_t, 3> store = {0xEE068004u, 2u << 18u | 7u << 23u, 10u};
  constexpr std::array<uint32_t, 3> invalidate = {0xEE0AC07Cu, 2u << 18u, 0u};
  const auto shape = [&](std::span<const uint32_t> access, std::span<const uint32_t> middle,
                         size_t invalidate_count) {
    std::vector<uint32_t> words(access.begin(), access.end());
    words.insert(words.end(), middle.begin(), middle.end());
    for (size_t i = 0; i < invalidate_count; ++i)
      words.insert(words.end(), invalidate.begin(), invalidate.end());
    words.push_back(0xBFB00000u);
    return words;
  };
  const std::array<uint32_t, 1> wait = {*wait_load};
  const std::array<uint32_t, 4> intervening_store = {*wait_load, store[0], store[1], store[2]};
  const std::array<uint32_t, 2> intervening_barrier = {*wait_load, 0xBF940000u};
  const std::array<uint32_t, 2> block_split = {*wait_load, 0xBFA00000u};
  const std::vector<std::vector<uint32_t>> rejected = {
      shape(load, {}, 1u),
      shape(load, wait, 0u),
      shape(load, intervening_store, 1u),
      shape(load, intervening_barrier, 1u),
      shape(load, block_split, 1u),
      shape(unordered_load, wait, 1u),
      shape(malformed_load, wait, 1u),
      shape(store, wait, 1u),
  };
  for (size_t index = 0; index < rejected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(acquire_count(rejected[index]), 0u);
  }

  // A second cache operation after the unique wait-adjacent invalidate is not
  // an ambiguous acquire match: the intervening first invalidate prevents a
  // second bounded load-to-cache proof. Preserve the exact compiler prefix
  // while admitting unrelated later cache maintenance.
  EXPECT_EQ(acquire_count(shape(load, wait, 2u)), 1u);
}

TEST(ConSan, OrdinaryAcquireMetadataRejectsCorruption) {
  ConSanSyncEvent load;
  load.kind = ConSanSyncKind::OrdinaryMemory;
  load.operation = ConSanSyncOperation::OrdinaryLoad;
  load.confidence = ConSanSemanticConfidence::Conservative;
  load.semantic_id.physical.code_object = make_consan_code_object_id(std::array<uint8_t, 1>{1u});
  load.source_site = {0};
  load.scope = ConSanMemoryScope::Agent;
  ConSanSyncEvent cache = load;
  cache.kind = ConSanSyncKind::Fence;
  cache.operation = ConSanSyncOperation::Fence;
  cache.source_site = {1};
  std::vector<ConSanProgramSite> program_sites(2);
  program_sites[0].container = {
      .id = {0}, .kind = ConSanProgramContainerKind::Kernel, .name = "kernel"};
  program_sites[1].container = program_sites[0].container;
  ConSanOrdinaryMemorySite load_source;
  load_source.width_bits = 32u;
  program_sites[0].payload = std::move(load_source);
  program_sites[0].execution_owners.push_back(
      {.descriptor_file_offset = 64u, .proof = ConSanOwnerProofKind::KernelLocal});
  ConSanFenceSite cache_source;
  cache_source.cache_operation = ConSanCacheOperation::Acquire;
  cache_source.mnemonic = "global_inv";
  program_sites[1].payload = std::move(cache_source);
  program_sites[1].execution_owners = program_sites[0].execution_owners;
  ConSanSyncSequence load_sequence;
  load_sequence.kind = ConSanSyncKind::OrdinaryMemory;
  load_sequence.operation = ConSanSyncOperation::OrdinaryLoad;
  load_sequence.basic_block_index = 3u;
  ConSanSyncSequence cache_sequence;
  cache_sequence.kind = ConSanSyncKind::Fence;
  cache_sequence.operation = ConSanSyncOperation::Fence;
  cache_sequence.basic_block_index = 3u;

  EXPECT_TRUE(consan_ordinary_acquire_metadata_compatible(program_sites, load, load_sequence, cache,
                                                          cache_sequence));
  ++cache.semantic_id.physical.code_object.collision_verifier;
  EXPECT_FALSE(consan_ordinary_acquire_metadata_compatible(program_sites, load, load_sequence,
                                                           cache, cache_sequence));
  cache.semantic_id.physical.code_object = load.semantic_id.physical.code_object;
  program_sites[1].execution_owners.front().descriptor_file_offset = 128u;
  EXPECT_FALSE(consan_ordinary_acquire_metadata_compatible(program_sites, load, load_sequence,
                                                           cache, cache_sequence));
  program_sites[1].execution_owners = program_sites[0].execution_owners;
  cache_sequence.basic_block_index = 4u;
  EXPECT_FALSE(consan_ordinary_acquire_metadata_compatible(program_sites, load, load_sequence,
                                                           cache, cache_sequence));
}

TEST(ConSan, AssociatesExactSameBlockOrdinaryReleaseStoreCacheSequence) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::array<uint32_t, 9> text_words = {
      0xEE0B0000u, 0x00000000u,
      0x00000000u, // global_wb
      *wait_store,
      0xBF870001u, // s_delay_alu is permitted bookkeeping after the wait
      0xEE068004u, 2u << 18u | 7u << 23u,
      10u,         // global_store_b32 v10, v7, s[4:5]
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanTransformArtifacts result =
      test_lower_consan(make_rdna4_lds_code_object(text_words, "ordinary_release"), options);

  const auto store = std::ranges::find_if(result.fault_sites, [](const ConSanFaultSite &site) {
    return site.kind == ConSanFaultSiteKind::OrdinaryMemory;
  });
  ASSERT_NE(store, result.fault_sites.end());
  ASSERT_NE(test_sync_event(result, *store), nullptr);
  const ConSanFaultSiteDiagnostic diagnostic = test_fault_diagnostic(result, *store);
  EXPECT_EQ(diagnostic.semantic_role, "ordinary-release-store");
  EXPECT_EQ(diagnostic.sync_memory_role, ConSanSyncMemoryRole::Release);
  EXPECT_EQ(diagnostic.sync_confidence, ConSanSemanticConfidence::Conservative);

  const ConSanSyncSequence *sequence = test_sync_sequence(result, *store);
  ASSERT_NE(sequence, nullptr);
  EXPECT_EQ(sequence->kind, ConSanSyncKind::OrdinaryMemory);
  EXPECT_EQ(sequence->operation, ConSanSyncOperation::OrdinaryStore);
  EXPECT_EQ(sequence->member_event_ids.size(), 2u);
  EXPECT_EQ(sequence->begin_text_offset, 0u);
  EXPECT_EQ(sequence->end_text_offset, 8u * sizeof(uint32_t));
  EXPECT_NE(sequence->identity.find("|release-cache="), std::string::npos);
}

TEST(ConSan, OrdinaryReleaseAssociationFailsClosedOnInexactShapes) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const auto release_count = [](std::span<const uint32_t> words) {
    ConSanOptions options = moi_options();
    options.fault_dry_run = true;
    const ConSanTransformArtifacts result =
        test_lower_consan(make_rdna4_lds_code_object(words, "inexact_release"), options);
    return std::ranges::count_if(result.program_inventory.sync().sync_sequences,
                                 [](const ConSanSyncSequence &sequence) {
                                   return sequence.kind == ConSanSyncKind::OrdinaryMemory &&
                                          sequence.memory_role == ConSanSyncMemoryRole::Release;
                                 });
  };
  constexpr std::array<uint32_t, 3> writeback = {0xEE0B0000u, 0u, 0u};
  constexpr std::array<uint32_t, 3> store = {0xEE068004u, 2u << 18u | 7u << 23u, 10u};
  constexpr std::array<uint32_t, 3> unordered_store = {0xEE068004u, 7u << 23u, 10u};
  constexpr std::array<uint32_t, 3> malformed_store = {0xEE068104u, 2u << 18u | 7u << 23u, 10u};
  constexpr std::array<uint32_t, 3> load = {0xEE050004u, 7u | (2u << 18u), 10u};
  const auto shape = [&](size_t writeback_count, std::span<const uint32_t> middle,
                         std::span<const uint32_t> access) {
    std::vector<uint32_t> words;
    for (size_t i = 0; i < writeback_count; ++i)
      words.insert(words.end(), writeback.begin(), writeback.end());
    words.insert(words.end(), middle.begin(), middle.end());
    words.insert(words.end(), access.begin(), access.end());
    words.push_back(0xBFB00000u);
    return words;
  };
  const std::array<uint32_t, 1> wait = {*wait_store};
  const std::array<uint32_t, 4> intervening_load = {*wait_store, load[0], load[1], load[2]};
  const std::array<uint32_t, 2> intervening_barrier = {*wait_store, 0xBF940000u};
  const std::array<uint32_t, 2> block_split = {*wait_store, 0xBFA00000u};
  // A compiler-proven wait-only release and redundant consecutive writebacks
  // are both semantically valid. Keep them explicit so the negative table
  // below tests missing ordering rather than historical shape restrictions.
  EXPECT_EQ(release_count(shape(0u, wait, store)), 1u);
  EXPECT_EQ(release_count(shape(2u, wait, store)), 1u);
  const std::vector<std::vector<uint32_t>> rejected = {
      shape(1u, {}, store),
      shape(0u, {}, store),
      shape(2u, {}, store),
      shape(1u, intervening_load, store),
      shape(1u, intervening_barrier, store),
      shape(1u, block_split, store),
      shape(1u, wait, unordered_store),
      shape(1u, wait, malformed_store),
      shape(1u, wait, load),
  };
  for (size_t index = 0; index < rejected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(release_count(rejected[index]), 0u);
  }
}

TEST(ConSan, AssociatesExactScopedOrdinaryReleaseWaitTail) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto wait_load_ds = build_s_wait_loadcnt_dscnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  ASSERT_TRUE(wait_load_ds);
  const std::array<uint32_t, 6> text_words = {
      *wait_store, *wait_load_ds, 0xEE068004u, 2u << 18u | 7u << 23u,
      10u, // global_store_b32 v10, v7, s[4:5] scope:SCOPE_DEV
      0xBFB00000u,
  };
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanTransformArtifacts result =
      test_lower_consan(make_rdna4_lds_code_object(text_words, "scoped_ordinary_release"), options);

  const auto store = std::ranges::find_if(result.fault_sites, [](const ConSanFaultSite &site) {
    return site.kind == ConSanFaultSiteKind::OrdinaryMemory;
  });
  ASSERT_NE(store, result.fault_sites.end());
  ASSERT_NE(test_sync_event(result, *store), nullptr);
  const ConSanFaultSiteDiagnostic diagnostic = test_fault_diagnostic(result, *store);
  EXPECT_EQ(diagnostic.semantic_role, "ordinary-release-store");
  EXPECT_EQ(diagnostic.sync_memory_role, ConSanSyncMemoryRole::Release);
  const ConSanSyncSequence *sequence = test_sync_sequence(result, *store);
  ASSERT_NE(sequence, nullptr);
  EXPECT_EQ(sequence->begin_text_offset, 0u);
  EXPECT_EQ(sequence->release_wait_text_offset, 0u);
  EXPECT_NE(sequence->identity.find("|release-waits="), std::string::npos);
}

TEST(ConSan, ScopedOrdinaryReleaseWaitTailFailsClosedOnInexactShapes) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto wait_load_ds = build_s_wait_loadcnt_dscnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  ASSERT_TRUE(wait_load_ds);
  const auto release_count = [](std::span<const uint32_t> words) {
    ConSanOptions options = moi_options();
    options.fault_dry_run = true;
    const ConSanTransformArtifacts result =
        test_lower_consan(make_rdna4_lds_code_object(words, "scoped_release_reject"), options);
    return std::ranges::count_if(result.program_inventory.sync().sync_sequences,
                                 [](const ConSanSyncSequence &sequence) {
                                   return sequence.kind == ConSanSyncKind::OrdinaryMemory &&
                                          sequence.memory_role == ConSanSyncMemoryRole::Release;
                                 });
  };
  constexpr std::array<uint32_t, 3> device_store = {0xEE068004u, 2u << 18u | 7u << 23u, 10u};
  constexpr std::array<uint32_t, 3> wave_store = {0xEE068004u, 7u << 23u, 10u};
  const std::vector<std::vector<uint32_t>> rejected = {
      {*wait_load_ds, device_store[0], device_store[1], device_store[2], 0xBFB00000u},
      {*wait_store | 1u, device_store[0], device_store[1], device_store[2], 0xBFB00000u},
      {*wait_store, *wait_load_ds | 1u, device_store[0], device_store[1], device_store[2],
       0xBFB00000u},
      {*wait_store, *wait_load_ds, 0xBF800000u, device_store[0], device_store[1], device_store[2],
       0xBFB00000u},
      {*wait_store, *wait_load_ds, wave_store[0], wave_store[1], wave_store[2], 0xBFB00000u},
  };
  for (size_t index = 0; index < rejected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(release_count(rejected[index]), 0u);
  }
}

TEST(ConSan, OrdinaryReleaseMetadataRejectsCorruption) {
  ConSanSyncEvent cache;
  cache.kind = ConSanSyncKind::Fence;
  cache.operation = ConSanSyncOperation::Fence;
  cache.source_site = {0};
  cache.confidence = ConSanSemanticConfidence::Conservative;
  cache.semantic_id.physical.code_object = make_consan_code_object_id(std::array<uint8_t, 1>{1u});
  ConSanSyncEvent store = cache;
  store.kind = ConSanSyncKind::OrdinaryMemory;
  store.operation = ConSanSyncOperation::OrdinaryStore;
  store.source_site = {1};
  store.scope = ConSanMemoryScope::Agent;
  std::vector<ConSanProgramSite> program_sites(2);
  program_sites[0].container = {
      .id = {0}, .kind = ConSanProgramContainerKind::Kernel, .name = "kernel"};
  program_sites[1].container = program_sites[0].container;
  ConSanFenceSite cache_source;
  cache_source.cache_operation = ConSanCacheOperation::Release;
  cache_source.mnemonic = "global_wb";
  program_sites[0].payload = std::move(cache_source);
  program_sites[0].execution_owners.push_back(
      {.descriptor_file_offset = 64u, .proof = ConSanOwnerProofKind::KernelLocal});
  ConSanOrdinaryMemorySite store_source;
  store_source.operation = ConSanOrdinaryMemoryOperation::Store;
  store_source.width_bits = 32u;
  store_source.mnemonic = "global_store_b32";
  program_sites[1].payload = std::move(store_source);
  program_sites[1].execution_owners = program_sites[0].execution_owners;
  ConSanSyncSequence cache_sequence;
  cache_sequence.kind = ConSanSyncKind::Fence;
  cache_sequence.operation = ConSanSyncOperation::Fence;
  cache_sequence.basic_block_index = 3u;
  ConSanSyncSequence store_sequence;
  store_sequence.kind = ConSanSyncKind::OrdinaryMemory;
  store_sequence.operation = ConSanSyncOperation::OrdinaryStore;
  store_sequence.basic_block_index = 3u;

  EXPECT_TRUE(consan_ordinary_release_metadata_compatible(program_sites, cache, cache_sequence,
                                                          store, store_sequence));
  ++store.semantic_id.physical.code_object.collision_verifier;
  EXPECT_FALSE(consan_ordinary_release_metadata_compatible(program_sites, cache, cache_sequence,
                                                           store, store_sequence));
  store.semantic_id.physical.code_object = cache.semantic_id.physical.code_object;
  store.scope = ConSanMemoryScope::Wavefront;
  EXPECT_FALSE(consan_ordinary_release_metadata_compatible(program_sites, cache, cache_sequence,
                                                           store, store_sequence));
  store.scope = ConSanMemoryScope::Agent;
  EXPECT_TRUE(consan_ordinary_release_metadata_compatible(program_sites, cache, cache_sequence,
                                                          store, store_sequence));
  program_sites[0].get_if<ConSanFenceSite>()->cache_operation = ConSanCacheOperation::Acquire;
  EXPECT_FALSE(consan_ordinary_release_metadata_compatible(program_sites, cache, cache_sequence,
                                                           store, store_sequence));
  program_sites[0].get_if<ConSanFenceSite>()->cache_operation = ConSanCacheOperation::Release;
  program_sites[1].execution_owners.front().descriptor_file_offset = 128u;
  EXPECT_FALSE(consan_ordinary_release_metadata_compatible(program_sites, cache, cache_sequence,
                                                           store, store_sequence));
  program_sites[1].execution_owners = program_sites[0].execution_owners;
  store_sequence.basic_block_index = 4u;
  EXPECT_FALSE(consan_ordinary_release_metadata_compatible(program_sites, cache, cache_sequence,
                                                           store, store_sequence));
}

TEST(ConSan, OrdinaryAcquireFaultDryRunExportsStableExactAddressOrderAndScopePlans) {
  const std::vector<uint8_t> bytes = make_rdna4_ordinary_acquire_code_object();
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanTransformArtifacts inventory = test_semantic_inventory(bytes, inventory_options);
  const auto load = std::ranges::find_if(inventory.fault_sites, [&](const ConSanFaultSite &site) {
    const auto diagnostic = consan_fault_site_diagnostic(inventory.program_inventory, site);
    return diagnostic.has_value() && diagnostic->semantic_role == "ordinary-acquire-load";
  });
  ASSERT_NE(load, inventory.fault_sites.end());
  const ConSanSyncSequence *load_sequence = test_sync_sequence(inventory, *load);
  ASSERT_NE(load_sequence, nullptr);

  ConSanOptions options = inventory_options;
  options.fault_dry_run = true;
  options.fault_site_identity = load->identity;
  options.fault_ordinary_wrong_address = true;
  options.fault_ordinary_weaken_order = true;
  options.fault_ordinary_weaken_scope = true;
  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_EQ(result.fault_plans.size(), 3u);
  EXPECT_EQ(result.fault_plans[0].kind, ConSanFaultMutationKind::OrdinaryWrongAddress);
  EXPECT_EQ(result.fault_plans[1].kind, ConSanFaultMutationKind::OrdinaryWeakenOrder);
  EXPECT_EQ(result.fault_plans[2].kind, ConSanFaultMutationKind::OrdinaryWeakenScope);
  for (const ConSanFaultMutationPlan &plan : result.fault_plans) {
    EXPECT_EQ(plan.primary_identity, load->identity);
    EXPECT_EQ(plan.logical_sequence_identity, load_sequence->identity);
    EXPECT_EQ(plan.ordered_member_identities.size(), 2u);
  }
  EXPECT_FALSE(result.fault_plans[0].companion_identity);
  ASSERT_TRUE(result.fault_plans[1].companion_identity);
  EXPECT_FALSE(result.fault_plans[2].companion_identity);
  EXPECT_EQ(result.mutation.fault.planned, 3u);
  EXPECT_FALSE(result.modified());
}

TEST(ConSan, OrdinaryAcquireWeakenOrderRemovesOnlyGlobalInvAndPreservesLoadWait) {
  const std::vector<uint8_t> bytes = make_rdna4_ordinary_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_ordinary_weaken_order = true;
  options.fault_require_exactly_one = true;
  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineOrdinaryOrderRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 4u * sizeof(uint32_t));
  EXPECT_EQ(result.mutation.fault.applied, 1u);
  ASSERT_EQ(result.program_inventory.text_sections().size(), 1u);
  const size_t text_file_offset = result.program_inventory.text_sections().front().file_offset;
  EXPECT_TRUE(std::equal(bytes.begin() + static_cast<ptrdiff_t>(text_file_offset),
                         bytes.begin() + static_cast<ptrdiff_t>(text_file_offset + 16u),
                         result.replacement.begin() + static_cast<ptrdiff_t>(text_file_offset)));
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  for (size_t offset = 16u; offset < 28u; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, result.replacement.data() + text_file_offset + offset, sizeof(word));
    EXPECT_EQ(word, nop);
  }
}

TEST(ConSan, OrdinaryAcquireWrongAddressChangesOnlyAlignedSignedIoffset) {
  const std::vector<uint8_t> bytes = make_rdna4_ordinary_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_ordinary_wrong_address = true;
  options.fault_ordinary_address_delta = 4u;
  options.fault_require_exactly_one = true;
  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineOrdinaryAddressRewrite);
  EXPECT_EQ(result.mutation.fault.applied, 1u);
  ASSERT_EQ(result.program_inventory.text_sections().size(), 1u);
  const size_t text_file_offset = result.program_inventory.text_sections().front().file_offset;
  std::array<uint32_t, 3> before{};
  std::array<uint32_t, 3> after{};
  std::memcpy(before.data(), bytes.data() + text_file_offset, sizeof(before));
  std::memcpy(after.data(), result.replacement.data() + text_file_offset, sizeof(after));
  EXPECT_EQ(after[0], before[0]);
  EXPECT_EQ(after[1], before[1]);
  EXPECT_EQ(after[2] & 0xffu, before[2] & 0xffu);
  EXPECT_EQ(after[2] >> 8u, 4u);
}

TEST(ConSan, OrdinaryAcquireWrongAddressRejectsInvalidDeltaAndIoffsetOverflow) {
  for (const uint32_t delta : {0u, 2u, 0x800000u}) {
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.fault_ordinary_wrong_address = true;
    options.fault_ordinary_address_delta = delta;
    const ConSanTransformArtifacts result =
        test_lower_consan(make_rdna4_ordinary_acquire_code_object(), options);
    EXPECT_FALSE(result.modified());
    EXPECT_FALSE(result.errors.empty());
  }

  ConSanOptions overflow;
  overflow.flavor = ConSanFlavor::SuperCollider;
  overflow.fault_ordinary_wrong_address = true;
  overflow.fault_ordinary_address_delta = 4u;
  const ConSanTransformArtifacts result = test_lower_consan(
      make_rdna4_ordinary_acquire_code_object(2u, true, false, 0x7ffffc), overflow);
  EXPECT_FALSE(result.modified());
  EXPECT_FALSE(result.errors.empty());
}

TEST(ConSan, OrdinaryAcquireWeakenScopeChangesOnlyDeviceScopeBits) {
  const std::vector<uint8_t> bytes = make_rdna4_ordinary_acquire_code_object(/*scope=*/3u);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_ordinary_weaken_scope = true;
  options.fault_require_exactly_one = true;
  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineOrdinaryScopeRewrite);
  ASSERT_EQ(result.program_inventory.text_sections().size(), 1u);
  const size_t word1_file_offset =
      result.program_inventory.text_sections().front().file_offset + sizeof(uint32_t);
  uint32_t before = 0;
  uint32_t after = 0;
  std::memcpy(&before, bytes.data() + word1_file_offset, sizeof(before));
  std::memcpy(&after, result.replacement.data() + word1_file_offset, sizeof(after));
  EXPECT_EQ((before >> 18u) & 0x3u, 3u);
  EXPECT_EQ((after >> 18u) & 0x3u, 0u);
  EXPECT_EQ(after, before & ~(0x3u << 18u));
}

TEST(ConSan, OrdinaryAcquireOrderAndScopeComposeAsTwoExactTransactionalMutations) {
  const std::vector<uint8_t> bytes = make_rdna4_ordinary_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_ordinary_weaken_order = true;
  options.fault_ordinary_weaken_scope = true;
  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.mutation.fault.applied, 2u);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineOrdinaryOrderRewrite);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineOrdinaryScopeRewrite);
  EXPECT_EQ(result.patches[0].fault_primary_identity, result.patches[1].fault_primary_identity);
  EXPECT_EQ(result.patches[0].fault_sequence_identity, result.patches[1].fault_sequence_identity);
  EXPECT_EQ(result.mutation.applied_fault_logical_identity,
            std::optional(result.patches[0].fault_sequence_identity));
}

TEST(ConSan, OrdinaryAcquireFaultRejectsStoreNoBoundaryAlreadyWaveAndWrongIdentity) {
  const auto rejected = [](const std::vector<uint8_t> &bytes, std::string identity = {}) {
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.fault_ordinary_weaken_order = true;
    options.fault_ordinary_weaken_scope = true;
    options.fault_site_identity = std::move(identity);
    return test_lower_consan(bytes, options);
  };
  for (const std::vector<uint8_t> &bytes :
       {make_rdna4_ordinary_acquire_code_object(2u, true, true),
        make_rdna4_ordinary_acquire_code_object(2u, false, false),
        make_rdna4_ordinary_acquire_code_object(0u, true, false)}) {
    const ConSanTransformArtifacts result = rejected(bytes);
    EXPECT_FALSE(result.modified());
    EXPECT_EQ(result.mutation.fault.applied, 0u);
  }
  const ConSanTransformArtifacts wrong =
      rejected(make_rdna4_ordinary_acquire_code_object(), "not-an-exact-site");
  EXPECT_FALSE(wrong.modified());
  EXPECT_EQ(wrong.mutation.fault.applied, 0u);
}

TEST(ConSan, FinalValidationRejectsCorruptedOrdinaryScopeMutation) {
  const std::vector<uint8_t> bytes = make_rdna4_ordinary_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_ordinary_weaken_scope = true;
  const ConSanTransformArtifacts valid = test_lower_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(valid.patches.size(), 1u);

  ConSanTransformArtifacts corrupted = valid;
  const size_t word1_file_offset = valid.program_inventory.text_sections().front().file_offset +
                                   valid.patches.front().anchor_offset + sizeof(uint32_t);
  uint32_t word1 = 0;
  std::memcpy(&word1, corrupted.replacement.data() + word1_file_offset, sizeof(word1));
  word1 ^= 1u << 20u;
  std::memcpy(corrupted.replacement.data() + word1_file_offset, &word1, sizeof(word1));
  const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("fields other than the exact ordinary load scope") != std::string::npos;
  }));
}

TEST(ConSan, LargeSyncInventoryDerivesEverySequenceOwner) {
  constexpr size_t kLoadCount = 8192u;
  std::vector<uint32_t> text_words;
  text_words.reserve(3u * kLoadCount + 1u);
  for (size_t i = 0; i < kLoadCount; ++i) {
    text_words.push_back(0xEE050004u);
    text_words.push_back(7u | (2u << 18u) | (1u << 20u));
    text_words.push_back(10u | (static_cast<uint32_t>(i) << 8u));
  }
  text_words.push_back(0xBFB00000u); // s_endpgm

  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanTransformArtifacts result =
      test_lower_consan(make_rdna4_lds_code_object(text_words, "large_sync_inventory"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.sync().sync_events.size(), kLoadCount);
  ASSERT_EQ(result.program_inventory.sync().sync_sequences.size(), kLoadCount);
  ASSERT_EQ(result.fault_sites.size(), kLoadCount);
  for (const ConSanFaultSite &site : result.fault_sites)
    EXPECT_NE(test_sync_event(result, site), nullptr);
  for (const ConSanSyncSequence &sequence : result.program_inventory.sync().sync_sequences) {
    ASSERT_EQ(sequence.member_event_ids.size(), 1u);
    const std::vector<ConSanExecutionOwner> owners =
        result.program_inventory.sync().execution_owners(sequence);
    ASSERT_EQ(owners.size(), 1u);
    EXPECT_EQ(owners.front().proof, ConSanOwnerProofKind::KernelLocal);
  }
}

} // namespace
} // namespace rocjitsu
