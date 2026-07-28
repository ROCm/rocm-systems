// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"

namespace rocjitsu {
namespace {

struct PhysicalAliasTestCandidate {
  uint64_t file_offset = 0;
  std::string container_name;
  uint32_t semantics = 0;
  std::vector<uint64_t> owners;
  std::optional<uint64_t> descriptor;

  bool operator==(const PhysicalAliasTestCandidate &) const = default;
};

auto make_physical_alias_test_canonicalizer(std::vector<std::string> &errors,
                                            size_t expected_candidate_count = 0) {
  return consan_detail::make_physical_site_alias_canonicalizer<PhysicalAliasTestCandidate>(
      errors, "ConSan incremental test site",
      [](const PhysicalAliasTestCandidate &candidate) { return candidate.file_offset; },
      [](const PhysicalAliasTestCandidate &candidate) -> std::string_view {
        return candidate.container_name;
      },
      [](const PhysicalAliasTestCandidate &lhs, const PhysicalAliasTestCandidate &rhs) {
        return lhs.semantics == rhs.semantics;
      },
      [](PhysicalAliasTestCandidate &retained, const PhysicalAliasTestCandidate &alias) {
        retained.owners.insert(retained.owners.end(), alias.owners.begin(), alias.owners.end());
        retained.descriptor.reset();
      },
      expected_candidate_count);
}

TEST(ConSan, PhysicalSiteAliasCanonicalizationRetainsOrderAndRunsTypedMerge) {
  std::vector<PhysicalAliasTestCandidate> candidates{
      {.file_offset = 24u,
       .container_name = "first",
       .semantics = 3u,
       .owners = {1u},
       .descriptor = 1u},
      {.file_offset = 8u,
       .container_name = "independent",
       .semantics = 7u,
       .owners = {9u},
       .descriptor = 9u},
      {.file_offset = 24u,
       .container_name = "alias",
       .semantics = 3u,
       .owners = {2u},
       .descriptor = 2u},
      {.file_offset = 24u,
       .container_name = "second-alias",
       .semantics = 3u,
       .owners = {3u},
       .descriptor = 3u},
  };
  std::vector<std::string> errors;

  ASSERT_TRUE(consan_detail::canonicalize_physical_site_aliases(
      candidates, errors, "ConSan test site",
      [](const PhysicalAliasTestCandidate &candidate) { return candidate.file_offset; },
      [](const PhysicalAliasTestCandidate &candidate) -> std::string_view {
        return candidate.container_name;
      },
      [](const PhysicalAliasTestCandidate &lhs, const PhysicalAliasTestCandidate &rhs) {
        return lhs.semantics == rhs.semantics;
      },
      [](PhysicalAliasTestCandidate &retained, const PhysicalAliasTestCandidate &alias) {
        retained.owners.insert(retained.owners.end(), alias.owners.begin(), alias.owners.end());
        retained.descriptor.reset();
      }));

  EXPECT_TRUE(errors.empty());
  ASSERT_EQ(candidates.size(), 2u);
  EXPECT_EQ(candidates[0].file_offset, 24u);
  EXPECT_EQ(candidates[0].container_name, "first");
  EXPECT_EQ(candidates[0].owners, (std::vector<uint64_t>{1u, 2u, 3u}));
  EXPECT_FALSE(candidates[0].descriptor);
  EXPECT_EQ(candidates[1].file_offset, 8u);
  EXPECT_EQ(candidates[1].owners, (std::vector<uint64_t>{9u}));
  EXPECT_EQ(candidates[1].descriptor, 9u);
}

TEST(ConSan, PhysicalSiteAliasCanonicalizationReportsExactTypedConflict) {
  std::vector<PhysicalAliasTestCandidate> candidates{
      {.file_offset = 24u,
       .container_name = "first",
       .semantics = 3u,
       .owners = {},
       .descriptor = std::nullopt},
      {.file_offset = 24u,
       .container_name = "conflict",
       .semantics = 4u,
       .owners = {},
       .descriptor = std::nullopt},
  };
  const std::vector<PhysicalAliasTestCandidate> original = candidates;
  std::vector<std::string> errors;

  EXPECT_FALSE(consan_detail::canonicalize_physical_site_aliases(
      candidates, errors, "ConSan test site",
      [](const PhysicalAliasTestCandidate &candidate) { return candidate.file_offset; },
      [](const PhysicalAliasTestCandidate &candidate) -> std::string_view {
        return candidate.container_name;
      },
      [](const PhysicalAliasTestCandidate &lhs, const PhysicalAliasTestCandidate &rhs) {
        return lhs.semantics == rhs.semantics;
      },
      [](PhysicalAliasTestCandidate &, const PhysicalAliasTestCandidate &) {}));

  ASSERT_EQ(errors.size(), 1u);
  EXPECT_EQ(errors.front(),
            "ConSan test site at file offset 24 was decoded inconsistently through aliases "
            "'first' and 'conflict'");
  EXPECT_EQ(candidates, original);
}

TEST(ConSan, IncrementalPhysicalSiteAliasConflictIsFailFastAndTransactional) {
  std::vector<std::string> errors;
  auto canonicalizer = make_physical_alias_test_canonicalizer(errors, 4u);
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "first",
      .semantics = 3u,
      .owners = {1u},
      .descriptor = std::nullopt,
  }));
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 8u,
      .container_name = "independent",
      .semantics = 7u,
      .owners = {9u},
      .descriptor = std::nullopt,
  }));
  const std::vector<PhysicalAliasTestCandidate> accepted = canonicalizer.candidates();

  EXPECT_FALSE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "conflict",
      .semantics = 4u,
      .owners = {2u},
      .descriptor = std::nullopt,
  }));
  EXPECT_TRUE(canonicalizer.failed());
  EXPECT_EQ(canonicalizer.candidates(), accepted);
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_EQ(errors.front(),
            "ConSan incremental test site at file offset 24 was decoded inconsistently through "
            "aliases 'first' and 'conflict'");

  // A conflict is sticky: later aliases cannot mutate the accepted inventory
  // or create a cascade of secondary diagnostics.
  EXPECT_FALSE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "later-alias",
      .semantics = 3u,
      .owners = {3u},
      .descriptor = std::nullopt,
  }));
  EXPECT_TRUE(canonicalizer.failed());
  EXPECT_EQ(canonicalizer.candidates(), accepted);
  EXPECT_EQ(errors.size(), 1u);

  // A failed inventory cannot be mistaken for a complete one.
  EXPECT_FALSE(std::move(canonicalizer).take());
  EXPECT_TRUE(canonicalizer.candidates().empty());
}

TEST(ConSan, IncrementalPhysicalSiteAliasMergeRetainsOrderAndClosesAfterTake) {
  std::vector<std::string> errors;
  auto canonicalizer = make_physical_alias_test_canonicalizer(errors, 4u);
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "first",
      .semantics = 3u,
      .owners = {1u},
      .descriptor = 1u,
  }));
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 8u,
      .container_name = "independent",
      .semantics = 7u,
      .owners = {9u},
      .descriptor = 9u,
  }));
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "alias",
      .semantics = 3u,
      .owners = {2u},
      .descriptor = 2u,
  }));
  ASSERT_TRUE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "second-alias",
      .semantics = 3u,
      .owners = {3u},
      .descriptor = 3u,
  }));

  EXPECT_FALSE(canonicalizer.failed());
  std::optional<std::vector<PhysicalAliasTestCandidate>> canonical =
      std::move(canonicalizer).take();
  ASSERT_TRUE(canonical);
  EXPECT_TRUE(errors.empty());
  ASSERT_EQ(canonical->size(), 2u);
  EXPECT_EQ((*canonical)[0].container_name, "first");
  EXPECT_EQ((*canonical)[0].owners, (std::vector<uint64_t>{1u, 2u, 3u}));
  EXPECT_FALSE((*canonical)[0].descriptor);
  EXPECT_EQ((*canonical)[1].container_name, "independent");
  EXPECT_EQ((*canonical)[1].owners, (std::vector<uint64_t>{9u}));
  EXPECT_EQ((*canonical)[1].descriptor, 9u);
  EXPECT_TRUE(canonicalizer.candidates().empty());

  // Extraction permanently closes the object; later use fails safely.
  EXPECT_FALSE(canonicalizer.insert({
      .file_offset = 24u,
      .container_name = "after-take",
      .semantics = 3u,
      .owners = {5u},
      .descriptor = std::nullopt,
  }));
  EXPECT_FALSE(std::move(canonicalizer).take());
  EXPECT_TRUE(canonicalizer.candidates().empty());
}

TEST(ConSan, DisabledModeDoesNotParseCodeObject) {
  const std::vector<uint8_t> bytes = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  ConSanOptions options;
  options.flavor = ConSanFlavor::None;
  options.delay_nops = 32;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.parsed_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_EQ(result.input_size, bytes.size());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.warnings.empty());
  EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_INVALID);
  EXPECT_TRUE(result.kernels.empty());
}

TEST(ConSan, ParsesFlavorNames) {
  EXPECT_EQ(parse_consan_flavor("supercollider"), ConSanFlavor::SuperCollider);
  EXPECT_EQ(parse_consan_flavor("SUPERCOLLIDER"), ConSanFlavor::SuperCollider);
  EXPECT_EQ(parse_consan_flavor("moi"), ConSanFlavor::Moi);
  EXPECT_EQ(parse_consan_flavor("MOI"), ConSanFlavor::Moi);
  EXPECT_EQ(consan_flavor_name(ConSanFlavor::None), std::string_view("none"));
  EXPECT_EQ(consan_flavor_name(ConSanFlavor::SuperCollider), std::string_view("supercollider"));
  EXPECT_EQ(consan_flavor_name(ConSanFlavor::Moi), std::string_view("moi"));
  EXPECT_EQ(consan_transform_outcome_name(ConSanTransformOutcome::Unchanged),
            std::string_view("unchanged"));
  EXPECT_EQ(consan_transform_outcome_name(ConSanTransformOutcome::ModifiedValid),
            std::string_view("modified-valid"));
  EXPECT_EQ(consan_transform_outcome_name(ConSanTransformOutcome::Unsupported),
            std::string_view("unsupported"));
  EXPECT_EQ(consan_transform_outcome_name(ConSanTransformOutcome::Invalid),
            std::string_view("invalid"));
  EXPECT_FALSE(parse_consan_flavor(""));
  EXPECT_FALSE(parse_consan_flavor("context"));
}

TEST(ConSan, ParsesMoiEngineNamesAndAliases) {
  EXPECT_EQ(parse_consan_moi_engine("record_replay"), ConSanMoiEngine::RecordReplay);
  EXPECT_EQ(parse_consan_moi_engine("record-replay"), ConSanMoiEngine::RecordReplay);
  EXPECT_EQ(parse_consan_moi_engine("context"), ConSanMoiEngine::RecordReplay);
  EXPECT_EQ(parse_consan_moi_engine("CONTEXT"), ConSanMoiEngine::RecordReplay);
  EXPECT_EQ(parse_consan_moi_engine("inline_shadow"), ConSanMoiEngine::InlineShadow);
  EXPECT_EQ(parse_consan_moi_engine("inline-shadow"), ConSanMoiEngine::InlineShadow);
  EXPECT_EQ(parse_consan_moi_engine("sampled_watchpoint"), ConSanMoiEngine::Sampled);
  EXPECT_EQ(parse_consan_moi_engine("sampled-watchpoint"), ConSanMoiEngine::Sampled);
  EXPECT_EQ(parse_consan_moi_engine("sampled"), ConSanMoiEngine::Sampled);
  EXPECT_EQ(consan_moi_engine_name(ConSanMoiEngine::RecordReplay),
            std::string_view("record_replay"));
  EXPECT_EQ(consan_moi_engine_name(ConSanMoiEngine::InlineShadow),
            std::string_view("inline_shadow"));
  EXPECT_EQ(consan_moi_engine_name(ConSanMoiEngine::Sampled), std::string_view("sampled"));
  EXPECT_FALSE(parse_consan_moi_engine(""));
  EXPECT_FALSE(parse_consan_moi_engine("moi"));
}

TEST(ConSan, PatchedImageGrowthPolicyPreservesAbsoluteDefault) {
  ConSanPatchedImageGrowthLimit policy;
  EXPECT_EQ(policy.kind, ConSanPatchedImageGrowthLimitKind::AbsoluteBytes);
  EXPECT_EQ(policy.absolute_bytes, kConSanDefaultMaxPatchedImageGrowthBytes);
  EXPECT_EQ(consan_patched_image_growth_limit_bytes(policy, 17u),
            kConSanDefaultMaxPatchedImageGrowthBytes);
}

TEST(ConSan, RelativePatchedImageGrowthPolicyRoundsDownWithoutOverflow) {
  ConSanPatchedImageGrowthLimit policy;
  policy.kind = ConSanPatchedImageGrowthLimitKind::InputPercent;
  policy.input_percent = 25u;
  EXPECT_EQ(consan_patched_image_growth_limit_bytes(policy, 1003u), 250u);

  policy.input_percent = 200u;
  EXPECT_EQ(consan_patched_image_growth_limit_bytes(policy, std::numeric_limits<size_t>::max()),
            std::numeric_limits<size_t>::max());
}

TEST(ConSan, PatchedImageGrowthPolicyRejectsInvalidKind) {
  ConSanPatchedImageGrowthLimit policy;
  policy.kind = static_cast<ConSanPatchedImageGrowthLimitKind>(255u);
  EXPECT_FALSE(consan_patched_image_growth_limit_bytes(policy, 1003u));
}

TEST(ConSan, PatchedImageGrowthBudgetIsSharedAcrossStages) {
  ConSanPatchedImageGrowthLimit policy{
      .kind = ConSanPatchedImageGrowthLimitKind::InputPercent,
      .input_percent = 25u,
  };
  const auto budget = consan_patched_image_growth_budget(policy, 1000u, 1100u);
  ASSERT_TRUE(budget);
  EXPECT_EQ(budget->input_image_bytes, 1000u);
  EXPECT_EQ(budget->current_image_bytes, 1100u);
  EXPECT_EQ(budget->total_limit_bytes, 250u);
  EXPECT_EQ(budget->existing_growth_bytes, 100u);
  EXPECT_EQ(budget->remaining_growth_bytes, 150u);
  EXPECT_FALSE(budget->already_exceeded);

  const auto exceeded = consan_patched_image_growth_budget(policy, 1000u, 1300u);
  ASSERT_TRUE(exceeded);
  EXPECT_EQ(exceeded->remaining_growth_bytes, 0u);
  EXPECT_TRUE(exceeded->already_exceeded);
}

TEST(ConSan, SharedDiagnosticVocabularyUsesStableNames) {
  EXPECT_STREQ(
      consan_register_allocation_source_name(ConSanRegisterAllocationSource::DescriptorGrowth),
      "descriptor-growth");
  EXPECT_STREQ(consan_delay_mode_name(ConSanDelayMode::SleepVar), "sleep_var");
  EXPECT_STREQ(
      consan_barrier_operand_source_name(ConSanBarrierSite::OperandSource::StaticM0Literal32),
      "static-m0-literal32");
  EXPECT_STREQ(consan_barrier_scope_name(ConSanBarrierSite::Scope::Workgroup), "workgroup");
  EXPECT_STREQ(consan_moi_candidate_source_name(ConSanMoiCandidateSource::FlatMaybeGroup),
               "flat-maybe-group");
  EXPECT_STREQ(consan_lds_access_kind_name(ConSanLdsAccessKind::Atomic), "atomic");
  EXPECT_STREQ(consan_flat_address_space_hint_name(ConSanFlatAddressSpaceHint::MaybePrivate),
               "maybe-private");
}

TEST(ConSan, SynchronizationConfidenceCompositionNeverStrengthensInputs) {
  constexpr std::array confidences = {
      ConSanSemanticConfidence::Exact,
      ConSanSemanticConfidence::Conservative,
      ConSanSemanticConfidence::Ambiguous,
      ConSanSemanticConfidence::Unsupported,
  };
  for (size_t lhs = 0; lhs < confidences.size(); ++lhs) {
    for (size_t rhs = 0; rhs < confidences.size(); ++rhs) {
      const ConSanSemanticConfidence combined =
          combine_consan_sync_confidence(confidences[lhs], confidences[rhs]);
      EXPECT_EQ(combined, confidences[std::max(lhs, rhs)]);
      EXPECT_EQ(combined, combine_consan_sync_confidence(confidences[rhs], confidences[lhs]));
    }
  }
}

TEST(ConSan, SynchronizationConsumerContractRequiresUniqueAcceptableSequence) {
  EXPECT_TRUE(consan_sync_confidence_meets(ConSanSemanticConfidence::Exact,
                                           ConSanSemanticConfidence::Conservative));
  EXPECT_TRUE(consan_sync_confidence_meets(ConSanSemanticConfidence::Conservative,
                                           ConSanSemanticConfidence::Conservative));
  EXPECT_FALSE(consan_sync_confidence_meets(ConSanSemanticConfidence::Ambiguous,
                                            ConSanSemanticConfidence::Conservative));
  EXPECT_FALSE(consan_sync_confidence_meets(ConSanSemanticConfidence::Unsupported,
                                            ConSanSemanticConfidence::Ambiguous));

  ConSanResult result;
  ConSanSyncSequence sequence;
  sequence.identity = "sequence-a";
  sequence.member_event_identities = {"event-a", "event-b"};
  result.sync_sequences.push_back(sequence);
  ASSERT_NE(find_consan_sync_sequence_for_event(result, "event-b"), nullptr);
  EXPECT_EQ(find_consan_sync_sequence_for_event(result, "missing"), nullptr);

  sequence.identity = "sequence-b";
  sequence.member_event_identities = {"event-b"};
  result.sync_sequences.push_back(sequence);
  EXPECT_EQ(find_consan_sync_sequence_for_event(result, "event-b"), nullptr);
}

TEST(ConSan, EnabledModeRejectsInvalidCodeObject) {
  const std::vector<uint8_t> bytes = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.parsed_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_EQ(result.input_size, bytes.size());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
  EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_INVALID);
  EXPECT_TRUE(result.kernels.empty());
}

TEST(ConSan, RejectsTargetsOutsideDocumentedSupport) {
  const std::array<uint32_t, 1> text_words = {build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4)};
  struct UnsupportedTarget {
    uint32_t machine;
    rj_code_target_id_t target;
  };
  constexpr std::array unsupported_targets = {
      UnsupportedTarget{EF_AMDGPU_MACH_AMDGCN_GFX90A, ROCJITSU_CODE_TARGET_GFX90A},
      UnsupportedTarget{EF_AMDGPU_MACH_AMDGCN_GFX1200, ROCJITSU_CODE_TARGET_GFX1200},
      UnsupportedTarget{0x1234u, ROCJITSU_CODE_TARGET_INVALID},
  };

  for (const UnsupportedTarget &unsupported : unsupported_targets) {
    SCOPED_TRACE(unsupported.machine);
    std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
    mutate_elf_header(bytes,
                      [unsupported](Elf64_Ehdr &header) { header.e_flags = unsupported.machine; });
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;

    const ConSanResult result = try_patch_consan(bytes, options);

    EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
    EXPECT_TRUE(result.parsed_code_object);
    EXPECT_FALSE(result.modified);
    EXPECT_TRUE(result.errors.empty());
    EXPECT_FALSE(result.semantic_arch_required);
    EXPECT_TRUE(consan_result_has_resolved_semantic_arch(result));
    EXPECT_EQ(result.target, unsupported.target);
    EXPECT_TRUE(std::ranges::any_of(result.warnings, [&](const std::string &warning) {
      return warning == "ConSan does not support target '" +
                            std::string(rj_code_target_name(unsupported.target)) + "'";
    })) << testing::PrintToString(result.warnings);
  }
}

TEST(ConSan, SemanticArchitectureGateTracksAnalysisStageRatherThanResultMembers) {
  ConSanResult parse_only;
  parse_only.outcome = ConSanTransformOutcome::Unsupported;
  parse_only.text_sections.push_back({});
  parse_only.kernels.push_back({});
  parse_only.functions.push_back({});
  parse_only.input_fingerprint = "parsed";
  EXPECT_TRUE(consan_result_has_resolved_semantic_arch(parse_only));

  parse_only.semantic_arch_required = true;
  EXPECT_FALSE(consan_result_has_resolved_semantic_arch(parse_only));

  parse_only.arch = ROCJITSU_CODE_ARCH_RDNA4;
  EXPECT_TRUE(consan_result_has_resolved_semantic_arch(parse_only));
}

TEST(ConSan, StubRejectsEmptyCodeObject) {
  const std::vector<uint8_t> bytes;
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.parsed_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_EQ(result.input_size, 0u);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
}

TEST(ConSan, InstallActionUsesTypedOutcomeAndValidatedReplacement) {
  ConSanResult unchanged;
  unchanged.outcome = ConSanTransformOutcome::Unchanged;
  EXPECT_EQ(consan_install_action(unchanged, false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(consan_install_action(unchanged, true), ConSanInstallAction::LoadOriginal);

  ConSanResult unsupported;
  unsupported.outcome = ConSanTransformOutcome::Unsupported;
  EXPECT_EQ(consan_install_action(unsupported, false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(consan_install_action(unsupported, true), ConSanInstallAction::Reject);

  ConSanResult invalid;
  invalid.outcome = ConSanTransformOutcome::Invalid;
  EXPECT_EQ(consan_install_action(invalid, false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(consan_install_action(invalid, true), ConSanInstallAction::Reject);

  ConSanResult replacement;
  replacement.outcome = ConSanTransformOutcome::ModifiedValid;
  replacement.modified = true;
  replacement.final_validation_passed = true;
  replacement.elf_bytes.push_back(1u);
  EXPECT_EQ(consan_install_action(replacement, false), ConSanInstallAction::LoadReplacement);
  EXPECT_EQ(consan_install_action(replacement, true), ConSanInstallAction::LoadReplacement);

  replacement.final_validation_passed = false;
  EXPECT_EQ(consan_install_action(replacement, false), ConSanInstallAction::Reject);
  EXPECT_EQ(consan_install_action(replacement, true), ConSanInstallAction::Reject);
}

TEST(ConSan, MalformedCodeObjectsNeverProduceReplacementBytes) {
  const std::array<uint32_t, 13> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> valid = make_rdna4_lds_code_object(text_words);
  std::vector<std::pair<std::string, std::vector<uint8_t>>> cases;

  const std::array<uint32_t, 1> truncated_instruction_words = {0xD8340000u};
  cases.emplace_back("truncated eight-byte instruction",
                     make_rdna4_lds_code_object(truncated_instruction_words));

  cases.emplace_back("truncated ELF header",
                     std::vector<uint8_t>(valid.begin(), valid.begin() + sizeof(Elf64_Ehdr) - 1));
  std::vector<uint8_t> truncated_sections = valid;
  Elf64_Ehdr valid_header{};
  std::memcpy(&valid_header, valid.data(), sizeof(valid_header));
  truncated_sections.resize(valid_header.e_shoff + sizeof(Elf64_Shdr) - 1);
  cases.emplace_back("truncated section table", std::move(truncated_sections));

  std::vector<uint8_t> zero_section_count = valid;
  mutate_elf_header(zero_section_count, [](Elf64_Ehdr &header) { header.e_shnum = 0; });
  cases.emplace_back("zero section count", std::move(zero_section_count));

  std::vector<uint8_t> bad_section_offset = valid;
  mutate_elf_header(bad_section_offset,
                    [&](Elf64_Ehdr &header) { header.e_shoff = valid.size() + 64u; });
  cases.emplace_back("section table outside image", std::move(bad_section_offset));

  std::vector<uint8_t> bad_section_size = valid;
  mutate_elf_section(bad_section_size, 1,
                     [&](Elf64_Shdr &section) { section.sh_size = valid.size(); });
  cases.emplace_back("text range outside image", std::move(bad_section_size));

  std::vector<uint8_t> bad_symtab_entry_size = valid;
  mutate_elf_section(bad_symtab_entry_size, 3, [](Elf64_Shdr &section) { section.sh_entsize = 0; });
  cases.emplace_back("zero symbol entry size", std::move(bad_symtab_entry_size));

  std::vector<uint8_t> bad_symbol_section = valid;
  mutate_elf_symbol(bad_symbol_section, 1, [](Elf64_Sym &symbol) { symbol.st_shndx = 0xfffeu; });
  cases.emplace_back("kernel symbol has invalid section", std::move(bad_symbol_section));

  std::vector<uint8_t> oversized_symbol = valid;
  mutate_elf_symbol(oversized_symbol, 1, [](Elf64_Sym &symbol) { symbol.st_size = UINT64_MAX; });
  cases.emplace_back("kernel symbol range overflows", std::move(oversized_symbol));

  std::vector<uint8_t> short_descriptor = valid;
  mutate_elf_section(short_descriptor, 2, [](Elf64_Shdr &section) { section.sh_size = 4; });
  cases.emplace_back("kernel descriptor is truncated", std::move(short_descriptor));

  std::vector<uint8_t> misaligned_descriptor = valid;
  mutate_elf_section(misaligned_descriptor, 2, [](Elf64_Shdr &section) { section.sh_offset += 4; });
  cases.emplace_back("kernel descriptor has a misaligned file offset",
                     std::move(misaligned_descriptor));

  std::vector<uint8_t> bad_entry_offset = valid;
  mutate_first_kernel_descriptor(bad_entry_offset, [](KD &descriptor) {
    descriptor.kernel_code_entry_byte_offset = INT64_MAX;
  });
  cases.emplace_back("kernel descriptor entry is outside text", std::move(bad_entry_offset));

  const std::array<uint32_t, 4> function_words = {0xBF800000u, 0xBF800000u, 0xBF800000u,
                                                  0xBFB00000u};
  std::vector<uint8_t> overlapping_symbols =
      make_rdna4_code_object_with_local_function(function_words, function_words);
  mutate_elf_symbol(overlapping_symbols, 2, [](Elf64_Sym &symbol) { symbol.st_value = 0x1104u; });
  cases.emplace_back("partially overlapping function symbols", std::move(overlapping_symbols));

  for (const auto &profile : all_consan_transform_profiles()) {
    for (const auto &[name, bytes] : cases) {
      SCOPED_TRACE(::testing::Message() << profile.name << ": " << name);
      const ConSanResult result = try_patch_consan(bytes, profile.options);
      EXPECT_NE(result.outcome, ConSanTransformOutcome::ModifiedValid);
      EXPECT_FALSE(result.modified);
      EXPECT_FALSE(result.final_validation_passed);
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.patches.empty());
      EXPECT_EQ(consan_install_action(result, false), ConSanInstallAction::LoadOriginal);
      EXPECT_EQ(consan_install_action(result, true),
                result.outcome == ConSanTransformOutcome::Unchanged
                    ? ConSanInstallAction::LoadOriginal
                    : ConSanInstallAction::Reject);
    }
  }
}

TEST(ConSan, RejectsCodeObjectWithMalformedKernelMetadataNote) {
  const std::array<uint32_t, 4> text_words = {
      build_v_mov_b32_e32(/*vdst=*/11, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  constexpr std::array<uint8_t, 3> kRequiredWorkgroupSize{2u, 4u, 8u};
  const auto make_bytes = [&] {
    std::vector<uint8_t> bytes =
        make_rdna4_lds_code_object(text_words, "malformed_kernel_metadata");
    append_kernel_metadata_note(bytes, "malformed_kernel_metadata",
                                /*uses_dynamic_stack=*/true, /*sgpr_count=*/24u,
                                /*private_segment_fixed_size=*/0u, kRequiredWorkgroupSize,
                                /*has_dynamic_lds=*/true);
    return bytes;
  };

  struct MetadataDamageCase {
    std::string_view description;
    std::vector<uint8_t> bytes;
    size_t malformed_note_count = 0;
    std::string expected_error;
  };
  std::vector<MetadataDamageCase> cases;
  auto malformed_payload = make_bytes();
  Elf64_Ehdr header{};
  std::memcpy(&header, malformed_payload.data(), sizeof(header));
  Elf64_Phdr note_segment{};
  std::memcpy(&note_segment, malformed_payload.data() + header.e_phoff, sizeof(note_segment));
  malformed_payload[note_segment.p_offset + sizeof(Elf64_Nhdr) + 8u] = 0xc1u;
  cases.push_back({.description = "payload",
                   .bytes = std::move(malformed_payload),
                   .malformed_note_count = 1u,
                   .expected_error =
                       "ConSan cannot safely transform a code object with 1 malformed AMDGPU "
                       "kernel metadata note"});

  auto malformed_framing = make_bytes();
  std::memcpy(&header, malformed_framing.data(), sizeof(header));
  std::memcpy(&note_segment, malformed_framing.data() + header.e_phoff, sizeof(note_segment));
  Elf64_Nhdr note_header{};
  std::memcpy(&note_header, malformed_framing.data() + note_segment.p_offset, sizeof(note_header));
  note_header.n_descsz += 64u;
  std::memcpy(malformed_framing.data() + note_segment.p_offset, &note_header, sizeof(note_header));
  cases.push_back({.description = "framing",
                   .bytes = std::move(malformed_framing),
                   .malformed_note_count = 1u,
                   .expected_error =
                       "ConSan cannot safely transform a code object with 1 malformed AMDGPU "
                       "kernel metadata note"});

  auto incomplete_scan = make_bytes();
  std::memcpy(&header, incomplete_scan.data(), sizeof(header));
  std::memcpy(&note_segment, incomplete_scan.data() + header.e_phoff, sizeof(note_segment));
  note_segment.p_filesz = incomplete_scan.size();
  std::memcpy(incomplete_scan.data() + header.e_phoff, &note_segment, sizeof(note_segment));
  cases.push_back(
      {.description = "out-of-range note segment",
       .bytes = std::move(incomplete_scan),
       .expected_error =
           "ConSan cannot safely transform a code object with incomplete AMDGPU kernel metadata"});

  // Absence is accepted by InlineShadowSpillingWorksWithoutMetadata; a note
  // that claims metadata but cannot be read is rejected by every engine.
  for (const MetadataDamageCase &damage : cases) {
    AmdGpuCodeObject malformed(damage.bytes.data(), damage.bytes.size());
    ASSERT_TRUE(malformed.is_valid());
    ASSERT_FALSE(malformed.kernel_metadata_is_trustworthy());
    ASSERT_EQ(malformed.malformed_kernel_metadata_note_count(), damage.malformed_note_count);

    for (const auto &profile : all_consan_transform_profiles()) {
      SCOPED_TRACE(::testing::Message() << damage.description << ": " << profile.name);
      const ConSanResult result = try_patch_consan(damage.bytes, profile.options);
      EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
      EXPECT_FALSE(result.modified);
      EXPECT_FALSE(result.final_validation_passed);
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.patches.empty());
      EXPECT_FALSE(result.kernel_metadata_trustworthy);
      EXPECT_EQ(result.malformed_kernel_metadata_note_count, damage.malformed_note_count);
      EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_GFX1201);
      ASSERT_EQ(result.errors.size(), 1u);
      EXPECT_EQ(result.errors.front(), damage.expected_error);
      EXPECT_EQ(consan_install_action(result, false), ConSanInstallAction::LoadOriginal);
      EXPECT_EQ(consan_install_action(result, true), ConSanInstallAction::Reject);
    }
  }
}

TEST(ConSan, ReportsMultipleMalformedKernelMetadataNotes) {
  const std::array<uint32_t, 1> text_words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "multiple_malformed_metadata");
  append_kernel_metadata_note(bytes, "multiple_malformed_metadata",
                              /*uses_dynamic_stack=*/true, /*sgpr_count=*/24u);

  Elf64_Ehdr header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  Elf64_Phdr note_segment{};
  std::memcpy(&note_segment, bytes.data() + header.e_phoff, sizeof(note_segment));
  const std::vector<uint8_t> duplicate = first_note_segment_bytes(bytes);
  ASSERT_FALSE(duplicate.empty());
  const uint64_t second_note_offset = note_segment.p_offset + duplicate.size();
  bytes.insert(bytes.end(), duplicate.begin(), duplicate.end());
  note_segment.p_filesz += duplicate.size();
  note_segment.p_memsz = note_segment.p_filesz;
  std::memcpy(bytes.data() + header.e_phoff, &note_segment, sizeof(note_segment));
  bytes[note_segment.p_offset + sizeof(Elf64_Nhdr) + 8u] = 0xc1u;
  bytes[second_note_offset + sizeof(Elf64_Nhdr) + 8u] = 0xc1u;

  AmdGpuCodeObject malformed(bytes.data(), bytes.size());
  ASSERT_TRUE(malformed.is_valid());
  ASSERT_EQ(malformed.malformed_kernel_metadata_note_count(), 2u);
  ASSERT_FALSE(malformed.kernel_metadata_is_trustworthy());

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(bytes, options);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_FALSE(result.kernel_metadata_trustworthy);
  EXPECT_EQ(result.malformed_kernel_metadata_note_count, 2u);
  ASSERT_EQ(result.errors.size(), 1u);
  EXPECT_NE(result.errors.front().find("2 malformed AMDGPU kernel metadata notes"),
            std::string::npos);
}

TEST(ConSan, InfersZeroSizedKernelFunctionThroughTextEnd) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBFB00000u, // s_endpgm
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 0; });

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().has_text_range);
  EXPECT_TRUE(result.kernels.front().decoded);
  EXPECT_EQ(result.kernels.front().code_size, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(result.kernels.front().stats.lds_write_count, 1u);
}

TEST(ConSan, UsesExplicitAliasedFunctionSizeForZeroSizedKernelSymbol) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 3> trailing_padding = {};
  std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, trailing_padding);
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 0; });
  mutate_elf_symbol(bytes, 2, [](Elf64_Sym &symbol) { symbol.st_value = 0x1100u; });

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().has_text_range);
  EXPECT_TRUE(result.kernels.front().decoded);
  EXPECT_EQ(result.kernels.front().code_size, kernel_words.size() * sizeof(uint32_t));
  EXPECT_FALSE(result.kernels.front().code_size_inferred_from_zero);
  EXPECT_EQ(result.kernels.front().stats.lds_write_count, 1u);
}

TEST(ConSan, ConflictingAliasedFunctionSizesFallBackToNextDistinctEntry) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.entry_nop_words = 2u;
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto first =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto unrelated =
      std::ranges::find(original.kernels(), "unrelated_kernel", &AmdGpuKernelInfo::name);
  ASSERT_NE(first, original.kernels().end());
  ASSERT_NE(unrelated, original.kernels().end());
  ASSERT_FALSE(original.text_sections().empty());
  const uint64_t first_entry_address =
      original.text_sections().front()->vaddr() + first->entry_text_offset;

  mutate_elf_symbol_by_name(bytes, "shared_owner_0",
                            [](Elf64_Sym &symbol) { symbol.st_size = 0u; });
  mutate_elf_symbol_by_name(bytes, "shared_owner_1", [&](Elf64_Sym &symbol) {
    symbol.st_value = first_entry_address;
    symbol.st_size = first->code_size;
  });
  mutate_elf_symbol_by_name(bytes, "shared_lds_helper", [&](Elf64_Sym &symbol) {
    symbol.st_value = first_entry_address;
    symbol.st_size = first->code_size + sizeof(uint32_t);
  });

  AmdGpuCodeObject conflicted(bytes.data(), bytes.size());
  ASSERT_TRUE(conflicted.is_valid());
  const auto inferred =
      std::ranges::find(conflicted.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto next =
      std::ranges::find(conflicted.kernels(), "unrelated_kernel", &AmdGpuKernelInfo::name);
  ASSERT_NE(inferred, conflicted.kernels().end());
  ASSERT_NE(next, conflicted.kernels().end());
  EXPECT_TRUE(inferred->code_size_inferred_from_zero);
  EXPECT_EQ(inferred->code_size, next->entry_text_offset - inferred->entry_text_offset);
}

TEST(ConSan, SkipsEmptyTargetSelectionKernelAtTextEnd) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 0> empty_specialization = {};
  const std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      kernel_words, empty_specialization, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 2u);
  const auto empty = std::ranges::find(result.kernels, "lds_helper", &ConSanKernelInfo::name);
  ASSERT_NE(empty, result.kernels.end());
  EXPECT_TRUE(empty->has_text_range);
  EXPECT_EQ(empty->code_size, 0u);
  EXPECT_TRUE(empty->decoded);
  EXPECT_EQ(empty->preflight_action, ConSanPreflightAction::Skip);
  EXPECT_EQ(empty->stats.instruction_count, 0u);
}

TEST(ConSan, ExcessiveAllocatedSectionAlignmentCannotDriveTextGrowthAllocation) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBFB00000u,
  };
  const std::vector<uint8_t> valid = make_rdna4_lds_code_object(text_words);
  std::vector<std::pair<std::string, std::vector<uint8_t>>> cases;

  std::vector<uint8_t> excessive_alignment = valid;
  mutate_elf_section(excessive_alignment, 2,
                     [](Elf64_Shdr &section) { section.sh_addralign = uint64_t{1} << 58u; });
  cases.emplace_back("excessive allocated-section alignment", std::move(excessive_alignment));

  std::vector<uint8_t> overflowing_program_headers = valid;
  mutate_elf_header(overflowing_program_headers, [](Elf64_Ehdr &header) {
    header.e_phoff = UINT64_MAX;
    header.e_phentsize = sizeof(Elf64_Phdr);
    header.e_phnum = 1;
  });
  cases.emplace_back("overflowing program-header table", std::move(overflowing_program_headers));

  for (const auto &profile : all_consan_replacement_profiles()) {
    for (const auto &[name, bytes] : cases) {
      SCOPED_TRACE(::testing::Message() << profile.name << ": " << name);
      const ConSanResult result = try_patch_consan(bytes, profile.options);
      EXPECT_NE(result.outcome, ConSanTransformOutcome::ModifiedValid);
      EXPECT_FALSE(result.modified);
      EXPECT_FALSE(result.final_validation_passed);
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.patches.empty());
    }
  }
}

TEST(ConSan, BoundedElfMutationsOnlyProduceValidatedReplacementOrOriginal) {
  const std::array<uint32_t, 13> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> valid = make_rdna4_lds_code_object(text_words);
  auto expect_transactional_result = [&](std::span<const uint8_t> input,
                                         const ConSanTransformProfile &profile) {
    const ConSanResult result = try_patch_consan(input, profile.options);
    EXPECT_TRUE(result.visited_code_object);
    EXPECT_EQ(result.input_size, input.size());
    if (result.outcome == ConSanTransformOutcome::ModifiedValid) {
      EXPECT_TRUE(result.modified);
      EXPECT_TRUE(result.final_validation_passed);
      EXPECT_FALSE(result.elf_bytes.empty());
      EXPECT_FALSE(result.patches.empty());
      EXPECT_TRUE(validate_consan_modified_elf(input, result).empty());
      EXPECT_EQ(consan_install_action(result, false), ConSanInstallAction::LoadReplacement);
    } else {
      EXPECT_FALSE(result.modified);
      EXPECT_FALSE(result.final_validation_passed);
      EXPECT_TRUE(result.elf_bytes.empty());
      EXPECT_TRUE(result.patches.empty());
      EXPECT_EQ(consan_install_action(result, false), ConSanInstallAction::LoadOriginal);
      EXPECT_EQ(consan_install_action(result, true),
                result.outcome == ConSanTransformOutcome::Unchanged
                    ? ConSanInstallAction::LoadOriginal
                    : ConSanInstallAction::Reject);
    }
  };

  for (const auto &profile : all_consan_transform_profiles()) {
    for (size_t size = 0; size <= valid.size(); ++size) {
      SCOPED_TRACE(::testing::Message() << profile.name << ": truncation size " << size);
      expect_transactional_result(std::span<const uint8_t>(valid.data(), size), profile);
    }
    for (size_t offset = 0; offset < valid.size(); ++offset) {
      SCOPED_TRACE(::testing::Message()
                   << profile.name << ": single-byte mutation offset " << offset);
      std::vector<uint8_t> mutated = valid;
      mutated[offset] ^= static_cast<uint8_t>(0xA5u ^ (offset & 0xFFu));
      expect_transactional_result(mutated, profile);
    }
  }
}

TEST(ConSan, FinalStructuralValidationRediscoversReplacementIdentity) {
  const std::array<uint32_t, 13> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  for (const auto &profile : all_consan_replacement_profiles()) {
    SCOPED_TRACE(profile.name);
    const ConSanResult valid = try_patch_consan(bytes, profile.options);
    ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid);
    EXPECT_TRUE(valid.final_validation_passed);
    EXPECT_TRUE(validate_consan_modified_elf(bytes, valid).empty());

    ConSanResult wrong_target = valid;
    mutate_elf_header(wrong_target.elf_bytes, [](Elf64_Ehdr &header) { header.e_flags = 0; });
    EXPECT_FALSE(validate_consan_modified_elf(bytes, wrong_target).empty());

    ConSanResult stale_text = valid;
    mutate_elf_section(stale_text.elf_bytes, 1,
                       [&](Elf64_Shdr &section) { section.sh_size = stale_text.elf_bytes.size(); });
    EXPECT_FALSE(validate_consan_modified_elf(bytes, stale_text).empty());

    ConSanResult unaccounted_change = valid;
    unaccounted_change.elf_bytes[0x100u + 48u] ^= 1u;
    const std::vector<std::string> unaccounted_errors =
        validate_consan_modified_elf(bytes, unaccounted_change);
    ASSERT_FALSE(unaccounted_errors.empty());
    EXPECT_TRUE(std::ranges::any_of(unaccounted_errors, [](const std::string &error) {
      return error.find("unaccounted executable byte change") != std::string::npos;
    }));

    ConSanResult overlapping_inventory = valid;
    ConSanPatchInfo overlap = overlapping_inventory.patches.front();
    overlap.anchor_offset += sizeof(uint32_t);
    overlapping_inventory.patches.push_back(overlap);
    const std::vector<std::string> overlap_errors =
        validate_consan_modified_elf(bytes, overlapping_inventory);
    ASSERT_FALSE(overlap_errors.empty());
    EXPECT_TRUE(std::ranges::any_of(overlap_errors, [](const std::string &error) {
      return error.find("partially overlapping patch ranges") != std::string::npos;
    }));

    ConSanResult stale_inventory = valid;
    stale_inventory.patches.front().anchor_offset = UINT64_MAX;
    const std::vector<std::string> stale_inventory_errors =
        validate_consan_modified_elf(bytes, stale_inventory);
    ASSERT_FALSE(stale_inventory_errors.empty());
    EXPECT_TRUE(std::ranges::any_of(stale_inventory_errors, [](const std::string &error) {
      return error.find("stale or unaligned patch range") != std::string::npos;
    }));

    ConSanResult undecodable_patch = valid;
    const uint32_t invalid_instruction = 0xffffffffu;
    std::memcpy(undecodable_patch.elf_bytes.data() + 0x100u + valid.patches.front().anchor_offset,
                &invalid_instruction, sizeof(invalid_instruction));
    const std::vector<std::string> decode_errors =
        validate_consan_modified_elf(bytes, undecodable_patch);
    ASSERT_FALSE(decode_errors.empty());
    EXPECT_TRUE(std::ranges::any_of(decode_errors, [](const std::string &error) {
      return error.find("re-decode patch anchor") != std::string::npos;
    }));
  }
}

TEST(ConSan, FinalValidationScalesAcrossManyDisjointPatchRanges) {
  constexpr size_t kPatchCount = 8192u;
  std::vector<uint32_t> text_words(kPatchCount + 1u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "many_disjoint_patch_ranges");

  ConSanResult result;
  result.visited_code_object = true;
  result.input_size = bytes.size();
  result.flavor = ConSanFlavor::SuperCollider;
  result.modified = true;
  result.elf_bytes = bytes;
  AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_EQ(replacement.text_sections().size(), 1u);
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();
  const uint32_t replacement_word = build_s_nop(1, ROCJITSU_CODE_ARCH_RDNA4);
  result.patches.reserve(kPatchCount);
  for (size_t index = 0; index < kPatchCount; ++index) {
    const uint64_t anchor = index * sizeof(uint32_t);
    std::memcpy(result.elf_bytes.data() + text_file_offset + anchor, &replacement_word,
                sizeof(replacement_word));
    ConSanPatchInfo patch;
    patch.kind = ConSanPatchKind::InlineNopRewrite;
    patch.anchor_offset = anchor;
    patch.original_size = sizeof(uint32_t);
    result.patches.push_back(std::move(patch));
  }

  EXPECT_TRUE(validate_consan_modified_elf(bytes, result).empty());
}

} // namespace
} // namespace rocjitsu
