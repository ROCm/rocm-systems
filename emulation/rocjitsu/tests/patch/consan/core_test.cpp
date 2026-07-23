// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

TEST(ConSan, DisabledModeDoesNotParseCodeObject) {
  const std::vector<uint8_t> bytes = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  ConSanOptions options;
  options.flavor = ConSanFlavor::None;
  options.delay_nops = 32;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_EQ(result.input_size, bytes.size());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(result.warnings.empty());
  EXPECT_TRUE(result.target_name.empty());
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
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_EQ(result.input_size, bytes.size());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
  EXPECT_TRUE(result.target_name.empty());
  EXPECT_TRUE(result.kernels.empty());
}

TEST(ConSan, RejectsTargetsOutsideDocumentedSupport) {
  const std::array<uint32_t, 1> text_words = {build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4)};
  constexpr std::array unsupported_targets = {
      EF_AMDGPU_MACH_AMDGCN_GFX90A,
      EF_AMDGPU_MACH_AMDGCN_GFX1200,
  };

  for (uint32_t target : unsupported_targets) {
    SCOPED_TRACE(target);
    std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
    mutate_elf_header(bytes, [target](Elf64_Ehdr &header) { header.e_flags = target; });
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;

    const ConSanResult result = try_patch_consan(bytes, options);

    EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
    EXPECT_FALSE(result.modified);
    EXPECT_TRUE(result.errors.empty());
    EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
      return warning.starts_with("ConSan does not support target '");
    }));
  }
}

TEST(ConSan, StubRejectsEmptyCodeObject) {
  const std::vector<uint8_t> bytes;
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.visited_code_object);
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
