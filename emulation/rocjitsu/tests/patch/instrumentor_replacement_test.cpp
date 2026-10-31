// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "../dbi_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/builders/instruction_builder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;

std::vector<uint8_t> instruction_bytes(std::initializer_list<uint32_t> words) {
  std::vector<uint8_t> bytes(words.size() * sizeof(uint32_t));
  std::memcpy(bytes.data(), words.begin(), bytes.size());
  return bytes;
}

InstrumentationPoint replacement_point(uint64_t offset, std::initializer_list<uint32_t> expected,
                                       std::initializer_list<uint32_t> replacement) {
  return {.anchor_offset = offset,
          .kind = InstrumentationKind::ReplaceWithInst,
          .probe_symbol = {},
          .probe_args = {},
          .replacement =
              InstructionReplacement{.expected_bytes = instruction_bytes(expected),
                                     .replacement_bytes = instruction_bytes(replacement)}};
}

void expect_no_output(const InstrumentedCodeObject &result) {
  EXPECT_FALSE(result.errors.empty());
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.sites.empty());
}

TEST(InstrumentorReplacement, OwnsQueuedPayloadAndReportsInPlaceRanges) {
  const uint32_t original = build_s_nop(1, kArch);
  const uint32_t replacement = build_s_nop(2, kArch);
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({original, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor instrumentor(source, kArch);
  InstrumentationPoint point = replacement_point(0, {original}, {replacement});
  instrumentor.add_point(point);
  point.replacement->expected_bytes[0] ^= 1;
  point.replacement->replacement_bytes.assign(4, 0xff);

  const Instrumentor::ValidationResult resolved = instrumentor.validate_points();
  ASSERT_TRUE(resolved.errors.empty()) << resolved.errors.front();
  ASSERT_EQ(resolved.sites.size(), 1u);
  EXPECT_EQ(resolved.sites.front().original_bytes, instruction_bytes({original}));
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  EXPECT_FALSE(result.edit_error);
  EXPECT_TRUE(result.patches.empty());
  ASSERT_EQ(result.sites.size(), 1u);
  const InstrumentedSite &site = result.sites.front();
  EXPECT_EQ(site.kind, InstrumentationKind::ReplaceWithInst);
  EXPECT_EQ(site.original_offset, 0u);
  EXPECT_EQ(site.original_size, 4u);
  EXPECT_EQ(site.instrumentation_offset, 0u);
  EXPECT_EQ(site.instrumentation_size, 4u);
  EXPECT_EQ(site.guest_size, 0u);
  std::vector<uint8_t> expected = image;
  std::memcpy(expected.data() + source.text_sections().front()->sectionOffset(), &replacement, 4);
  EXPECT_EQ(result.elf_bytes, expected);
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);

  expect_no_output(instrumentor.patch());
}

TEST(InstrumentorReplacement, ForkSharesFactsButNotRequestsOrPatchBudget) {
  const uint32_t nop = build_s_nop(0, kArch);
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({nop, nop, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  std::unique_ptr<Instrumentor> first;
  std::unique_ptr<Instrumentor> second;
  const InstructionBoundaries *shared_facts = nullptr;
  TextEditError error;
  {
    Instrumentor parent(source, kArch);
    shared_facts = parent.instruction_boundaries(error);
    ASSERT_NE(shared_facts, nullptr) << error.message;
    parent.add_point(replacement_point(0, {nop}, {build_s_nop(1, kArch)}));
    FailureOr<std::unique_ptr<Instrumentor>> child = parent.fork(error);
    ASSERT_TRUE(child.succeeded()) << error.message;
    first = std::move(child).value();
    const InstrumentedCodeObject parent_result = parent.patch();
    ASSERT_TRUE(parent_result.errors.empty()) << parent_result.errors.front();
    child = parent.fork(error);
    ASSERT_TRUE(child.succeeded()) << error.message;
    second = std::move(child).value();
  }
  // Both children retain the same index after the parent is gone. Their queues
  // start empty even though the parent had a request and consumed its patch budget.
  EXPECT_EQ(first->instruction_boundaries(error), shared_facts);
  EXPECT_EQ(second->instruction_boundaries(error), shared_facts);
  first->add_point(replacement_point(0, {nop}, {build_s_nop(2, kArch)}));
  second->add_point(replacement_point(4, {nop}, {build_s_nop(3, kArch)}));
  const InstrumentedCodeObject first_result = first->patch();
  const InstrumentedCodeObject second_result = second->patch();
  ASSERT_TRUE(first_result.errors.empty()) << first_result.errors.front();
  ASSERT_TRUE(second_result.errors.empty()) << second_result.errors.front();
  ASSERT_EQ(first_result.sites.size(), 1u);
  ASSERT_EQ(second_result.sites.size(), 1u);
  std::vector<uint8_t> first_expected = image;
  std::vector<uint8_t> second_expected = image;
  const uint64_t text = source.text_sections().front()->sectionOffset();
  const uint32_t first_word = build_s_nop(2, kArch);
  const uint32_t second_word = build_s_nop(3, kArch);
  std::memcpy(first_expected.data() + text, &first_word, sizeof(first_word));
  std::memcpy(second_expected.data() + text + 4, &second_word, sizeof(second_word));
  EXPECT_EQ(first_result.elf_bytes, first_expected);
  EXPECT_EQ(second_result.elf_bytes, second_expected);
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);

  const AmdGpuCodeObject mutant(first_result.elf_bytes.data(), first_result.elf_bytes.size());
  Instrumentor fresh_mutant(mutant, kArch);
  const InstructionBoundaries *mutant_facts = fresh_mutant.instruction_boundaries(error);
  ASSERT_NE(mutant_facts, nullptr) << error.message;
  EXPECT_NE(mutant_facts, shared_facts);
  fresh_mutant.add_point(replacement_point(0, {nop}, {build_s_nop(4, kArch)}));
  const InstrumentedCodeObject stale = fresh_mutant.patch();
  expect_no_output(stale);
  ASSERT_TRUE(stale.edit_error);
  EXPECT_EQ(stale.edit_error->code, TextEditErrorCode::StaleBytes);

  constexpr rj_code_arch_t other_arch = ROCJITSU_CODE_ARCH_RDNA4;
  const std::vector<uint8_t> other_image =
      test::make_gfx1200_kernel_elf({build_s_nop(0, other_arch), build_s_endpgm(other_arch)}, 0);
  const AmdGpuCodeObject other_source(other_image.data(), other_image.size());
  Instrumentor other_profile(other_source, other_arch);
  const InstructionBoundaries *other_facts = other_profile.instruction_boundaries(error);
  ASSERT_NE(other_facts, nullptr) << error.message;
  EXPECT_NE(other_facts, shared_facts);
  Instrumentor mismatched_profile(source, other_arch);
  EXPECT_TRUE(mismatched_profile.fork(error).failed());
  EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedTarget);
}

TEST(InstrumentorReplacement, ForkOwnsCopiedCfgFactsAfterObservationalParentDies) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t nop = build_s_nop(0, arch);
  // Dispatch skips the clause at 8 and its member at 12. The unreachable branch
  // at 20 still references that member, splitting the full-text CFG's clause run.
  const std::vector<uint8_t> image =
      test::make_gfx1200_kernel_elf({nop, build_s_branch(2, arch), 0xbf850000u, nop,
                                     build_s_endpgm(arch), build_s_branch(-3, arch)},
                                    0);
  const AmdGpuCodeObject source(image.data(), image.size());
  std::unique_ptr<Instrumentor> child;
  std::unique_ptr<Instrumentor> blocked_child;
  const InstructionBoundaries *shared_facts = nullptr;
  TextEditError error;
  {
    Instrumentor parent(source, arch);
    parent.add_point_by_offset(0);
    const Instrumentor::ValidationResult validated = parent.validate_points();
    ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
    ASSERT_EQ(validated.sites.size(), 1u);
    // Validation built the ordinary CFG. Fork now takes the trusted-index-copy
    // path, rather than the standalone linear scan used by replacement-only seeds.
    FailureOr<std::unique_ptr<Instrumentor>> batch = parent.fork(error);
    ASSERT_TRUE(batch.succeeded()) << error.message;
    child = std::move(batch).value();
    shared_facts = child->instruction_boundaries(error);
    ASSERT_NE(shared_facts, nullptr) << error.message;
    const InstrumentedCodeObject parent_result = parent.patch();
    ASSERT_TRUE(parent_result.errors.empty()) << parent_result.errors.front();
    EXPECT_GT(parent_result.elf_bytes.size(), image.size());
    batch = parent.fork(error);
    ASSERT_TRUE(batch.succeeded()) << error.message;
    blocked_child = std::move(batch).value();
  }

  ASSERT_EQ(child->instruction_boundaries(error), shared_facts);
  EXPECT_EQ(blocked_child->instruction_boundaries(error), shared_facts);
  const InstructionBoundaries::Entry *clause = shared_facts->find(8);
  const InstructionBoundaries::Entry *member = shared_facts->find(12);
  ASSERT_NE(clause, nullptr);
  ASSERT_NE(member, nullptr);
  EXPECT_TRUE(clause->clause);
  EXPECT_TRUE(member->clause_member);
  const std::span<const uint64_t> references = shared_facts->references();
  EXPECT_NE(std::ranges::find(references, 12u), references.end());
  EXPECT_NE(std::ranges::find(references, 16u), references.end());

  blocked_child->add_point(replacement_point(12, {nop}, {build_s_nop(1, arch)}));
  const InstrumentedCodeObject rejected = blocked_child->patch();
  expect_no_output(rejected);
  ASSERT_TRUE(rejected.edit_error);
  EXPECT_EQ(rejected.edit_error->code, TextEditErrorCode::Clause);
  EXPECT_EQ(rejected.edit_error->offset, 12u);

  const uint32_t replacement = build_s_nop(2, arch);
  child->add_point(replacement_point(0, {nop}, {replacement}));
  const InstrumentedCodeObject result = child->patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 1u);
  std::vector<uint8_t> expected = image;
  std::memcpy(expected.data() + source.text_sections().front()->sectionOffset(), &replacement, 4);
  EXPECT_EQ(result.elf_bytes, expected);
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
}

TEST(InstrumentorReplacement, DoesNotRequireRelocatableAnchorsOrReplayGuestInstructions) {
  const uint32_t getpc = build_s_getpc_b64(0, kArch);
  const uint32_t branch = build_s_branch(1, kArch);
  const uint32_t nop = build_s_nop(0, kArch);
  const uint32_t end = build_s_endpgm(kArch);
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({getpc, branch, nop, end}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor observational(source, kArch);
  observational.add_point_by_offset(0);
  expect_no_output(observational.patch());

  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point(replacement_point(0, {getpc}, {nop}));
  instrumentor.add_point(replacement_point(4, {branch}, {nop}));
  instrumentor.add_point(replacement_point(12, {end}, {nop}));
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 3u);
  EXPECT_TRUE(result.patches.empty());
  std::vector<uint8_t> expected = image;
  for (const uint64_t offset : {0u, 4u, 12u})
    std::memcpy(expected.data() + source.text_sections().front()->sectionOffset() + offset, &nop,
                sizeof(nop));
  EXPECT_EQ(result.elf_bytes, expected);
  for (const InstrumentedSite &site : result.sites)
    EXPECT_EQ(site.guest_size, 0u);
}

TEST(InstrumentorReplacement, DoesNotApplyObservationalBranchStackGate) {
  const uint32_t nop = build_s_nop(0, kArch);
  const uint32_t changed = build_s_nop(1, kArch);
  // s_cbranch_join elsewhere in .text makes observational liveness unsupported.
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({nop, 0xbe802e22u, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor observational(source, kArch);
  observational.add_point_by_offset(0);
  expect_no_output(observational.patch());

  Instrumentor instrumentor(source, kArch);
  TextEditError error;
  const InstructionBoundaries *facts = instrumentor.instruction_boundaries(error);
  ASSERT_NE(facts, nullptr) << error.message;
  ASSERT_NE(facts->find(4), nullptr);
  instrumentor.add_point(replacement_point(0, {nop}, {changed}));
  const InstrumentedCodeObject result = instrumentor.patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  std::vector<uint8_t> expected = image;
  std::memcpy(expected.data() + source.text_sections().front()->sectionOffset(), &changed, 4);
  EXPECT_EQ(result.elf_bytes, expected);
}

TEST(InstrumentorReplacement, DoesNotRequireProbeRegisterOrScratchResources) {
  const uint32_t nop = build_s_nop(0, kArch);
  // Eight declared SGPRs cannot supply a callable probe's s[30:31] link pair.
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({nop, build_s_endpgm(kArch)}, 0, 0);
  const std::vector<uint8_t> probe_image =
      test::make_gfx950_probe_elf("probe", {build_s_setpc_b64(30, kArch)});
  const AmdGpuCodeObject source(image.data(), image.size());
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  Instrumentor observational(source, kArch);
  InstrumentationPoint probe_point = {
      .probe_obj = &probe, .probe_symbol = "probe", .probe_args = {}};
  observational.add_point(probe_point);
  expect_no_output(observational.patch());

  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point(replacement_point(0, {nop}, {build_s_nop(1, kArch)}));
  const InstrumentedCodeObject result = instrumentor.patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  EXPECT_EQ(result.elf_bytes.size(), image.size());
  EXPECT_EQ(result.resources.sgprs, 0u);
  EXPECT_EQ(result.resources.vgprs, 0u);
  EXPECT_EQ(result.resources.kernarg_bytes, 0u);
}

TEST(InstrumentorReplacement, LateStaleBytesFailValidationAndEmissionWithoutPartialSites) {
  const uint32_t nop = build_s_nop(0, kArch);
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({nop, nop, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point(replacement_point(0, {nop}, {build_s_nop(1, kArch)}));
  instrumentor.add_point(replacement_point(4, {build_s_nop(2, kArch)}, {nop}));
  const Instrumentor::ValidationResult resolved = instrumentor.validate_points();
  EXPECT_FALSE(resolved.errors.empty());
  EXPECT_TRUE(resolved.sites.empty());
  ASSERT_TRUE(resolved.edit_error);
  EXPECT_EQ(resolved.edit_error->code, TextEditErrorCode::StaleBytes);
  EXPECT_EQ(resolved.edit_error->offset, 4u);
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  expect_no_output(result);
  EXPECT_TRUE(result.patches.empty());
  ASSERT_TRUE(result.edit_error);
  EXPECT_EQ(result.edit_error->code, TextEditErrorCode::StaleBytes);
  EXPECT_EQ(result.edit_error->offset, 4u);
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
}

TEST(InstrumentorReplacement, RejectsPhysicalOverlapBeforePublishingAnySite) {
  const uint32_t nop = build_s_nop(0, kArch);
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({0xbe8000ffu, 0x12345678u, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point(replacement_point(0, {0xbe8000ffu, 0x12345678u}, {nop, nop}));
  instrumentor.add_point(replacement_point(4, {0x12345678u}, {nop}));
  const InstrumentedCodeObject result = instrumentor.patch();
  expect_no_output(result);
  ASSERT_TRUE(result.edit_error);
  EXPECT_EQ(result.edit_error->code, TextEditErrorCode::Overlap);
  EXPECT_EQ(result.edit_error->offset, 4u);
}

TEST(InstrumentorReplacement, RejectsMixedReplacementAndProbeInEitherQueueOrder) {
  const uint32_t nop = build_s_nop(0, kArch);
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({nop, nop, build_s_endpgm(kArch)}, 0);
  const std::vector<uint8_t> probe_image =
      test::make_gfx950_probe_elf("probe", {build_s_setpc_b64(30, kArch)});
  const AmdGpuCodeObject source(image.data(), image.size());
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  for (const bool replacement_first : {false, true}) {
    SCOPED_TRACE(replacement_first);
    Instrumentor instrumentor(source, kArch);
    InstrumentationPoint probe_point = {
        .anchor_offset = 4, .probe_obj = &probe, .probe_symbol = "probe", .probe_args = {}};
    const InstrumentationPoint replace = replacement_point(0, {nop}, {build_s_nop(1, kArch)});
    instrumentor.add_point(replacement_first ? replace : probe_point);
    instrumentor.add_point(replacement_first ? probe_point : replace);
    const Instrumentor::ValidationResult resolved = instrumentor.validate_points();
    EXPECT_FALSE(resolved.errors.empty());
    EXPECT_TRUE(resolved.sites.empty());
    expect_no_output(instrumentor.patch());
  }
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
}

TEST(InstrumentorReplacement, RejectsMissingPayloadAndObservationalFields) {
  const uint32_t nop = build_s_nop(0, kArch);
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  for (int mode = 0; mode < 6; ++mode) {
    SCOPED_TRACE(mode);
    InstrumentationPoint point = replacement_point(0, {nop}, {nop});
    switch (mode) {
    case 0:
      point.replacement.reset();
      break;
    case 1:
      point.kind = InstrumentationKind::BeforeInst;
      break;
    case 2:
      point.probe_symbol = "unused";
      break;
    case 3:
      point.probe_args.push_back(probe_arg_imm(1));
      break;
    case 4:
      point.force_full_exec = true;
      break;
    case 5:
      point.filter_flags = 1;
      break;
    }
    Instrumentor instrumentor(source, kArch);
    instrumentor.add_point(point);
    expect_no_output(instrumentor.patch());
  }
}

TEST(InstrumentorReplacement, CannotReuseExpectedBytesFromAnotherSource) {
  const uint32_t original = build_s_nop(1, kArch);
  const uint32_t different = build_s_nop(2, kArch);
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({different, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point(replacement_point(0, {original}, {build_s_nop(0, kArch)}));
  const InstrumentedCodeObject result = instrumentor.patch();
  expect_no_output(result);
  ASSERT_TRUE(result.edit_error);
  EXPECT_EQ(result.edit_error->code, TextEditErrorCode::StaleBytes);
  EXPECT_EQ(result.edit_error->offset, 0u);
  EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
}

TEST(InstrumentorReplacement, ForkRetainsUnsupportedObservationalOptions) {
  const uint32_t nop = build_s_nop(0, kArch);
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, build_s_endpgm(kArch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    InstrumentationOptions options;
    switch (mode) {
    case 0:
      options.kernarg_binding = DbiKernargBinding::CallerSuppliedWrapper;
      break;
    case 1:
      options.preservation = GuestStatePreservation::DeclaredRegisters;
      break;
    case 2:
      options.register_growth = OriginalRegisterCounts{.sgprs = 32, .vgprs = 4};
      break;
    }
    Instrumentor parent(source, kArch, options);
    TextEditError error;
    FailureOr<std::unique_ptr<Instrumentor>> batch = parent.fork(error);
    ASSERT_TRUE(batch.succeeded()) << error.message;
    batch.value()->add_point(replacement_point(0, {nop}, {build_s_nop(1, kArch)}));
    const InstrumentedCodeObject result = batch.value()->patch();
    expect_no_output(result);
    ASSERT_TRUE(result.edit_error);
    EXPECT_EQ(result.edit_error->code, TextEditErrorCode::InvalidRange);
  }
}

TEST(InstrumentorReplacement, MutantReceivesFreshCallableProbeAnalysis) {
  const uint32_t nop = build_s_nop(0, kArch);
  const uint32_t original = test::kMovS8Zero;
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({original, nop, build_s_endpgm(kArch)}, 0);
  std::vector<uint8_t> mutant_image;
  {
    const AmdGpuCodeObject source(image.data(), image.size());
    Instrumentor replacement(source, kArch);
    replacement.add_point(replacement_point(0, {original}, {nop}));
    InstrumentedCodeObject result = replacement.patch();
    ASSERT_TRUE(result.errors.empty()) << result.errors.front();
    mutant_image = std::move(result.elf_bytes);
  }
  // Both the pristine object and its Instrumentor are gone before analysis of
  // the mutant. The callable path must relocate the new NOP, not the old move.
  const AmdGpuCodeObject mutant(mutant_image.data(), mutant_image.size());
  const std::vector<uint8_t> probe_image =
      test::make_gfx950_probe_elf("probe", {build_s_setpc_b64(30, kArch)});
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  Instrumentor instrumentor(mutant, kArch);
  InstrumentationPoint point = {.probe_obj = &probe, .probe_symbol = "probe", .probe_args = {}};
  instrumentor.add_point(point);
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.patches.size(), 1u);
  ASSERT_EQ(result.sites.size(), 1u);
  EXPECT_TRUE(result.patches.front().is_probe_call);
  EXPECT_EQ(result.patches.front().original_bytes, instruction_bytes({nop}));
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  const std::vector<uint32_t> words = test::section_words(candidate, ".text");
  EXPECT_EQ(words.at(result.sites.front().guest_offset / 4), nop);
  EXPECT_GT(candidate.text_sections().front()->size(), mutant.text_sections().front()->size());
  EXPECT_EQ(std::memcmp(mutant.image_data(), mutant_image.data(), mutant_image.size()), 0);
}

TEST(InstrumentorReplacement, ConcreteVariantControlsSourceAndReplacementDecoding) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  const uint32_t nop = build_s_nop(0, arch);
  for (const uint32_t machine : {EF_AMDGPU_MACH_AMDGCN_GFX1250, EF_AMDGPU_MACH_AMDGCN_GFX1251}) {
    for (const bool variant_in_source : {false, true}) {
      SCOPED_TRACE(machine);
      SCOPED_TRACE(variant_in_source);
      // gfx1251-only VOP3P, also used by the callable-probe concrete-decoder tests.
      const std::vector<uint32_t> words =
          variant_in_source ? std::vector<uint32_t>{0xcc4c4004u, 0x1a021908u, build_s_endpgm(arch)}
                            : std::vector<uint32_t>{0xbe8000ffu, 0x12345678u, build_s_endpgm(arch)};
      const std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, machine);
      const AmdGpuCodeObject source(image.data(), image.size());
      Instrumentor instrumentor(source, arch);
      instrumentor.add_point(
          variant_in_source
              ? replacement_point(0, {0xcc4c4004u, 0x1a021908u}, {nop, nop})
              : replacement_point(0, {0xbe8000ffu, 0x12345678u}, {0xcc4c4004u, 0x1a021908u}));
      const InstrumentedCodeObject result = instrumentor.patch();
      if (machine == EF_AMDGPU_MACH_AMDGCN_GFX1250) {
        expect_no_output(result);
        ASSERT_TRUE(result.edit_error);
        EXPECT_EQ(result.edit_error->code, TextEditErrorCode::InvalidInstruction);
      } else {
        ASSERT_TRUE(result.errors.empty()) << result.errors.front();
        const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
        EXPECT_EQ(candidate.target_id(), ROCJITSU_CODE_TARGET_GFX1251);
        EXPECT_EQ(result.elf_bytes.size(), image.size());
      }
    }
  }
}

TEST(InstrumentorReplacement, RejectsArchitectureOverrideForConcreteAndLegacySources) {
  for (const uint32_t machine : {EF_AMDGPU_MACH_AMDGCN_GFX908, EF_AMDGPU_MACH_AMDGCN_GFX950}) {
    const rj_code_arch_t source_arch = arch_for_elf_mach(machine);
    const rj_code_arch_t wrong_arch = source_arch == kArch ? ROCJITSU_CODE_ARCH_CDNA1 : kArch;
    const uint32_t nop = build_s_nop(0, source_arch);
    const std::vector<uint8_t> image =
        test::make_amdgpu_kernel_elf({nop, build_s_endpgm(source_arch)}, 0, 3, machine);
    const AmdGpuCodeObject source(image.data(), image.size());
    Instrumentor instrumentor(source, wrong_arch);
    instrumentor.add_point(replacement_point(0, {nop}, {nop}));
    expect_no_output(instrumentor.patch());
  }
}

TEST(InstrumentorReplacement, FullTextFactsRetainClauseMembershipAcrossUnreachableGap) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t nop = build_s_nop(0, arch);
  const std::vector<uint8_t> image = test::make_gfx1200_kernel_elf(
      {build_s_branch(2, arch), 0xbf850001u, nop, nop, build_s_endpgm(arch)}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor instrumentor(source, arch);
  TextEditError error;
  const InstructionBoundaries *facts = instrumentor.instruction_boundaries(error);
  ASSERT_NE(facts, nullptr) << error.message;
  EXPECT_EQ(instrumentor.instruction_boundaries(error), facts);
  const InstructionBoundaries::Entry *entry = facts->find(12);
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->clause_member);
  instrumentor.add_point(replacement_point(12, {nop}, {build_s_nop(1, arch)}));
  const InstrumentedCodeObject result = instrumentor.patch();
  expect_no_output(result);
  ASSERT_TRUE(result.edit_error);
  EXPECT_EQ(result.edit_error->code, TextEditErrorCode::Clause);
  EXPECT_EQ(result.edit_error->offset, 12u);
}

} // namespace
} // namespace rocjitsu
