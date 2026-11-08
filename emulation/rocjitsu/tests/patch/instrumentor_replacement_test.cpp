// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "../dbi_test_util.h"
#include "replacement_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/operand_types.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/result.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
constexpr util::FailureOr<uint32_t> kNop0Encoding = build_s_nop(0, kArch);
static_assert(kNop0Encoding.succeeded());
constexpr uint32_t kNop0Word = kNop0Encoding.value();
constexpr util::FailureOr<uint32_t> kNop1Encoding = build_s_nop(1, kArch);
static_assert(kNop1Encoding.succeeded());
constexpr uint32_t kNop1Word = kNop1Encoding.value();
constexpr util::FailureOr<uint32_t> kNop2Encoding = build_s_nop(2, kArch);
static_assert(kNop2Encoding.succeeded());
constexpr uint32_t kNop2Word = kNop2Encoding.value();
constexpr util::FailureOr<uint32_t> kNop3Encoding = build_s_nop(3, kArch);
static_assert(kNop3Encoding.succeeded());
constexpr uint32_t kNop3Word = kNop3Encoding.value();
constexpr util::FailureOr<uint32_t> kNop4Encoding = build_s_nop(4, kArch);
static_assert(kNop4Encoding.succeeded());
constexpr uint32_t kNop4Word = kNop4Encoding.value();
constexpr util::FailureOr<uint32_t> kEndpgmEncoding = build_s_endpgm(kArch);
static_assert(kEndpgmEncoding.succeeded());
constexpr uint32_t kEndpgmWord = kEndpgmEncoding.value();
constexpr util::FailureOr<uint32_t> kGetpcEncoding = build_s_getpc_b64(0, kArch);
static_assert(kGetpcEncoding.succeeded());
constexpr uint32_t kGetpcWord = kGetpcEncoding.value();
constexpr util::FailureOr<uint32_t> kBranch1Encoding = build_s_branch(1, kArch);
static_assert(kBranch1Encoding.succeeded());
constexpr uint32_t kBranch1Word = kBranch1Encoding.value();
constexpr util::FailureOr<uint32_t> kSetpc30Encoding = build_s_setpc_b64(30, kArch);
static_assert(kSetpc30Encoding.succeeded());
constexpr uint32_t kSetpc30Word = kSetpc30Encoding.value();

std::vector<uint8_t> instruction_bytes(std::initializer_list<uint32_t> words) {
  std::vector<uint8_t> bytes(words.size() * sizeof(uint32_t));
  if (!bytes.empty())
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

void expect_pure_replacements(const AmdGpuCodeObject &source, const InstrumentedCodeObject &result,
                              std::initializer_list<test::ExpectedReplacement> edits,
                              rj_code_arch_t arch = kArch) {
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  const std::span<const test::ExpectedReplacement> requests(edits.begin(), edits.size());
  const std::optional<std::vector<uint8_t>> expected =
      test::expected_replacement_image(source, requests, arch);
  ASSERT_TRUE(expected);
  EXPECT_EQ(result.elf_bytes, *expected);
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(candidate.is_valid());
  EXPECT_EQ(candidate.target_id(), source.target_id());
  const AmdGpuCodeObject expected_object(expected->data(), expected->size());
  const Section &text = *expected_object.text_sections().front();
  ASSERT_EQ(result.sites.size(), requests.size());
  for (size_t index = 0; index < requests.size(); ++index) {
    const test::ExpectedReplacement &edit = requests[index];
    const InstrumentedSite &site = result.sites[index];
    uint32_t branch = 0;
    std::memcpy(&branch, text.data() + edit.offset, sizeof(branch));
    int64_t displacement = branch & 0xffffu;
    if (displacement >= 0x8000)
      displacement -= 0x10000;
    const uint64_t cave =
        static_cast<uint64_t>(static_cast<int64_t>(edit.offset) + 4 + displacement * 4);
    EXPECT_EQ(site.kind, InstrumentationKind::ReplaceWithInst);
    EXPECT_EQ(site.original_offset, edit.offset);
    EXPECT_EQ(site.original_size, edit.expected.size());
    EXPECT_EQ(site.guest_offset, cave);
    EXPECT_EQ(site.guest_size, edit.replacement.size());
    EXPECT_EQ(site.instrumentation_offset, cave);
    EXPECT_EQ(site.instrumentation_size, 0u);
  }
}

void expect_replacement_guest(const InstrumentedCodeObjectDebug &result, uint64_t source_offset,
                              uint32_t original_size, std::span<const uint8_t> replacement,
                              rj_code_arch_t arch, bool has_probe) {
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 1u);
  ASSERT_EQ(result.patches.size(), 1u);
  const InstrumentedSite &site = result.sites.front();
  EXPECT_EQ(site.kind, InstrumentationKind::ReplaceWithInst);
  EXPECT_EQ(site.original_offset, source_offset);
  EXPECT_EQ(site.original_size, original_size);
  EXPECT_EQ(site.guest_size, replacement.size());
  EXPECT_EQ(site.instrumentation_offset + site.instrumentation_size, site.guest_offset);
  if (has_probe) {
    EXPECT_GT(site.instrumentation_size, 0u);
  } else {
    EXPECT_EQ(site.instrumentation_size, 0u);
  }
  const InstrumentationPatch &patch = result.patches.front();
  EXPECT_EQ(patch.is_probe_call, has_probe);
  EXPECT_EQ(patch.trampoline_offset, site.instrumentation_offset);
  const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(candidate.is_valid());
  const Section &text = *candidate.text_sections().front();
  ASSERT_LE(site.guest_offset + replacement.size() + 4, text.size());
  if (!replacement.empty()) {
    EXPECT_EQ(std::memcmp(text.data() + site.guest_offset, replacement.data(), replacement.size()),
              0);
  }
  const std::optional<uint32_t> entry_branch =
      test::expected_replacement_branch(source_offset, site.instrumentation_offset, arch);
  ASSERT_TRUE(entry_branch);
  uint32_t actual = 0;
  std::memcpy(&actual, text.data() + source_offset, sizeof(actual));
  EXPECT_EQ(actual, *entry_branch);
  const std::optional<uint32_t> return_branch = test::expected_replacement_branch(
      site.guest_offset + replacement.size(), source_offset + original_size, arch);
  ASSERT_TRUE(return_branch);
  std::memcpy(&actual, text.data() + site.guest_offset + replacement.size(), sizeof(actual));
  EXPECT_EQ(actual, *return_branch);
  ASSERT_TRUE(patch.return_target);
  EXPECT_EQ(*patch.return_target, source_offset + original_size);
}

void check_wide_entry_source_attribution(bool with_entry, bool with_replacement) {
  struct SourceCase {
    rj_code_arch_t arch;
    uint32_t machine;
    std::vector<uint32_t> original_words;
    std::string_view mnemonic;
  };
  const std::array<SourceCase, 2> cases = {
      SourceCase{.arch = ROCJITSU_CODE_ARCH_CDNA4,
                 .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
                 .original_words = {0xbe8000ffu, 0x12345678u},
                 .mnemonic = "s_mov_b32"},
      SourceCase{.arch = ROCJITSU_CODE_ARCH_RDNA4,
                 .machine = EF_AMDGPU_MACH_AMDGCN_GFX1200,
                 .original_words = {0xee068002u, 0x00800000u, 0},
                 .mnemonic = "global_store_b32"}};
  for (const SourceCase &source_case : cases) {
    SCOPED_TRACE(source_case.arch);
    const util::FailureOr<uint32_t> end = build_s_endpgm(source_case.arch);
    const util::FailureOr<uint32_t> nop = build_s_nop(0, source_case.arch);
    const util::FailureOr<uint32_t> replacement_nop = build_s_nop(1, source_case.arch);
    const util::FailureOr<uint32_t> probe_return = build_s_setpc_b64(30, source_case.arch);
    ASSERT_TRUE(end.succeeded());
    ASSERT_TRUE(nop.succeeded());
    ASSERT_TRUE(replacement_nop.succeeded());
    ASSERT_TRUE(probe_return.succeeded());
    const std::vector<uint8_t> replacement = instruction_bytes({replacement_nop.value()});
    std::vector<uint8_t> original(source_case.original_words.size() * sizeof(uint32_t));
    std::memcpy(original.data(), source_case.original_words.data(), original.size());
    std::unique_ptr<Decoder> decoder = Decoder::create(source_case.arch);
    ASSERT_NE(decoder, nullptr);
    const DecodeResult decoded = decoder->decode_window(source_case.original_words);
    ASSERT_TRUE(decoded.succeeded());
    ASSERT_EQ(decoded.value()->size(), original.size());
    ASSERT_EQ(decoded.value()->mnemonic(), source_case.mnemonic);
    std::vector<uint32_t> words = source_case.original_words;
    words.push_back(end.value());
    const std::vector<uint8_t> image =
        test::make_amdgpu_kernel_elf(words, 0, 3, source_case.machine);
    const std::vector<uint8_t> pristine = image;
    const std::vector<uint8_t> probe_image =
        test::make_amdgpu_probe_elf("entry_probe", {probe_return.value()}, source_case.machine);
    const AmdGpuCodeObject source(image.data(), image.size());
    const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
    ASSERT_TRUE(source.is_valid());
    ASSERT_TRUE(probe.is_valid());
    const InstrumentationPoint entry = {.anchor_offset = 0,
                                        .kind = InstrumentationKind::KernelEntry,
                                        .probe_obj = &probe,
                                        .probe_symbol = "entry_probe",
                                        .probe_args = {}};
    const InstrumentationPoint replace = {
        .anchor_offset = 0,
        .kind = InstrumentationKind::ReplaceWithInst,
        .probe_symbol = {},
        .probe_args = {},
        .replacement =
            InstructionReplacement{.expected_bytes = original, .replacement_bytes = replacement}};
    const int orders = with_entry && with_replacement ? 2 : 1;
    for (int order = 0; order < orders; ++order) {
      SCOPED_TRACE(order);
      Instrumentor instrumentor(source, source_case.arch);
      Instrumentor production(source, source_case.arch);
      std::vector<InstrumentationPoint> points;
      if (with_replacement && order == 1)
        points.push_back(replace);
      if (with_entry)
        points.push_back(entry);
      if (with_replacement && order == 0)
        points.push_back(replace);
      for (const InstrumentationPoint &point : points) {
        instrumentor.add_point(point);
        production.add_point(point);
      }
      const Instrumentor::ValidationResult resolved = instrumentor.validate_points();
      ASSERT_TRUE(resolved.errors.empty()) << resolved.errors.front();
      EXPECT_FALSE(resolved.edit_error);
      ASSERT_EQ(resolved.sites.size(), points.size());
      for (const ResolvedInstrumentationSite &site : resolved.sites) {
        EXPECT_EQ(site.anchor_offset, 0u);
        EXPECT_EQ(site.original_size, original.size());
        EXPECT_EQ(site.original_bytes, original);
        if (site.kind == InstrumentationKind::KernelEntry) {
          EXPECT_EQ(site.mnemonic, source_case.mnemonic);
          EXPECT_TRUE(site.is_probe_call());
        } else {
          EXPECT_EQ(site.kind, InstrumentationKind::ReplaceWithInst);
          EXPECT_FALSE(site.is_probe_call());
          EXPECT_EQ(site.replacement_words, std::vector<uint32_t>{replacement_nop.value()});
        }
      }
      const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
      ASSERT_TRUE(result.errors.empty()) << result.errors.front();
      EXPECT_FALSE(result.edit_error);
      ASSERT_EQ(result.sites.size(), points.size());
      ASSERT_EQ(result.patches.size(), points.size());
      const InstrumentedCodeObject emitted = production.patch();
      ASSERT_TRUE(emitted.errors.empty()) << emitted.errors.front();
      EXPECT_EQ(emitted.elf_bytes, result.elf_bytes);
      ASSERT_EQ(emitted.sites.size(), result.sites.size());
      const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(candidate.is_valid());
      const Section &text = *candidate.text_sections().front();
      const InstrumentedSite *entry_site = nullptr;
      const InstrumentedSite *replacement_site = nullptr;
      for (size_t index = 0; index < result.sites.size(); ++index) {
        const InstrumentedSite &site = result.sites[index];
        const InstrumentationPatch &patch = result.patches[index];
        const bool callback = site.kind == InstrumentationKind::KernelEntry;
        EXPECT_EQ(site.original_offset, 0u);
        EXPECT_EQ(site.original_size, original.size());
        EXPECT_EQ(patch.anchor_offset, 0u);
        EXPECT_EQ(patch.original_size, original.size());
        EXPECT_EQ(patch.original_bytes, original);
        EXPECT_EQ(patch.is_probe_call, callback);
        EXPECT_EQ(emitted.sites[index].kind, site.kind);
        EXPECT_EQ(emitted.sites[index].original_size, original.size());
        EXPECT_EQ(emitted.sites[index].guest_size, callback ? 0u : replacement.size());
        EXPECT_EQ(site.instrumentation_offset, patch.trampoline_offset);
        EXPECT_EQ(site.guest_offset, site.instrumentation_offset + site.instrumentation_size);
        ASSERT_TRUE(patch.return_target);
        EXPECT_EQ(*patch.return_target, callback ? 0u : original.size());
        ASSERT_LE(site.guest_offset + site.guest_size + sizeof(uint32_t), text.size());
        if (callback) {
          entry_site = &site;
          EXPECT_EQ(site.guest_size, 0u);
          EXPECT_GT(site.instrumentation_size, 0u);
          EXPECT_TRUE(patch.patched_anchor_bytes.empty());
        } else {
          replacement_site = &site;
          EXPECT_EQ(site.kind, InstrumentationKind::ReplaceWithInst);
          EXPECT_EQ(site.guest_size, replacement.size());
          EXPECT_EQ(site.instrumentation_size, 0u);
          EXPECT_EQ(patch.patched_anchor_bytes.size(), original.size());
          EXPECT_EQ(
              std::memcmp(text.data() + site.guest_offset, replacement.data(), replacement.size()),
              0);
        }
        const std::optional<uint32_t> returned = test::expected_replacement_branch(
            site.guest_offset + site.guest_size, callback ? 0u : original.size(), source_case.arch);
        ASSERT_TRUE(returned);
        uint32_t actual_return = 0;
        std::memcpy(&actual_return, text.data() + site.guest_offset + site.guest_size,
                    sizeof(actual_return));
        EXPECT_EQ(actual_return, *returned);
      }
      const std::optional<uint64_t> dispatch = test::patched_entry_text_offset(candidate);
      ASSERT_TRUE(dispatch);
      if (with_entry) {
        ASSERT_NE(entry_site, nullptr);
        const std::optional<uint32_t> dispatch_branch = test::expected_replacement_branch(
            *dispatch, entry_site->instrumentation_offset, source_case.arch);
        ASSERT_TRUE(dispatch_branch);
        ASSERT_LE(*dispatch + sizeof(uint32_t), text.size());
        uint32_t actual_dispatch = 0;
        std::memcpy(&actual_dispatch, text.data() + *dispatch, sizeof(actual_dispatch));
        EXPECT_EQ(actual_dispatch, *dispatch_branch);
      } else {
        EXPECT_EQ(*dispatch, 0u);
      }
      if (with_replacement) {
        ASSERT_NE(replacement_site, nullptr);
        const std::optional<uint32_t> splice =
            test::expected_replacement_branch(0, replacement_site->guest_offset, source_case.arch);
        ASSERT_TRUE(splice);
        uint32_t actual_splice = 0;
        std::memcpy(&actual_splice, text.data(), sizeof(actual_splice));
        EXPECT_EQ(actual_splice, *splice);
        for (size_t offset = sizeof(uint32_t); offset < original.size();
             offset += sizeof(uint32_t)) {
          uint32_t padding = 0;
          std::memcpy(&padding, text.data() + offset, sizeof(padding));
          EXPECT_EQ(padding, nop.value());
        }
        if (with_entry) {
          EXPECT_TRUE(
              entry_site->guest_offset + sizeof(uint32_t) <= replacement_site->guest_offset ||
              replacement_site->guest_offset + replacement_site->guest_size + sizeof(uint32_t) <=
                  entry_site->instrumentation_offset);
        } else {
          expect_pure_replacements(
              source, result, {{.offset = 0, .expected = original, .replacement = replacement}},
              source_case.arch);
        }
      } else {
        EXPECT_EQ(std::memcmp(text.data(), original.data(), original.size()), 0);
      }
      EXPECT_EQ(image, pristine);
      EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
    }
  }
}

TEST(InstrumentorReplacement, OwnsQueuedPayloadAndReportsTrampolineRanges) {
  const uint32_t original = kNop1Word;
  const uint32_t replacement = kNop2Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({original, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
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
  expect_pure_replacements(source, result,
                           {{.offset = 0,
                             .expected = instruction_bytes({original}),
                             .replacement = instruction_bytes({replacement})}});
  expect_replacement_guest(result, 0, 4, instruction_bytes({replacement}), kArch, false);
  EXPECT_FALSE(result.edit_error);
  EXPECT_EQ(image, pristine);
  expect_no_output(instrumentor.patch());
}

TEST(InstrumentorReplacement, ForkSharesFactsButNotRequestsOrPatchBudget) {
  const uint32_t nop = kNop0Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, nop, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  std::unique_ptr<Instrumentor> first;
  std::unique_ptr<Instrumentor> second;
  const InstructionBoundaries *shared_facts = nullptr;
  TextEditError error;
  {
    Instrumentor parent(source, kArch);
    shared_facts = parent.instruction_boundaries(error);
    ASSERT_NE(shared_facts, nullptr) << error.message;
    parent.add_point(replacement_point(0, {nop}, {kNop1Word}));
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
  first->add_point(replacement_point(0, {nop}, {kNop2Word}));
  second->add_point(replacement_point(4, {nop}, {kNop3Word}));
  const InstrumentedCodeObject first_result = first->patch();
  const InstrumentedCodeObject second_result = second->patch();
  ASSERT_TRUE(first_result.errors.empty()) << first_result.errors.front();
  ASSERT_TRUE(second_result.errors.empty()) << second_result.errors.front();
  ASSERT_EQ(first_result.sites.size(), 1u);
  ASSERT_EQ(second_result.sites.size(), 1u);
  expect_pure_replacements(source, first_result,
                           {{.offset = 0,
                             .expected = instruction_bytes({nop}),
                             .replacement = instruction_bytes({kNop2Word})}});
  expect_pure_replacements(source, second_result,
                           {{.offset = 4,
                             .expected = instruction_bytes({nop}),
                             .replacement = instruction_bytes({kNop3Word})}});
  EXPECT_EQ(image, pristine);

  const AmdGpuCodeObject mutant(first_result.elf_bytes.data(), first_result.elf_bytes.size());
  Instrumentor fresh_mutant(mutant, kArch);
  const InstructionBoundaries *mutant_facts = fresh_mutant.instruction_boundaries(error);
  ASSERT_NE(mutant_facts, nullptr) << error.message;
  EXPECT_NE(mutant_facts, shared_facts);
  fresh_mutant.add_point(replacement_point(0, {nop}, {kNop4Word}));
  const InstrumentedCodeObject stale = fresh_mutant.patch();
  expect_no_output(stale);
  ASSERT_TRUE(stale.edit_error);
  EXPECT_EQ(stale.edit_error->code, TextEditErrorCode::StaleBytes);

  constexpr rj_code_arch_t other_arch = ROCJITSU_CODE_ARCH_RDNA4;
  const util::FailureOr<uint32_t> other_nop = build_s_nop(0, other_arch);
  ASSERT_TRUE(other_nop.succeeded());
  const util::FailureOr<uint32_t> other_endpgm = build_s_endpgm(other_arch);
  ASSERT_TRUE(other_endpgm.succeeded());
  const std::vector<uint8_t> other_image =
      test::make_gfx1200_kernel_elf({other_nop.value(), other_endpgm.value()}, 0);
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
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  const util::FailureOr<uint32_t> forward_branch = build_s_branch(2, arch);
  ASSERT_TRUE(forward_branch.succeeded());
  const util::FailureOr<uint32_t> backward_branch = build_s_branch(-3, arch);
  ASSERT_TRUE(backward_branch.succeeded());
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  const util::FailureOr<uint32_t> clause_replacement = build_s_nop(1, arch);
  ASSERT_TRUE(clause_replacement.succeeded());
  // Dispatch skips the clause at 8 and its member at 12. The unreachable branch
  // at 20 still references that member, splitting the full-text CFG's clause run.
  const std::vector<uint8_t> image = test::make_gfx1200_kernel_elf(
      {nop, forward_branch.value(), 0xbf850000u, nop, endpgm.value(), backward_branch.value()}, 0);
  const std::vector<uint8_t> pristine = image;
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
  const std::span<const uint64_t> references = shared_facts->reference_addresses();
  const uint64_t text_address = source.text_sections().front()->vaddr();
  EXPECT_NE(std::ranges::find(references, text_address + 12), references.end());
  EXPECT_NE(std::ranges::find(references, text_address + 16), references.end());

  blocked_child->add_point(replacement_point(12, {nop}, {clause_replacement.value()}));
  const InstrumentedCodeObject rejected = blocked_child->patch();
  expect_no_output(rejected);
  ASSERT_TRUE(rejected.edit_error);
  EXPECT_EQ(rejected.edit_error->code, TextEditErrorCode::Clause);
  EXPECT_EQ(rejected.edit_error->offset, 12u);

  const util::FailureOr<uint32_t> maybe_replacement = build_s_nop(2, arch);
  ASSERT_TRUE(maybe_replacement.succeeded());
  const uint32_t replacement = maybe_replacement.value();
  child->add_point(replacement_point(0, {nop}, {replacement}));
  const InstrumentedCodeObject result = child->patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 1u);
  expect_pure_replacements(source, result,
                           {{.offset = 0,
                             .expected = instruction_bytes({nop}),
                             .replacement = instruction_bytes({replacement})}},
                           arch);
  EXPECT_EQ(image, pristine);
}

TEST(InstrumentorReplacement, DoesNotRequireRelocatableAnchorsOrReplayGuestInstructions) {
  const uint32_t getpc = kGetpcWord;
  const uint32_t branch = kBranch1Word;
  const uint32_t nop = kNop0Word;
  const uint32_t end = kEndpgmWord;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({getpc, branch, nop, end}, 0);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor observational(source, kArch);
  observational.add_point_by_offset(0);
  expect_no_output(observational.patch());
  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point(replacement_point(0, {getpc}, {nop}));
  instrumentor.add_point(replacement_point(4, {branch}, {nop}));
  // A final source terminator has no continuation. Its replacement must also
  // terminate, rather than accidentally returning into the appended cave.
  instrumentor.add_point(replacement_point(12, {end}, {end}));
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  expect_pure_replacements(source, result,
                           {{.offset = 0,
                             .expected = instruction_bytes({getpc}),
                             .replacement = instruction_bytes({nop})},
                            {.offset = 4,
                             .expected = instruction_bytes({branch}),
                             .replacement = instruction_bytes({nop})},
                            {.offset = 12,
                             .expected = instruction_bytes({end}),
                             .replacement = instruction_bytes({end})}});
  ASSERT_EQ(result.patches.size(), 3u);
  EXPECT_FALSE(result.patches.back().return_target);
  EXPECT_EQ(image, pristine);
}

TEST(InstrumentorReplacement, DoesNotApplyObservationalBranchStackGate) {
  const uint32_t nop = kNop0Word;
  const uint32_t changed = kNop1Word;
  // s_cbranch_join elsewhere in .text makes observational liveness unsupported.
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({nop, 0xbe802e22u, kEndpgmWord}, 0);
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
  expect_pure_replacements(source, result,
                           {{.offset = 0,
                             .expected = instruction_bytes({nop}),
                             .replacement = instruction_bytes({changed})}});
}

TEST(InstrumentorReplacement, DoesNotRequireProbeRegisterOrScratchResources) {
  const uint32_t nop = kNop0Word;
  // Eight declared SGPRs cannot supply a callable probe's s[30:31] link pair.
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, kEndpgmWord}, 0, 0);
  const std::vector<uint8_t> probe_image = test::make_gfx950_probe_elf("probe", {kSetpc30Word});
  const AmdGpuCodeObject source(image.data(), image.size());
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  Instrumentor observational(source, kArch);
  InstrumentationPoint probe_point = {
      .probe_obj = &probe, .probe_symbol = "probe", .probe_args = {}};
  observational.add_point(probe_point);
  expect_no_output(observational.patch());

  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point(replacement_point(0, {nop}, {kNop1Word}));
  const InstrumentedCodeObject result = instrumentor.patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  expect_pure_replacements(source, result,
                           {{.offset = 0,
                             .expected = instruction_bytes({nop}),
                             .replacement = instruction_bytes({kNop1Word})}});
  EXPECT_EQ(result.resources.sgprs, 0u);
  EXPECT_EQ(result.resources.vgprs, 0u);
  EXPECT_EQ(result.resources.kernarg_bytes, 0u);
}

TEST(InstrumentorReplacement, LateStaleBytesFailValidationAndEmissionWithoutPartialSites) {
  const uint32_t nop = kNop0Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, nop, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point(replacement_point(0, {nop}, {kNop1Word}));
  instrumentor.add_point(replacement_point(4, {kNop2Word}, {nop}));
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
  EXPECT_EQ(image, pristine);
}

TEST(InstrumentorReplacement, RejectsPhysicalOverlapBeforePublishingAnySite) {
  const uint32_t nop = kNop0Word;
  const std::vector<uint8_t> image =
      test::make_gfx950_kernel_elf({0xbe8000ffu, 0x12345678u, kEndpgmWord}, 0);
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

TEST(InstrumentorReplacement, ComposesDistinctReplacementAndObservationInEitherQueueOrder) {
  const uint32_t nop = kNop0Word;
  const uint32_t changed = kNop1Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, nop, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
  const std::vector<uint8_t> probe_image = test::make_gfx950_probe_elf("probe", {kSetpc30Word});
  const AmdGpuCodeObject source(image.data(), image.size());
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  for (const bool callable : {false, true}) {
    for (const bool replacement_first : {false, true}) {
      SCOPED_TRACE(callable);
      SCOPED_TRACE(replacement_first);
      Instrumentor instrumentor(source, kArch);
      const InstrumentationPoint observation = {.anchor_offset = 4,
                                                .probe_obj = callable ? &probe : nullptr,
                                                .probe_symbol = callable ? "probe" : "",
                                                .probe_args = {}};
      const InstrumentationPoint replace = replacement_point(0, {nop}, {changed});
      instrumentor.add_point(replacement_first ? replace : observation);
      instrumentor.add_point(replacement_first ? observation : replace);
      const Instrumentor::ValidationResult resolved = instrumentor.validate_points();
      ASSERT_TRUE(resolved.errors.empty()) << resolved.errors.front();
      ASSERT_EQ(resolved.sites.size(), 2u);
      const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
      ASSERT_TRUE(result.errors.empty()) << result.errors.front();
      ASSERT_EQ(result.sites.size(), 2u);
      ASSERT_EQ(result.patches.size(), 2u);
      const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(candidate.is_valid());
      const Section &text = *candidate.text_sections().front();
      for (size_t index = 0; index < result.sites.size(); ++index) {
        const InstrumentedSite &site = result.sites[index];
        const bool replacement = site.kind == InstrumentationKind::ReplaceWithInst;
        EXPECT_EQ(site.original_offset, replacement ? 0u : 4u);
        EXPECT_EQ(site.original_size, 4u);
        EXPECT_EQ(site.guest_size, 4u);
        ASSERT_LE(site.guest_offset + 8, text.size());
        uint32_t guest = 0;
        std::memcpy(&guest, text.data() + site.guest_offset, sizeof(guest));
        EXPECT_EQ(guest, replacement ? changed : nop);
        const std::optional<uint32_t> branch = test::expected_replacement_branch(
            site.original_offset, site.instrumentation_offset, kArch);
        ASSERT_TRUE(branch);
        uint32_t actual = 0;
        std::memcpy(&actual, text.data() + site.original_offset, sizeof(actual));
        EXPECT_EQ(actual, *branch);
        const std::optional<uint32_t> returned = test::expected_replacement_branch(
            site.guest_offset + site.guest_size, site.original_offset + 4, kArch);
        ASSERT_TRUE(returned);
        std::memcpy(&actual, text.data() + site.guest_offset + site.guest_size, sizeof(actual));
        EXPECT_EQ(actual, *returned);
        EXPECT_EQ(result.patches[index].is_probe_call, !replacement && callable);
      }
      EXPECT_FALSE(result.edit_error);
      EXPECT_EQ(image, pristine);
    }
  }
}

TEST(InstrumentorReplacement, ComposesProbeAndReplacementAtTheSameSite) {
  const uint32_t original = test::kMovS8Zero;
  const uint32_t replacement = kNop1Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({original, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
  const std::vector<uint8_t> probe_image = test::make_gfx950_probe_elf("probe", {kSetpc30Word});
  const AmdGpuCodeObject source(image.data(), image.size());
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  for (const bool with_arguments : {false, true}) {
    SCOPED_TRACE(with_arguments);
    const std::vector<ProbeArgValue> arguments = with_arguments
                                                     ? std::vector<ProbeArgValue>{probe_arg_imm(42)}
                                                     : std::vector<ProbeArgValue>{};
    std::vector<uint8_t> direct_image;
    for (int mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(mode);
      Instrumentor instrumentor(source, kArch);
      InstrumentationPoint replace = replacement_point(0, {original}, {replacement});
      const InstrumentationPoint before = {.anchor_offset = 0,
                                           .probe_obj = &probe,
                                           .probe_symbol = "probe",
                                           .probe_args = arguments,
                                           .force_full_exec = with_arguments};
      if (mode == 0) {
        replace.probe_obj = &probe;
        replace.probe_symbol = "probe";
        replace.probe_args = arguments;
        replace.force_full_exec = with_arguments;
        instrumentor.add_point(replace);
      } else {
        instrumentor.add_point(mode == 1 ? replace : before);
        instrumentor.add_point(mode == 1 ? before : replace);
      }
      const Instrumentor::ValidationResult resolved = instrumentor.validate_points();
      ASSERT_TRUE(resolved.errors.empty()) << resolved.errors.front();
      ASSERT_EQ(resolved.sites.size(), 1u);
      EXPECT_TRUE(resolved.sites.front().is_probe_call());
      EXPECT_EQ(resolved.sites.front().probe_args, arguments);
      const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
      ASSERT_TRUE(result.errors.empty()) << result.errors.front();
      expect_replacement_guest(result, 0, 4, instruction_bytes({replacement}), kArch, true);
      if (mode == 0)
        direct_image = result.elf_bytes;
      else
        // Equal bytes also check that pair coalescing preserves the full-EXEC
        // policy, not just the argument vector exposed by validate_points().
        EXPECT_EQ(result.elf_bytes, direct_image);
      EXPECT_FALSE(result.edit_error);
      EXPECT_EQ(image, pristine);
    }
  }
}

TEST(InstrumentorReplacement, KernelEntryOnlyReportsPristineWideSource) {
  ASSERT_NO_FATAL_FAILURE(check_wide_entry_source_attribution(true, false));
}

TEST(InstrumentorReplacement, ReplacementOnlyReportsPristineWideSource) {
  ASSERT_NO_FATAL_FAILURE(check_wide_entry_source_attribution(false, true));
}

TEST(InstrumentorReplacement, KernelEntryWithReplacementReportsPristineWideSourceInBothOrders) {
  ASSERT_NO_FATAL_FAILURE(check_wide_entry_source_attribution(true, true));
}

TEST(InstrumentorReplacement, SameSiteConflictsAndInvalidBodiesPublishNothing) {
  const uint32_t nop = kNop0Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
  const std::vector<uint8_t> probe_image = test::make_gfx950_probe_elf("probe", {kSetpc30Word});
  const AmdGpuCodeObject source(image.data(), image.size());
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  for (int mode = 0; mode < 5; ++mode) {
    SCOPED_TRACE(mode);
    Instrumentor instrumentor(source, kArch);
    InstrumentationPoint replace = replacement_point(0, {nop}, {kNop1Word});
    const InstrumentationPoint before = {
        .anchor_offset = 0, .probe_obj = &probe, .probe_symbol = "probe", .probe_args = {}};
    switch (mode) {
    case 0:
      instrumentor.add_point(replace);
      instrumentor.add_point(replace);
      break;
    case 1:
      instrumentor.add_point(replace);
      instrumentor.add_point(before);
      instrumentor.add_point(before);
      break;
    case 2:
      replace.probe_obj = &probe;
      replace.probe_symbol = "probe";
      instrumentor.add_point(replace);
      instrumentor.add_point(before);
      break;
    case 3:
      instrumentor.add_point(replace);
      instrumentor.add_point_by_offset(0);
      break;
    case 4:
      replace.probe_obj = &probe;
      replace.probe_symbol = "probe";
      replace.replacement->replacement_bytes.assign(4, 0xff);
      instrumentor.add_point(replace);
      break;
    }
    const Instrumentor::ValidationResult resolved = instrumentor.validate_points();
    EXPECT_FALSE(resolved.errors.empty());
    EXPECT_TRUE(resolved.sites.empty());
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    expect_no_output(result);
    EXPECT_TRUE(result.patches.empty());
    if (mode == 4) {
      ASSERT_TRUE(result.edit_error);
      EXPECT_EQ(result.edit_error->code, TextEditErrorCode::InvalidInstruction);
    }
    EXPECT_EQ(image, pristine);
  }
}

TEST(InstrumentorReplacement, EmptyAndVariableLengthBodiesUseTrampolines) {
  const uint32_t original = kNop1Word;
  const uint32_t nop = kNop0Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({original, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  for (size_t words = 0; words < 4; ++words) {
    SCOPED_TRACE(words);
    InstrumentationPoint point = replacement_point(0, {original}, {});
    point.replacement->replacement_bytes.resize(words * 4);
    for (size_t index = 0; index < words; ++index)
      std::memcpy(point.replacement->replacement_bytes.data() + index * 4, &nop, sizeof(nop));
    Instrumentor instrumentor(source, kArch);
    instrumentor.add_point(point);
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    expect_pure_replacements(source, result,
                             {{.offset = 0,
                               .expected = instruction_bytes({original}),
                               .replacement = point.replacement->replacement_bytes}});
    expect_replacement_guest(result, 0, 4, point.replacement->replacement_bytes, kArch, false);
    EXPECT_EQ(image, pristine);
  }
}

TEST(InstrumentorReplacement, RejectsMalformedReplacementAndProbeRequests) {
  const uint32_t nop = kNop0Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
  const std::vector<uint8_t> probe_image = test::make_gfx950_probe_elf("probe", {kSetpc30Word});
  const AmdGpuCodeObject source(image.data(), image.size());
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  for (int mode = 0; mode < 7; ++mode) {
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
    case 6:
      point.probe_obj = &probe;
      break;
    }
    Instrumentor instrumentor(source, kArch);
    instrumentor.add_point(point);
    const Instrumentor::ValidationResult resolved = instrumentor.validate_points();
    EXPECT_FALSE(resolved.errors.empty());
    EXPECT_TRUE(resolved.sites.empty());
    EXPECT_FALSE(resolved.edit_error);
    const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
    expect_no_output(result);
    EXPECT_TRUE(result.patches.empty());
    EXPECT_FALSE(result.edit_error);
    EXPECT_EQ(image, pristine);
  }
}

TEST(InstrumentorReplacement, CannotReuseExpectedBytesFromAnotherSource) {
  const uint32_t original = kNop1Word;
  const uint32_t different = kNop2Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({different, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor instrumentor(source, kArch);
  instrumentor.add_point(replacement_point(0, {original}, {kNop0Word}));
  const InstrumentedCodeObject result = instrumentor.patch();
  expect_no_output(result);
  ASSERT_TRUE(result.edit_error);
  EXPECT_EQ(result.edit_error->code, TextEditErrorCode::StaleBytes);
  EXPECT_EQ(result.edit_error->offset, 0u);
  EXPECT_EQ(image, pristine);
}

TEST(InstrumentorReplacement, ForkRetainsSharedResourceOptions) {
  const uint32_t nop = kNop0Word;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({nop, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
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
    batch.value()->add_point(replacement_point(0, {nop}, {kNop1Word}));
    const Instrumentor::ValidationResult resolved = batch.value()->validate_points();
    const InstrumentedCodeObjectDebug result = batch.value()->patch_with_debug_summaries();
    if (mode == 0) {
      // Binding permission alone requests no entry-storage wrapper. A pure
      // replacement must preserve the ordinary guest kernarg ABI and metadata.
      ASSERT_TRUE(resolved.errors.empty()) << resolved.errors.front();
      expect_pure_replacements(source, result,
                               {{.offset = 0,
                                 .expected = instruction_bytes({nop}),
                                 .replacement = instruction_bytes({kNop1Word})}});
      ASSERT_EQ(result.patches.size(), 1u);
    } else {
      expect_no_output(result);
      EXPECT_TRUE(result.patches.empty());
      const std::string_view expected =
          mode == 1 ? "declared-register preservation requires original metadata counts"
                    : "metadata";
      EXPECT_TRUE(std::ranges::any_of(result.errors, [expected](const std::string &message) {
        return message.find(expected) != std::string::npos;
      }));
    }
    EXPECT_FALSE(resolved.edit_error);
    EXPECT_FALSE(result.edit_error);
    EXPECT_EQ(image, pristine);
  }
}

TEST(InstrumentorReplacement, MutantReceivesFreshCallableProbeAnalysis) {
  const uint32_t nop = kNop0Word;
  const uint32_t original = test::kMovS8Zero;
  const std::vector<uint8_t> image = test::make_gfx950_kernel_elf({original, nop, kEndpgmWord}, 0);
  const std::vector<uint8_t> pristine = image;
  std::vector<uint8_t> mutant_image;
  uint64_t mutant_guest = 0;
  {
    const AmdGpuCodeObject source(image.data(), image.size());
    Instrumentor replacement(source, kArch);
    replacement.add_point(replacement_point(0, {original}, {nop}));
    InstrumentedCodeObject result = replacement.patch();
    expect_pure_replacements(source, result,
                             {{.offset = 0,
                               .expected = instruction_bytes({original}),
                               .replacement = instruction_bytes({nop})}});
    ASSERT_EQ(result.sites.size(), 1u);
    mutant_guest = result.sites.front().guest_offset;
    mutant_image = std::move(result.elf_bytes);
  }
  // The pristine object and Instrumentor are gone. Observe the effective guest
  // in the emitted cave; the source site now contains a branch, not that guest.
  const std::vector<uint8_t> pristine_mutant = mutant_image;
  const AmdGpuCodeObject mutant(mutant_image.data(), mutant_image.size());
  const std::vector<uint8_t> probe_image = test::make_gfx950_probe_elf("probe", {kSetpc30Word});
  const AmdGpuCodeObject probe(probe_image.data(), probe_image.size());
  Instrumentor instrumentor(mutant, kArch);
  InstrumentationPoint point = {.anchor_offset = mutant_guest,
                                .probe_obj = &probe,
                                .probe_symbol = "probe",
                                .probe_args = {}};
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
  EXPECT_EQ(mutant_image, pristine_mutant);
  EXPECT_EQ(image, pristine);
}

TEST(InstrumentorReplacement, ConcreteVariantControlsSourceAndReplacementDecoding) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  for (const uint32_t machine : {EF_AMDGPU_MACH_AMDGCN_GFX1250, EF_AMDGPU_MACH_AMDGCN_GFX1251}) {
    for (const bool variant_in_source : {false, true}) {
      SCOPED_TRACE(machine);
      SCOPED_TRACE(variant_in_source);
      // gfx1251-only VOP3P, also used by the callable-probe concrete-decoder tests.
      const std::vector<uint32_t> words =
          variant_in_source ? std::vector<uint32_t>{0xcc4c4004u, 0x1a021908u, endpgm.value()}
                            : std::vector<uint32_t>{0xbe8000ffu, 0x12345678u, endpgm.value()};
      const std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, machine);
      const AmdGpuCodeObject source(image.data(), image.size());
      Instrumentor instrumentor(source, arch);
      const InstrumentationPoint point =
          variant_in_source
              ? replacement_point(0, {0xcc4c4004u, 0x1a021908u}, {nop, nop})
              : replacement_point(0, {0xbe8000ffu, 0x12345678u}, {0xcc4c4004u, 0x1a021908u});
      instrumentor.add_point(point);
      const InstrumentedCodeObject result = instrumentor.patch();
      if (machine == EF_AMDGPU_MACH_AMDGCN_GFX1250) {
        expect_no_output(result);
        ASSERT_TRUE(result.edit_error);
        EXPECT_EQ(result.edit_error->code, TextEditErrorCode::InvalidInstruction);
      } else {
        ASSERT_TRUE(result.errors.empty()) << result.errors.front();
        const AmdGpuCodeObject candidate(result.elf_bytes.data(), result.elf_bytes.size());
        EXPECT_EQ(candidate.target_id(), ROCJITSU_CODE_TARGET_GFX1251);
        expect_pure_replacements(source, result,
                                 {{.offset = 0,
                                   .expected = point.replacement->expected_bytes,
                                   .replacement = point.replacement->replacement_bytes}},
                                 arch);
      }
    }
  }
}

TEST(InstrumentorReplacement, RejectsArchitectureOverrideForConcreteAndLegacySources) {
  for (const uint32_t machine : {EF_AMDGPU_MACH_AMDGCN_GFX908, EF_AMDGPU_MACH_AMDGCN_GFX950}) {
    const rj_code_arch_t source_arch = arch_for_elf_mach(machine);
    const rj_code_arch_t wrong_arch = source_arch == kArch ? ROCJITSU_CODE_ARCH_CDNA1 : kArch;
    const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, source_arch);
    ASSERT_TRUE(maybe_nop.succeeded());
    const uint32_t nop = maybe_nop.value();
    const util::FailureOr<uint32_t> endpgm = build_s_endpgm(source_arch);
    ASSERT_TRUE(endpgm.succeeded());
    const std::vector<uint8_t> image =
        test::make_amdgpu_kernel_elf({nop, endpgm.value()}, 0, 3, machine);
    const AmdGpuCodeObject source(image.data(), image.size());
    Instrumentor instrumentor(source, wrong_arch);
    instrumentor.add_point(replacement_point(0, {nop}, {nop}));
    expect_no_output(instrumentor.patch());
  }
}

TEST(InstrumentorReplacement, CallerArchitecturePolicyAgreesAcrossQueryForkValidationAndPatch) {
  struct ArchitectureCase {
    uint32_t machine;
    rj_code_arch_t architecture;
    bool supported;
  };
  constexpr std::array<ArchitectureCase, 3> kCases = {
      ArchitectureCase{.machine = 0, .architecture = ROCJITSU_CODE_ARCH_RDNA4, .supported = true},
      ArchitectureCase{.machine = EF_AMDGPU_MACH_AMDGCN_GFX1010,
                       .architecture = ROCJITSU_CODE_ARCH_RDNA1,
                       .supported = true},
      ArchitectureCase{.machine = EF_AMDGPU_MACH_AMDGCN_GFX1200,
                       .architecture = ROCJITSU_CODE_ARCH_RDNA1,
                       .supported = false}};
  for (const ArchitectureCase &test_case : kCases) {
    SCOPED_TRACE(test_case.machine);
    const rj_code_arch_t source_architecture =
        test_case.machine == 0 ? test_case.architecture : arch_for_elf_mach(test_case.machine);
    const util::FailureOr<uint32_t> maybe_original = build_s_nop(0, source_architecture);
    ASSERT_TRUE(maybe_original.succeeded());
    const uint32_t original = maybe_original.value();
    const util::FailureOr<uint32_t> maybe_replacement = build_s_nop(1, source_architecture);
    ASSERT_TRUE(maybe_replacement.succeeded());
    const uint32_t replacement = maybe_replacement.value();
    const util::FailureOr<uint32_t> endpgm = build_s_endpgm(source_architecture);
    ASSERT_TRUE(endpgm.succeeded());
    const std::vector<uint8_t> image =
        test::make_amdgpu_kernel_elf({original, endpgm.value()}, 0, 3, test_case.machine);
    const std::vector<uint8_t> pristine = image;
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    Instrumentor seed(source, test_case.architecture);
    TextEditError error;
    const InstructionBoundaries *facts = seed.instruction_boundaries(error);
    const TextEditError query_error = error;
    FailureOr<std::unique_ptr<Instrumentor>> fork = seed.fork(error);
    if (test_case.supported) {
      ASSERT_NE(facts, nullptr) << error.message;
      ASSERT_TRUE(fork.succeeded()) << error.message;
      EXPECT_EQ(fork.value()->instruction_boundaries(error), facts);
      const std::array<Instrumentor *, 2> batches = {&seed, fork.value().get()};
      for (Instrumentor *batch : batches) {
        batch->add_point(replacement_point(0, {original}, {replacement}));
        const Instrumentor::ValidationResult validated = batch->validate_points();
        ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
        EXPECT_FALSE(validated.edit_error);
        ASSERT_EQ(validated.sites.size(), 1u);
        const InstrumentedCodeObject result = batch->patch();
        ASSERT_TRUE(result.errors.empty()) << result.errors.front();
        EXPECT_FALSE(result.edit_error);
        ASSERT_EQ(result.sites.size(), 1u);
        expect_pure_replacements(source, result,
                                 {{.offset = 0,
                                   .expected = instruction_bytes({original}),
                                   .replacement = instruction_bytes({replacement})}},
                                 test_case.architecture);
      }
    } else {
      EXPECT_EQ(facts, nullptr);
      EXPECT_EQ(query_error.code, TextEditErrorCode::UnsupportedTarget);
      EXPECT_TRUE(fork.failed());
      EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedTarget);
      seed.add_point(replacement_point(0, {original}, {replacement}));
      const Instrumentor::ValidationResult validated = seed.validate_points();
      EXPECT_TRUE(validated.sites.empty());
      EXPECT_FALSE(validated.errors.empty());
      ASSERT_TRUE(validated.edit_error);
      EXPECT_EQ(validated.edit_error->code, TextEditErrorCode::UnsupportedTarget);
      const InstrumentedCodeObject result = seed.patch();
      expect_no_output(result);
      ASSERT_TRUE(result.edit_error);
      EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedTarget);
    }
    EXPECT_EQ(image, pristine);
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  }
}

TEST(InstrumentorReplacement, FullTextFactsRetainClauseMembershipAcrossUnreachableGap) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  const util::FailureOr<uint32_t> branch = build_s_branch(2, arch);
  ASSERT_TRUE(branch.succeeded());
  const util::FailureOr<uint32_t> nop1 = build_s_nop(1, arch);
  ASSERT_TRUE(nop1.succeeded());
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  const std::vector<uint8_t> image =
      test::make_gfx1200_kernel_elf({branch.value(), 0xbf850001u, nop, nop, endpgm.value()}, 0);
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor instrumentor(source, arch);
  TextEditError error;
  const InstructionBoundaries *facts = instrumentor.instruction_boundaries(error);
  ASSERT_NE(facts, nullptr) << error.message;
  EXPECT_EQ(instrumentor.instruction_boundaries(error), facts);
  const InstructionBoundaries::Entry *entry = facts->find(12);
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->clause_member);
  instrumentor.add_point(replacement_point(12, {nop}, {nop1.value()}));
  const InstrumentedCodeObject result = instrumentor.patch();
  expect_no_output(result);
  ASSERT_TRUE(result.edit_error);
  EXPECT_EQ(result.edit_error->code, TextEditErrorCode::Clause);
  EXPECT_EQ(result.edit_error->offset, 12u);
}

TEST(InstrumentorReplacement, ForkBatchesCheckDescriptorEntriesBeforePublishingAnySite) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint64_t kEntry = 256;
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  const util::FailureOr<uint32_t> nop1 = build_s_nop(1, arch);
  ASSERT_TRUE(nop1.succeeded());
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  for (const uint64_t anchor : {248u, 252u, 256u}) {
    SCOPED_TRACE(anchor);
    std::vector<uint32_t> words(67, nop);
    words[anchor / sizeof(uint32_t)] = 0xbe8000ffu;
    words[anchor / sizeof(uint32_t) + 1] = 0x12345678u;
    words.back() = endpgm.value();
    const std::vector<uint8_t> image =
        test::make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1200, 0, 0, false, false,
                                     false, false, false, false, 0, kEntry);
    const std::vector<uint8_t> pristine = image;
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    const Section &text = *source.text_sections().front();
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(image, text.sectionOffset(), text.size());
    ASSERT_EQ(kernels.size(), 1u);
    ASSERT_EQ(kernels.front().entry_text_offset, kEntry);
    Instrumentor seed(source, arch);
    TextEditError error;
    const InstructionBoundaries *facts = seed.instruction_boundaries(error);
    ASSERT_NE(facts, nullptr) << error.message;
    for (int attempt = 0; attempt < 2; ++attempt) {
      SCOPED_TRACE(attempt);
      FailureOr<std::unique_ptr<Instrumentor>> fork = seed.fork(error);
      ASSERT_TRUE(fork.succeeded()) << error.message;
      EXPECT_EQ(fork.value()->instruction_boundaries(error), facts);
      // The later eight-byte edit surrounds the descriptor entry only at 252.
      // Neither validation nor emission may publish the earlier valid edit.
      fork.value()->add_point(replacement_point(0, {nop}, {nop1.value()}));
      fork.value()->add_point(replacement_point(anchor, {0xbe8000ffu, 0x12345678u}, {nop, nop}));
      const Instrumentor::ValidationResult validated = fork.value()->validate_points();
      const InstrumentedCodeObjectDebug result = fork.value()->patch_with_debug_summaries();
      if (anchor == 252) {
        EXPECT_TRUE(validated.sites.empty());
        EXPECT_FALSE(validated.errors.empty());
        ASSERT_TRUE(validated.edit_error);
        EXPECT_EQ(validated.edit_error->code, TextEditErrorCode::UnsupportedReferences);
        EXPECT_EQ(validated.edit_error->offset, kEntry);
        expect_no_output(result);
        EXPECT_TRUE(result.patches.empty());
        ASSERT_TRUE(result.edit_error);
        EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
        EXPECT_EQ(result.edit_error->offset, kEntry);
      } else {
        ASSERT_TRUE(validated.errors.empty()) << validated.errors.front();
        ASSERT_EQ(validated.sites.size(), 2u);
        ASSERT_TRUE(result.errors.empty()) << result.errors.front();
        EXPECT_FALSE(result.edit_error);
        ASSERT_EQ(result.sites.size(), 2u);
        ASSERT_EQ(result.patches.size(), 2u);
        expect_pure_replacements(source, result,
                                 {{.offset = 0,
                                   .expected = instruction_bytes({nop}),
                                   .replacement = instruction_bytes({nop1.value()})},
                                  {.offset = anchor,
                                   .expected = instruction_bytes({0xbe8000ffu, 0x12345678u}),
                                   .replacement = instruction_bytes({nop, nop})}},
                                 arch);
      }
      EXPECT_EQ(image, pristine);
      EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
    }
  }
}

TEST(InstrumentorReplacement, RejectsSourceDebugBranchesWithoutDestinationMetadata) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA3;
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  const util::FailureOr<uint32_t> control_branch = build_s_branch(0, arch);
  ASSERT_TRUE(control_branch.succeeded());
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  // These gfx942 debug branches target the literal inside the edit at [4,12).
  // Their absent branch metadata must not hide that uncollected reference.
  constexpr std::array<uint32_t, 4> kDebugBranches = {0xbf970001u, 0xbf980001u, 0xbf990001u,
                                                      0xbf9a0001u};
  for (const uint32_t branch : kDebugBranches) {
    SCOPED_TRACE(branch);
    const std::vector<uint8_t> image =
        test::make_gfx942_kernel_elf({branch, 0xbe8000ffu, 0x12345678u, endpgm.value()}, 0);
    const std::vector<uint8_t> pristine = image;
    const AmdGpuCodeObject source(image.data(), image.size());
    Instrumentor instrumentor(source, arch);
    instrumentor.add_point(replacement_point(4, {0xbe8000ffu, 0x12345678u}, {nop, nop}));
    const Instrumentor::ValidationResult validated = instrumentor.validate_points();
    EXPECT_TRUE(validated.sites.empty());
    EXPECT_FALSE(validated.errors.empty());
    ASSERT_TRUE(validated.edit_error);
    EXPECT_EQ(validated.edit_error->code, TextEditErrorCode::UnsupportedReferences);
    EXPECT_EQ(validated.edit_error->offset, 0u);
    const InstrumentedCodeObject result = instrumentor.patch();
    expect_no_output(result);
    ASSERT_TRUE(result.edit_error);
    EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
    EXPECT_EQ(result.edit_error->offset, 0u);
    EXPECT_EQ(image, pristine);
    EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
  }
  const std::vector<uint8_t> image = test::make_gfx942_kernel_elf(
      {control_branch.value(), 0xbe8000ffu, 0x12345678u, endpgm.value()}, 0);
  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  Instrumentor control(source, arch);
  control.add_point(replacement_point(4, {0xbe8000ffu, 0x12345678u}, {nop, nop}));
  const InstrumentedCodeObject result = control.patch();
  ASSERT_TRUE(result.errors.empty()) << result.errors.front();
  ASSERT_EQ(result.sites.size(), 1u);
  expect_pure_replacements(source, result,
                           {{.offset = 4,
                             .expected = instruction_bytes({0xbe8000ffu, 0x12345678u}),
                             .replacement = instruction_bytes({nop, nop})}},
                           arch);
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(InstrumentorReplacement, SubvectorLoopReferencesRejectQueryForkValidationAndEmission) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    SCOPED_TRACE(arch);
    const uint32_t machine = arch == ROCJITSU_CODE_ARCH_RDNA1 ? EF_AMDGPU_MACH_AMDGCN_GFX1010
                                                              : EF_AMDGPU_MACH_AMDGCN_GFX1030;
    const std::array<uint16_t, 2> opcodes =
        arch == ROCJITSU_CODE_ARCH_RDNA1
            ? std::array{rdna1::kSSubvectorLoopBeginSopk, rdna1::kSSubvectorLoopEndSopk}
            : std::array{rdna2::kSSubvectorLoopBeginSopk, rdna2::kSSubvectorLoopEndSopk};
    const uint16_t literal = arch == ROCJITSU_CODE_ARCH_RDNA1
                                 ? static_cast<uint16_t>(rdna1::OPR_SRC_SRC_LITERAL)
                                 : static_cast<uint16_t>(rdna2::OPR_SRC_SRC_LITERAL);
    const util::FailureOr<uint32_t> maybe_move = build_s_mov_b32(0, literal, arch);
    ASSERT_TRUE(maybe_move.succeeded());
    const uint32_t move = maybe_move.value();
    const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
    ASSERT_TRUE(maybe_nop.succeeded());
    const uint32_t nop = maybe_nop.value();
    const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
    ASSERT_TRUE(endpgm.succeeded());
    for (const uint16_t opcode : opcodes) {
      SCOPED_TRACE(opcode);
      // A label displacement of one names offset 8, inside the source literal
      // instruction at [4,12). Neither loop form exposes that reference today.
      const util::FailureOr<uint32_t> maybe_loop = build_sopk_encoding(arch, opcode, 0, 1);
      ASSERT_TRUE(maybe_loop.succeeded());
      const uint32_t loop = maybe_loop.value();
      const std::vector<uint32_t> words = {loop, move, 0x12345678u, endpgm.value()};
      const std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, machine);
      const AmdGpuCodeObject source(image.data(), image.size());
      ASSERT_TRUE(source.is_valid());
      std::unique_ptr<Decoder> decoder = Decoder::create(arch);
      ASSERT_NE(decoder, nullptr);
      const DecodeResult decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      EXPECT_FALSE(decoded.value()->branch_offset_bytes());
      const std::string mnemonic(decoded.value()->mnemonic());
      ASSERT_TRUE(mnemonic == "s_subvector_loop_begin" || mnemonic == "s_subvector_loop_end");

      Instrumentor streamed(source, arch);
      TextEditError error;
      EXPECT_EQ(streamed.instruction_boundaries(error), nullptr);
      EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences);
      EXPECT_EQ(error.offset, 0u);
      EXPECT_NE(error.message.find(mnemonic), std::string::npos);
      EXPECT_TRUE(streamed.fork(error).failed());
      EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences);
      EXPECT_EQ(error.offset, 0u);

      Instrumentor replacement(source, arch);
      replacement.add_point(replacement_point(4, {move, 0x12345678u}, {nop, nop}));
      const Instrumentor::ValidationResult validated = replacement.validate_points();
      EXPECT_TRUE(validated.sites.empty());
      EXPECT_FALSE(validated.errors.empty());
      ASSERT_TRUE(validated.edit_error);
      EXPECT_EQ(validated.edit_error->code, TextEditErrorCode::UnsupportedReferences);
      EXPECT_EQ(validated.edit_error->offset, 0u);
      EXPECT_NE(validated.edit_error->message.find(mnemonic), std::string::npos);
      const InstrumentedCodeObjectDebug result = replacement.patch_with_debug_summaries();
      expect_no_output(result);
      EXPECT_TRUE(result.patches.empty());
      ASSERT_TRUE(result.edit_error);
      EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
      EXPECT_EQ(result.edit_error->offset, 0u);

      // Exercise both the ordinary observation guard and the path that copies
      // an already decoded pristine CFG into shared source facts.
      Instrumentor observed(source, arch);
      observed.add_point_by_offset(4);
      const Instrumentor::ValidationResult eligible = observed.validate_points();
      ASSERT_TRUE(eligible.errors.empty()) << eligible.errors.front();
      ASSERT_EQ(eligible.sites.size(), 1u);
      EXPECT_EQ(observed.instruction_boundaries(error), nullptr);
      EXPECT_EQ(error.code, TextEditErrorCode::UnsupportedReferences);
      EXPECT_NE(error.message.find(mnemonic), std::string::npos);
      EXPECT_TRUE(observed.fork(error).failed());
      const InstrumentedCodeObjectDebug observation = observed.patch_with_debug_summaries();
      expect_no_output(observation);
      EXPECT_TRUE(observation.patches.empty());
      ASSERT_TRUE(observation.edit_error);
      EXPECT_EQ(observation.edit_error->code, TextEditErrorCode::UnsupportedReferences);
      EXPECT_EQ(observation.edit_error->offset, 0u);
      EXPECT_NE(observation.edit_error->message.find(mnemonic), std::string::npos);
      EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
    }
    // Known branch destinations at either source boundary remain supported.
    for (const int16_t displacement : {0, 2}) {
      SCOPED_TRACE(displacement);
      const util::FailureOr<uint32_t> branch = build_s_branch(displacement, arch);
      ASSERT_TRUE(branch.succeeded());
      const std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
          {branch.value(), move, 0x12345678u, endpgm.value()}, 0, 3, machine);
      const AmdGpuCodeObject source(image.data(), image.size());
      Instrumentor control(source, arch);
      control.add_point(replacement_point(4, {move, 0x12345678u}, {nop, nop}));
      const InstrumentedCodeObject result = control.patch();
      expect_pure_replacements(source, result,
                               {{.offset = 4,
                                 .expected = instruction_bytes({move, 0x12345678u}),
                                 .replacement = instruction_bytes({nop, nop})}},
                               arch);
    }
  }
}

TEST(InstrumentorReplacement, SubvectorLoopPayloadsRequireDestinationMetadata) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    SCOPED_TRACE(arch);
    const uint32_t machine = arch == ROCJITSU_CODE_ARCH_RDNA1 ? EF_AMDGPU_MACH_AMDGCN_GFX1010
                                                              : EF_AMDGPU_MACH_AMDGCN_GFX1030;
    const std::array<uint16_t, 2> opcodes =
        arch == ROCJITSU_CODE_ARCH_RDNA1
            ? std::array{rdna1::kSSubvectorLoopBeginSopk, rdna1::kSSubvectorLoopEndSopk}
            : std::array{rdna2::kSSubvectorLoopBeginSopk, rdna2::kSSubvectorLoopEndSopk};
    const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
    ASSERT_TRUE(maybe_nop.succeeded());
    const uint32_t nop = maybe_nop.value();
    const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
    ASSERT_TRUE(endpgm.succeeded());
    const std::vector<uint8_t> image =
        test::make_amdgpu_kernel_elf({nop, endpgm.value()}, 0, 3, machine);
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    for (const uint16_t opcode : opcodes) {
      SCOPED_TRACE(opcode);
      const util::FailureOr<uint32_t> loop = build_sopk_encoding(arch, opcode, 0, 0);
      ASSERT_TRUE(loop.succeeded());
      Instrumentor replacement(source, arch);
      replacement.add_point(replacement_point(0, {nop}, {loop.value()}));
      const Instrumentor::ValidationResult validated = replacement.validate_points();
      EXPECT_TRUE(validated.sites.empty());
      EXPECT_FALSE(validated.errors.empty());
      ASSERT_TRUE(validated.edit_error);
      EXPECT_EQ(validated.edit_error->code, TextEditErrorCode::InvalidInstruction);
      EXPECT_EQ(validated.edit_error->offset, 0u);
      const InstrumentedCodeObject result = replacement.patch();
      expect_no_output(result);
      ASSERT_TRUE(result.edit_error);
      EXPECT_EQ(result.edit_error->code, TextEditErrorCode::InvalidInstruction);
      EXPECT_EQ(result.edit_error->offset, 0u);
      EXPECT_EQ(std::memcmp(source.image_data(), image.data(), image.size()), 0);
    }
  }
}

TEST(InstrumentorReplacement, RelocatableSymbolEndpointsKeepSectionRelativeCoordinates) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  ASSERT_TRUE(maybe_nop.succeeded());
  const uint32_t nop = maybe_nop.value();
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(endpgm.succeeded());
  for (const bool end_reference : {false, true}) {
    SCOPED_TRACE(end_reference);
    for (const uint64_t endpoint : {8u, 12u, 16u}) {
      SCOPED_TRACE(endpoint);
      std::vector<uint8_t> image =
          test::make_gfx1200_kernel_elf({nop, nop, 0xbe8000ffu, 0x12345678u, endpgm.value()}, 0);
      const AmdGpuCodeObject original(image.data(), image.size());
      const Section &original_text = *original.text_sections().front();
      std::vector<Elf64_Shdr> sections = CodeObjectPatcher(original).section_headers();
      const std::vector<Elf64_Shdr>::iterator symbols =
          std::ranges::find(sections, SHT_SYMTAB, &Elf64_Shdr::sh_type);
      ASSERT_NE(symbols, sections.end());
      Elf64_Ehdr header = {};
      std::memcpy(&header, image.data(), sizeof(header));
      header.e_type = ET_REL;
      header.e_phoff = 0;
      header.e_phnum = 0;
      // Remove linked addresses with the program headers. Text growth must not
      // shift inherited linked data addresses without their LOAD alignment.
      for (Elf64_Shdr &section : sections) {
        if ((section.sh_flags & SHF_ALLOC) != 0)
          section.sh_addr = 0;
      }
      // An ET_REL symbol remains section-relative even when sh_addr is nonzero.
      sections[*original_text.sectionHeaderIndex()].sh_addr = 8;
      const Elf64_Sym symbol = {
          .st_name = 0,
          .st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeNone),
          .st_other = 0,
          .st_shndx = static_cast<uint16_t>(*original_text.sectionHeaderIndex()),
          .st_value = end_reference ? 0 : endpoint,
          .st_size = end_reference ? endpoint : 0};
      std::memcpy(image.data() + symbols->sh_offset + sizeof(Elf64_Sym), &symbol, sizeof(symbol));
      std::memcpy(image.data(), &header, sizeof(header));
      std::memcpy(image.data() + header.e_shoff, sections.data(),
                  sections.size() * sizeof(Elf64_Shdr));
      const std::vector<uint8_t> pristine = image;
      const AmdGpuCodeObject source(image.data(), image.size());
      ASSERT_TRUE(source.is_valid());
      Instrumentor instrumentor(source, arch);
      instrumentor.add_point(replacement_point(8, {0xbe8000ffu, 0x12345678u}, {nop, nop}));
      const InstrumentedCodeObject result = instrumentor.patch();
      if (endpoint == 12) {
        expect_no_output(result);
        ASSERT_TRUE(result.edit_error);
        EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
        EXPECT_EQ(result.edit_error->offset, endpoint);
      } else {
        ASSERT_TRUE(result.errors.empty()) << result.errors.front();
        ASSERT_EQ(result.sites.size(), 1u);
        expect_pure_replacements(source, result,
                                 {{.offset = 8,
                                   .expected = instruction_bytes({0xbe8000ffu, 0x12345678u}),
                                   .replacement = instruction_bytes({nop, nop})}},
                                 arch);
      }
      EXPECT_EQ(image, pristine);
      EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
    }
  }
}

void expect_loaded_prefix_relocation_policy(uint64_t bytes_before_text, bool overlaps_text) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint64_t loaded_prefix_size = sizeof(uint64_t);
  const util::FailureOr<uint32_t> barrier =
      build_sop1_encoding(arch, cdna5::kSBarrierSignalSop1, 0, 193);
  const util::FailureOr<uint32_t> nop = build_s_nop(0, arch);
  const util::FailureOr<uint32_t> replacement_nop = build_s_nop(1, arch);
  const util::FailureOr<uint32_t> endpgm = build_s_endpgm(arch);
  ASSERT_TRUE(barrier.succeeded());
  ASSERT_TRUE(nop.succeeded());
  ASSERT_TRUE(replacement_nop.succeeded());
  ASSERT_TRUE(endpgm.succeeded());
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      {barrier.value(), nop.value(), endpgm.value()}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const AmdGpuCodeObject original(image.data(), image.size());
  ASSERT_TRUE(original.is_valid());
  const Section &original_text = *original.text_sections().front();
  const CodeObjectPatcher original_patcher(original);
  std::vector<Elf64_Shdr> sections = original_patcher.section_headers();
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_VERSION] = 1;
  header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  const std::vector<Elf64_Phdr>::iterator text_load =
      std::ranges::find_if(*segments, [](const Elf64_Phdr &segment) {
        return segment.p_type == PT_LOAD && (segment.p_flags & PF_X) != 0;
      });
  const std::vector<Elf64_Phdr>::iterator data_load =
      std::ranges::find_if(*segments, [](const Elf64_Phdr &segment) {
        return segment.p_type == PT_LOAD && (segment.p_flags & PF_X) == 0;
      });
  const std::vector<Elf64_Shdr>::iterator symbols =
      std::ranges::find(sections, SHT_SYMTAB, &Elf64_Shdr::sh_type);
  ASSERT_NE(text_load, segments->end());
  ASSERT_NE(data_load, segments->end());
  ASSERT_NE(symbols, sections.end());
  ASSERT_EQ(text_load->p_offset, original_text.sectionOffset());
  ASSERT_EQ(text_load->p_vaddr, original_text.vaddr());
  ASSERT_GE(text_load->p_offset, loaded_prefix_size);
  ASSERT_LE(bytes_before_text, loaded_prefix_size);
  text_load->p_offset -= loaded_prefix_size;
  text_load->p_vaddr -= loaded_prefix_size;
  text_load->p_paddr -= loaded_prefix_size;
  text_load->p_filesz += loaded_prefix_size;
  text_load->p_memsz += loaded_prefix_size;
  text_load->p_flags |= 0x2u; // PF_W
  std::fill_n(image.data() + text_load->p_offset, loaded_prefix_size, uint8_t{0x5a});

  // This is actual loaded, file-backed padding, not an address before the LOAD.
  // ABS64 writes eight bytes even when only its tail reaches the first instruction.
  const uint64_t relocation_offset = header.e_shoff;
  const uint64_t relocation_vaddr = data_load->p_vaddr + relocation_offset - data_load->p_offset;
  const Elf64_Rela relocation = {.r_offset = original_text.vaddr() - bytes_before_text,
                                 .r_info = (uint64_t{1} << 32) | R_AMDGPU_ABS64,
                                 .r_addend = 0};
  const uint32_t symbol_table_index = static_cast<uint32_t>(symbols - sections.begin());
  symbols->sh_type = SHT_DYNSYM;
  header.e_shoff += sizeof(relocation);
  data_load->p_filesz = header.e_shoff - data_load->p_offset;
  data_load->p_memsz = data_load->p_filesz;
  sections.push_back({.sh_name = 0,
                      .sh_type = SHT_RELA,
                      .sh_flags = SHF_ALLOC,
                      .sh_addr = relocation_vaddr,
                      .sh_offset = relocation_offset,
                      .sh_size = sizeof(relocation),
                      .sh_link = symbol_table_index,
                      .sh_info = SHN_UNDEF,
                      .sh_addralign = alignof(Elf64_Rela),
                      .sh_entsize = sizeof(relocation)});
  header.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(header.e_shoff + sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data() + relocation_offset, &relocation, sizeof(relocation));
  std::memcpy(image.data(), &header, sizeof(header));
  std::memcpy(image.data() + header.e_phoff, segments->data(),
              segments->size() * sizeof(Elf64_Phdr));
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));

  const std::vector<uint8_t> pristine = image;
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.target_id(), ROCJITSU_CODE_TARGET_GFX1250);
  ASSERT_EQ(source.text_sections().size(), 1u);
  const Section &text = *source.text_sections().front();
  ASSERT_EQ(text.vaddr(), original_text.vaddr());
  ASSERT_EQ(classify_rocr_relocation_section(header, sections, sections.back()),
            RocrRelocationSectionMode::Dynamic);
  ASSERT_EQ(sections[symbol_table_index].sh_type, SHT_DYNSYM);
  Elf64_Sym symbol = {};
  std::memcpy(&symbol, image.data() + sections[symbol_table_index].sh_offset + sizeof(symbol),
              sizeof(symbol));
  ASSERT_NE(symbol.st_shndx, *text.sectionHeaderIndex());
  ASSERT_NE(symbol.st_shndx, SHN_UNDEF);
  ASSERT_EQ(elf_reloc_type(relocation.r_info), R_AMDGPU_ABS64);
  ASSERT_LT(relocation.r_offset, text.vaddr());
  ASSERT_GE(relocation.r_offset, text_load->p_vaddr);
  ASSERT_LE(relocation.r_offset + sizeof(uint64_t), text_load->p_vaddr + text_load->p_filesz);
  ASSERT_EQ(relocation.r_offset + sizeof(uint64_t) > text.vaddr(), overlaps_text);
  ASSERT_TRUE(std::ranges::none_of(sections, [&](const Elf64_Shdr &section) {
    return (section.sh_flags & SHF_ALLOC) != 0 && relocation.r_offset >= section.sh_addr &&
           relocation.r_offset - section.sh_addr < section.sh_size;
  }));
  const uint64_t write_file_offset = text_load->p_offset + relocation.r_offset - text_load->p_vaddr;
  ASSERT_LT(write_file_offset, text.sectionOffset());
  ASSERT_LE(write_file_offset + sizeof(uint64_t), image.size());
  ASSERT_EQ(image[write_file_offset], 0x5a);
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), source.target_id());
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 1> barrier_words = {barrier.value()};
  const DecodeResult decoded = decoder->decode_window(barrier_words);
  ASSERT_TRUE(decoded.succeeded());
  ASSERT_EQ(decoded.value()->mnemonic(), "s_barrier_signal");
  ASSERT_EQ(decoded.value()->size(), sizeof(uint32_t));
  const CodeObjectPatcher patcher(source);
  EXPECT_FALSE(patcher.has_malformed_rocr_relocation_section());
  EXPECT_FALSE(patcher.has_rocr_rejected_none_relocation());
  EXPECT_FALSE(patcher.has_unsupported_relocation_to_text());
  EXPECT_EQ(patcher.has_relocations_within_text(), overlaps_text);

  Instrumentor instrumentor(source, arch);
  instrumentor.add_point(replacement_point(4, {nop.value()}, {replacement_nop.value()}));
  instrumentor.add_point(replacement_point(0, {barrier.value()}, {nop.value()}));
  const Instrumentor::ValidationResult validation = instrumentor.validate_points();
  const InstrumentedCodeObjectDebug result = instrumentor.patch_with_debug_summaries();
  if (overlaps_text) {
    EXPECT_FALSE(validation.errors.empty());
    EXPECT_TRUE(validation.sites.empty());
    EXPECT_TRUE(validation.edit_error);
    if (validation.edit_error) {
      EXPECT_EQ(validation.edit_error->code, TextEditErrorCode::UnsupportedReferences);
    }
    expect_no_output(result);
    EXPECT_TRUE(result.patches.empty());
    EXPECT_TRUE(result.edit_error);
    if (result.edit_error) {
      EXPECT_EQ(result.edit_error->code, TextEditErrorCode::UnsupportedReferences);
    }
  } else {
    ASSERT_TRUE(validation.errors.empty()) << validation.errors.front();
    EXPECT_FALSE(validation.edit_error);
    ASSERT_EQ(validation.sites.size(), 2u);
    expect_pure_replacements(source, result,
                             {{.offset = 4,
                               .expected = instruction_bytes({nop.value()}),
                               .replacement = instruction_bytes({replacement_nop.value()})},
                              {.offset = 0,
                               .expected = instruction_bytes({barrier.value()}),
                               .replacement = instruction_bytes({nop.value()})}},
                             arch);
  }
  EXPECT_EQ(image, pristine);
  EXPECT_EQ(std::memcmp(source.image_data(), pristine.data(), pristine.size()), 0);
}

TEST(InstrumentorReplacement, RejectsRelocationWriteStartingInLoadedPrefix) {
  expect_loaded_prefix_relocation_policy(4, true);
}

TEST(InstrumentorReplacement, AllowsRelocationWriteEndingAtTextBoundary) {
  expect_loaded_prefix_relocation_policy(8, false);
}

} // namespace
} // namespace rocjitsu
