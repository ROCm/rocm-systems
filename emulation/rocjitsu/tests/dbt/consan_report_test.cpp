// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "patch/consan/consan_model_test_support.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_conflict_rendering.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <ranges>
#include <string>

namespace rocjitsu::consan::hook {
namespace {

TEST(ConSanReportTest, CompactEvidencePreservesClassificationAcrossMetadataCopies) {
  static_assert(sizeof(Evidence) == 128);
  static_assert(alignof(Evidence) == alignof(uint64_t));
  static_assert(sizeof(SyncDecodeResult) == 32);

  Evidence evidence;
  evidence.sync.classification = SyncClassification::ChangedDuringRead;
  const SyncMetadata metadata{
      .address = 0xfedcba9876543210ull,
      .byte_count = 16,
      .kind = SyncMetadataKind::Atomic,
      .role = SyncRole::Release,
      .scope = SyncScope::Workgroup,
      .epoch_before = 7,
      .epoch_after = 7,
  };
  // Metadata assignment must not overwrite the classification in its tail padding.
  evidence.sync.metadata = metadata;
  EXPECT_EQ(evidence.sync.classification, SyncClassification::ChangedDuringRead);
  const auto copy = evidence;
  EXPECT_EQ(copy.sync.metadata, metadata);
  EXPECT_EQ(copy.sync.classification, SyncClassification::ChangedDuringRead);
  evidence.sync.metadata = {};
  EXPECT_EQ(evidence.sync.classification, SyncClassification::ChangedDuringRead);
}

Evidence access(uint32_t index, uint32_t owner, uint32_t byte, ShadowAccessKind kind) {
  Evidence result;
  result.index = index;
  result.generation = 7;
  result.dispatch_id = 11;
  result.workgroup_x = 2;
  result.workgroup_y = 3;
  result.workgroup_z = 4;
  result.epoch = 3;
  result.entry = {.valid = true,
                  .kind = kind,
                  .owner_id = owner,
                  .epoch = 3,
                  .generation = 7,
                  .start_byte = byte,
                  .byte_count = 4};
  return result;
}

std::string conflict_line(const ConflictRendering &rendered) {
  const auto found = std::ranges::find_if(
      rendered.details, [](const auto &line) { return line.text.starts_with("ConSan conflict "); });
  return found == rendered.details.end() ? std::string{} : found->text;
}

TEST(ConSanReportTest, MaskIsAcceptedOnlyWithItsCommittedCurrentWindow) {
  ReportPipelineInput input;
  auto &layout = input.layout;
  layout.watchpoint_capacity = layout.causal_window_capacity = 1;
  layout.causal_windows_offset = sizeof(ReportHeader);
  layout.watchpoints_offset = layout.causal_windows_offset + sizeof(CausalWindow);
  layout.required_bytes = layout.watchpoints_offset + sizeof(uint64_t);
  std::vector<uint64_t> storage(layout.required_bytes / sizeof(uint64_t));
  auto bytes = std::span(reinterpret_cast<uint8_t *>(storage.data()), layout.required_bytes);
  ReportHeader header;
  header.generation = 7;
  header.causal_window_count = 1;
  auto *window = reinterpret_cast<CausalWindow *>(bytes.data() + layout.causal_windows_offset);
  *window = {.generation = 7,
             .dispatch_id = 11,
             .first_entry = 0,
             .entry_count = 1,
             .publication_state = static_cast<uint32_t>(CausalPublicationState::Ready),
             .exact_lane_mask = 0x8000000000000001ull};
  const uint64_t packed = pack_watchpoint_entry(ShadowAccessKind::Write, 0, 0, 7, 0, 4);
  std::memcpy(bytes.data() + layout.watchpoints_offset, &packed, sizeof(packed));
  ReportSummary summary;
  auto decoded = decode_evidence(input, header, bytes, summary);
  ASSERT_EQ(decoded.evidence.size(), 1u);
  EXPECT_EQ(decoded.evidence[0].exact_lane_mask, window->exact_lane_mask);
  EXPECT_EQ(analyze_conflicts(decoded.evidence, true).conflict_count, 1u);
  AccessStaticMapping mapping;
  mapping.range_count = mapping.bank_count = 1;
  mapping.uniform_lds_store = true;
  AccessStaticMetadata metadata{{mapping}, false};
  input.static_metadata = &metadata;
  for (bool malformed : {false, true}) {
    auto &details = metadata;
    details.malformed = malformed;
    summary = {};
    auto proof = decode_evidence(input, header, bytes, summary);
    EXPECT_EQ(analyze_conflicts(proof.evidence, true, 8, true).conflict_count, malformed ? 1u : 0u);
  }
  auto &details = metadata;
  details.malformed = false;
  details.mappings.push_back(mapping); // Even agreeing overlapping maps are ambiguous.
  summary = {};
  auto ambiguous = decode_evidence(input, header, bytes, summary);
  EXPECT_EQ(analyze_conflicts(ambiguous.evidence, true, 8, true).conflict_count, 1u);
  input.static_metadata = nullptr;
  for (auto state : {CausalPublicationState::Empty, CausalPublicationState::Publishing}) {
    window->publication_state = static_cast<uint32_t>(state);
    // A partially stored or leftover mask must not create evidence.
    summary = {};
    EXPECT_TRUE(decode_evidence(input, header, bytes, summary).evidence.empty());
  }
  window->publication_state = static_cast<uint32_t>(CausalPublicationState::Ready);
  window->generation += uint64_t{1} << watchpoint::generation_bits;
  summary = {};
  EXPECT_TRUE(decode_evidence(input, header, bytes, summary).evidence.empty());
  EXPECT_EQ(summary.malformed_snapshot_count, 1u);
}

TEST(ConSanReportTest, ExactSingleGroupWriteReportsParticipatingLanes) {
  auto entry = access(0, 0, 16, ShadowAccessKind::Write);
  for (uint64_t mask : {uint64_t{0x80000005}, uint64_t{0x8000000100000000}}) {
    entry.exact_lane_mask = mask;
    const std::array evidence{entry};
    const auto result = analyze_conflicts(evidence, true);
    ASSERT_EQ(result.conflict_count, 1u);
    ASSERT_EQ(result.examples.size(), 1u);
    EXPECT_EQ(result.examples[0].first.exact_lane_mask & result.examples[0].second.exact_lane_mask,
              0u);
    EXPECT_EQ(result.examples[0].first.exact_lane_mask | result.examples[0].second.exact_lane_mask,
              mask);
    EXPECT_EQ(result.examples[0].first.entry.owner_id, 0u);
    EXPECT_EQ(result.examples[0].second.entry.owner_id, 0u);
    const auto capped = analyze_conflicts(evidence, true, 0);
    EXPECT_EQ(capped.conflict_count, 1u);
    EXPECT_TRUE(capped.examples.empty());
  }
}

TEST(ConSanReportTest, UniformStoreSuppressesOnlyItsOwnLaneCollision) {
  AccessStaticMapping mapping;
  mapping.uniform_lds_store = true;
  auto first = access(0, 0, 16, ShadowAccessKind::Write);
  first.exact_lane_mask = 3;
  first.static_mapping = &mapping;
  EXPECT_EQ(analyze_conflicts(std::array{first}, true).conflict_count, 1u);
  EXPECT_EQ(analyze_conflicts(std::array{first}, true, 8, false).conflict_count, 1u);
  EXPECT_EQ(analyze_conflicts(std::array{first}, true, 8, true).conflict_count, 0u);
  auto second = first;
  second.entry.owner_id = 1;
  EXPECT_EQ(analyze_conflicts(std::array{first, second}, true, 8, true).conflict_count, 1u);
  second.entry.kind = ShadowAccessKind::Read;
  EXPECT_EQ(analyze_conflicts(std::array{first, second}, true, 8, true).conflict_count, 1u);
  first.static_mapping = nullptr;
  EXPECT_EQ(analyze_conflicts(std::array{first}, true, 8, true).conflict_count, 1u);
}

TEST(ConSanReportTest, ExactGroupDoesNotInventReadAtomicOrSingleLaneRaces) {
  for (auto kind : {ShadowAccessKind::Read, ShadowAccessKind::Atomic, ShadowAccessKind::Write}) {
    auto entry = access(0, 0, 16, kind);
    for (uint64_t mask : {0ull, 1ull, 0x8000000000000000ull, 3ull}) {
      entry.exact_lane_mask = mask;
      const auto result = analyze_conflicts(std::array{entry}, false);
      EXPECT_EQ(result.conflict_count, kind == ShadowAccessKind::Write && mask == 3 ? 1u : 0u);
    }
  }
}

TEST(ConSanReportTest, CrossWaveMasksAreIndependentAndMissingIsExplicit) {
  DecodedEvidence mode;
  mode.evidence = {access(0, 0, 0, ShadowAccessKind::Write),
                   access(1, 1, 0, ShadowAccessKind::Read)};
  mode.evidence[0].exact_lane_mask = 1;
  for (uint64_t read_mask : {0u, 0xf0u}) {
    mode.evidence[1].exact_lane_mask = read_mask;
    const auto analysis = analyze_conflicts(mode.evidence, false);
    ASSERT_EQ(analysis.conflict_count, 1u);
    const auto line = conflict_line(render_conflicts({}, {}, {}, mode, analysis));
    EXPECT_NE(line.find("first_lanes=0x0000000000000001"), std::string::npos);
    EXPECT_NE(line.find(read_mask ? "second_lanes=0x00000000000000f0" : "second_lanes=unavailable"),
              std::string::npos);
  }
}

TEST(ConSanReportTest, ConflictSitesRemainVisibleBeyondDetailLimit) {
  DecodedEvidence mode;
  for (uint32_t index = 0; index < 68; ++index)
    mode.evidence.push_back(access(index, 0, 1024 + 4 * index, ShadowAccessKind::Read));
  mode.evidence.push_back(access(68, 1, 16, ShadowAccessKind::Write));
  mode.evidence.push_back(access(69, 2, 16, ShadowAccessKind::Read));
  std::array<AccessStaticMapping, 2> mappings{};
  mappings[0].instruction_offset = 0x120;
  mappings[1].instruction_offset = 0x240;
  for (size_t i = 0; i < mappings.size(); ++i) {
    mappings[i].first_slot = 68 + i;
    mappings[i].range_count = 1;
    mappings[i].bank_count = 1;
    mode.evidence[68 + i].static_mapping = &mappings[i];
  }
  const auto analysis = analyze_conflicts(mode.evidence, false);
  ASSERT_EQ(analysis.conflict_count, 1u);
  ReportPipelineInput input;
  input.reader = 101;
  input.input_fingerprint = "object-a";
  const auto rendered = render_conflicts(input, {}, {}, mode, analysis);
  const auto line = conflict_line(rendered);
  EXPECT_NE(line.find("first_instruction=0x120 second_instruction=0x240"), std::string::npos);
  EXPECT_NE(line.find("code_object=object-a"), std::string::npos);
  EXPECT_NE(line.find("dispatch=0xb workgroup=(2,3,4)"), std::string::npos);
  EXPECT_TRUE(std::ranges::any_of(rendered.details, [](const auto &entry) {
    return entry.text.find("omitted=6 after log limit=64") != std::string::npos;
  }));
}

TEST(ConSanReportTest, MissingMappingIsNotInstructionZeroAndObjectsRemainDistinct) {
  DecodedEvidence mode;
  mode.evidence = {access(0, 1, 16, ShadowAccessKind::Write),
                   access(1, 2, 16, ShadowAccessKind::Read)};
  AccessStaticMapping mapping;
  mapping.first_slot = 1;
  mapping.range_count = 1;
  mapping.bank_count = 1;
  mapping.instruction_offset = 0;
  mode.evidence[1].static_mapping = &mapping;
  const auto analysis = analyze_conflicts(mode.evidence, false);
  ASSERT_EQ(analysis.conflict_count, 1u);
  for (const char *object : {"object-a", "object-b"}) {
    ReportPipelineInput input;
    input.input_fingerprint = object;
    const auto line = conflict_line(render_conflicts(input, {}, {}, mode, analysis));
    EXPECT_NE(line.find("first_instruction=unavailable second_instruction=0x0"), std::string::npos);
    EXPECT_NE(line.find(std::string("code_object=") + object), std::string::npos);
  }
}

TEST(ConSanReportTest, OverlappingMappingsDoNotInventUniqueInstructionAttribution) {
  DecodedEvidence mode;
  mode.evidence = {access(0, 1, 16, ShadowAccessKind::Write),
                   access(1, 2, 16, ShadowAccessKind::Read)};
  AccessStaticMetadata metadata;
  for (uint64_t pc : {0x120u, 0x240u}) {
    AccessStaticMapping mapping;
    mapping.range_count = 1;
    mapping.bank_count = 2;
    mapping.instruction_offset = pc;
    metadata.mappings.push_back(mapping);
  }
  for (auto &entry : mode.evidence)
    entry.static_mapping = &metadata.mappings.front();
  const auto analysis = analyze_conflicts(mode.evidence, false);
  ASSERT_EQ(analysis.conflict_count, 1u);
  AccessStaticMetadata runtime_metadata = metadata;
  ReportPipelineInput input;
  input.static_metadata = &runtime_metadata;
  const auto line = conflict_line(render_conflicts(input, {}, {}, mode, analysis));
  EXPECT_NE(line.find("first_instruction=ambiguous second_instruction=ambiguous"),
            std::string::npos);
  // Multiple aliases of the same original instruction still give a unique PC.
  runtime_metadata.mappings[1].instruction_offset = 0x120;
  const auto aliases = conflict_line(render_conflicts(input, {}, {}, mode, analysis));
  EXPECT_NE(aliases.find("first_instruction=0x120 second_instruction=0x120"), std::string::npos);
  // A supplied mapping table is authoritative even if an entry still has an
  // older mapping pointer. Do not invent attribution for a slot it omits.
  runtime_metadata.mappings.clear();
  const auto missing = conflict_line(render_conflicts(input, {}, {}, mode, analysis));
  EXPECT_NE(missing.find("first_instruction=unavailable second_instruction=unavailable"),
            std::string::npos);
}

TEST(ConSanReportTest, ExamplesAreBoundedAndDeduplicatedWithoutChangingPairCount) {
  DecodedEvidence mode;
  AccessStaticMapping mapping;
  mapping.instruction_offset = 0x100;
  mapping.bank_count = 1;
  for (uint32_t i = 0; i < 3; ++i) {
    mode.evidence.push_back(access(i, i + 1, 16, ShadowAccessKind::Write));
    mode.evidence.back().static_mapping = &mapping;
  }
  mode.evidence.push_back(mode.evidence.front());
  mode.evidence.back().index = 3;
  for (uint32_t limit : {0u, 1u, 2u, 8u}) {
    const auto analysis = analyze_conflicts(mode.evidence, false, limit);
    EXPECT_EQ(analysis.conflict_count, 5u);
    EXPECT_EQ(analysis.examples.size(), std::min(limit, 3u));
    ReportPipelineInput input;
    input.input_fingerprint = "bounded";
    const auto rendered = render_conflicts(input, {}, {}, mode, analysis);
    EXPECT_EQ(std::ranges::count_if(
                  rendered.details,
                  [](const auto &line) { return line.text.starts_with("ConSan conflict "); }),
              analysis.examples.size());
    EXPECT_NE(rendered.summary_fields.find("conflict_pairs_without_example=" +
                                           std::to_string(5u - std::min(limit, 3u))),
              std::string::npos);
    if (limit) {
      EXPECT_EQ(analysis.examples.front().first.index, 0u);
      EXPECT_EQ(analysis.examples.front().second.index, 1u);
    }
  }
}

TEST(ConSanReportTest, ReadOnlyAndOrderedEpochsDoNotCreateExamples) {
  auto first = access(0, 1, 16, ShadowAccessKind::Read);
  auto second = access(1, 2, 16, ShadowAccessKind::Read);
  auto analysis = analyze_conflicts(std::array{first, second}, false);
  EXPECT_EQ(analysis.conflict_count, 0u);
  EXPECT_TRUE(analysis.examples.empty());
  first.entry.kind = ShadowAccessKind::Write;
  ++second.epoch;
  ++second.entry.epoch;
  analysis = analyze_conflicts(std::array{first, second}, false);
  EXPECT_EQ(analysis.conflict_count, 0u);
  EXPECT_TRUE(analysis.examples.empty());
}

} // namespace
} // namespace rocjitsu::consan::hook
