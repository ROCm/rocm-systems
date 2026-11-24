// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "capture_fixture.h"
#include "rocjitsu/code/consan/analysis/analyze.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/target.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>

namespace rocjitsu::consan {
namespace {

class ConSanAnalysis : public ::testing::Test {
protected:
  void SetUp() override {
    capture_.header = test::capture_header();
    test::manifest(capture_.header).entry.original_group_bytes = 128;
    test::manifest(capture_.header).entry.prepared_group_bytes = 128;
    append_execution();
  }

  void append_execution(LaunchDimensions dimensions = {{64, 1, 1}, {64, 1, 1}}) {
    test::refresh_image(capture_.header);
    const uint64_t serial = capture_.executions.size() + 1;
    CapturedExecution execution =
        test::execution(test::manifest(capture_.header), serial, serial, dimensions);
    const FailureOr<LaunchLayout> layout = plan_launch(
        test::manifest(capture_.header), execution.dimensions, execution.group_bytes, 0);
    ASSERT_TRUE(layout.succeeded());
    stride_ = layout.value().row_stride;
    for (uint32_t wave = 0; wave < layout.value().wave_count; ++wave) {
      wire::write_u32(execution.report, row(wave) + wire::kRowStatus,
                      wire::kRowEntered | wire::kRowExited);
    }
    capture_.executions.push_back(std::move(execution));
  }

  size_t row(uint32_t wave) const { return kReportHeaderBytes + wave * stride_; }
  size_t record(uint32_t wave, uint32_t site, uint32_t address_group = 0) const {
    return row(wave) + kRowHeaderBytes +
           (site * kAddressGroupsPerSite + address_group) * kAccessRecordBytes;
  }

  size_t completion(uint32_t wave, uint32_t slot = 0) const {
    return row(wave) + sync_offset(test::manifest(capture_.header).sites.size()) +
           slot * kSyncRecordBytes;
  }

  void access(uint32_t wave, uint32_t site, uint32_t sequence, uint32_t offset = 0,
              uint64_t mask = 1, uint32_t address_group = 0, uint32_t execution = 0) {
    std::vector<uint8_t> &report = capture_.executions[execution].report;
    const Manifest *owner =
        selected_manifest(capture_.header.image, capture_.executions[execution].entry_key);
    ASSERT_NE(owner, nullptr);
    const SiteFacts &facts = owner->sites[site];
    const size_t base = record(wave, site, address_group);
    for (const auto &[field, value] : std::array<std::pair<uint32_t, uint32_t>, 9>{
             {{wire::kRecordCommit, wire::kAccessCommitted},
              {wire::kRecordAnalysis, uint32_t(facts.analysis)},
              {wire::kRecordKind, uint32_t(facts.record)},
              {wire::kRecordSite, site},
              {wire::kRecordRange, 0},
              {wire::kRecordSequence, sequence},
              {wire::kRecordLdsOffset, offset},
              {wire::kRecordWidth, facts.width},
              {wire::kRecordFlags, wire::kExactGuestMask}}}) {
      wire::write_u32(report, base + field, value);
    }
    const uint32_t phase = static_cast<uint32_t>(
        std::ranges::count_if(owner->barriers, [&](const BarrierFacts &barrier) {
          return barrier.original_wait < facts.original_offset;
        }));
    wire::write_u32(report, base + wire::kRecordPhase, phase);
    wire::write_u64(report, base + wire::kRecordGuestMask, mask);
    const uint32_t final = *wire::read_u32(report, row(wave) + wire::kRowFinalSequence);
    wire::write_u32(report, row(wave) + wire::kRowFinalSequence, std::max(final, sequence));
  }

  void set_transpose_site(uint32_t target) {
    Manifest &facts = test::manifest(capture_.header);
    facts.target = target;
    facts.wave_size = target == 950 ? 64 : 32;
    facts.sites[1].width = 8;
    facts.sites[1].lds_ranges.shape = LdsRangeShape::Transpose8;
    capture_.executions.clear();
  }

  void transpose_access(uint64_t mask, uint64_t anchor_exec, uint32_t group = 0,
                        uint32_t offset = 0) {
    access(0, 1, 1, offset, mask, group);
    std::vector<uint8_t> &report = capture_.executions.front().report;
    wire::write_u32(report, record(0, 1, group) + wire::kRecordFlags,
                    wire::kExactTransposeRequestMask);
    wire::write_u64(report, record(0, 1, group) + wire::kRecordAnchorExec, anchor_exec);
  }

  FailureOr<std::vector<uint8_t>> encode() {
    test::refresh_image(capture_.header);
    CaptureEncoder encoder;
    FailureOr<std::vector<uint8_t>> bytes = encoder.start(capture_.header, error_.emitter());
    if (bytes.failed()) {
      return Result::failure();
    }
    for (const CapturedExecution &execution : capture_.executions) {
      const FailureOr<std::vector<uint8_t>> frame = encoder.append(execution, error_.emitter());
      if (frame.failed()) {
        return Result::failure();
      }
      bytes.value().insert(bytes.value().end(), frame.value().begin(), frame.value().end());
    }
    const FailureOr<std::vector<uint8_t>> end = encoder.finish(error_.emitter());
    if (end.failed()) {
      return Result::failure();
    }
    bytes.value().insert(bytes.value().end(), end.value().begin(), end.value().end());
    return bytes;
  }

  FailureOr<LdsRaceResult> analyze_typed(Capture &capture, AnalysisOptions options = {},
                                         util::DiagnosticEmitter diagnostic = {}) {
    test::refresh_image(capture.header);
    return analyze_capture(capture, options, diagnostic);
  }

  FailureOr<LdsRaceResult> analyze(AnalysisOptions options = {}) {
    const FailureOr<std::vector<uint8_t>> bytes = encode();
    if (bytes.failed()) {
      return Result::failure();
    }
    const FailureOr<Capture> decoded = decode_capture(bytes.value(), error_.emitter());
    if (decoded.failed()) {
      return Result::failure();
    }
    return analyze_capture(decoded.value(), options, error_.emitter());
  }

  void many_sites(uint32_t count) {
    Manifest &facts = test::manifest(capture_.header);
    facts.sites.clear();
    facts.original_text_bytes = 8 * count + 64;
    facts.prepared_text_bytes = 128 * count + 768;
    facts.entry.original_descriptor_offset = facts.original_text_bytes + 64;
    facts.entry.prepared_descriptor_offset = facts.prepared_text_bytes + 64;
    facts.entry.prepared_entry_offset = 128 * count + 256;
    facts.original.bytes = facts.entry.original_descriptor_offset + 64;
    facts.prepared.bytes = facts.entry.prepared_descriptor_offset + 64;
    for (uint32_t site = 0; site < count; ++site) {
      facts.sites.push_back({.id = site,
                             .analysis = AnalysisKind::LdsRaceDetection,
                             .access = AccessKind::Write,
                             .width = 4,
                             .original_offset = uint64_t{site} * 8,
                             .original_size = 8,
                             .prepared_guest_offset = uint64_t{site} * 128,
                             .prepared_guest_size = 8,
                             .probe_offset = uint64_t{site} * 128 + 64,
                             .probe_size = 64,
                             .first_slot = site * kAddressGroupsPerSite});
    }
    capture_.executions.clear();
    append_execution();
  }

  void barrier() {
    Manifest &facts = test::manifest(capture_.header);
    facts.target = 1250;
    facts.scope = ScopeKind::NativeLdsWorkgroupBarrier;
    facts.sites[1].original_offset = 32;
    facts.barriers = {BarrierFacts{.original_drain = 8,
                                   .original_signal = 12,
                                   .original_wait = 16,
                                   .prepared_signal = 320,
                                   .prepared_wait = 324,
                                   .completion_probe = 328,
                                   .probe_size = 64}};
    const uint32_t old_stride = stride_;
    const std::vector<uint8_t> old_report = capture_.executions.front().report;
    const LaunchDimensions dimensions = capture_.executions.front().dimensions;
    capture_.executions.clear();
    append_execution(dimensions);
    for (uint32_t wave = 0; wave < (dimensions.grid[0] / dimensions.workgroup[0]) *
                                       ((dimensions.workgroup[0] + 31) / 32);
         ++wave) {
      std::copy_n(old_report.begin() + kReportHeaderBytes + wave * old_stride,
                  std::min(old_stride, stride_),
                  capture_.executions.front().report.begin() + row(wave));
      for (const SiteFacts &site : facts.sites) {
        for (uint32_t group = 0; group < kAddressGroupsPerSite; ++group) {
          const size_t offset = record(wave, site.id, group);
          if (wire::read_u32(capture_.executions.front().report, offset + wire::kRecordCommit) ==
              wire::kAccessCommitted) {
            wire::write_u32(capture_.executions.front().report, offset + wire::kRecordPhase,
                            site.id == 1 ? 1 : 0);
          }
        }
      }
    }
    std::vector<uint8_t> &report = capture_.executions.front().report;
    for (const uint32_t wave : {0u, 1u}) {
      wire::write_u32(report, row(wave) + wire::kRowFinalPhase, 1);
      wire::write_u32(report, completion(wave) + wire::kSyncPhase, 1);
      wire::write_u32(report, completion(wave) + wire::kSyncCommit, wire::kBarrierCompleted);
      wire::write_u32(report, completion(wave) + wire::kSyncGroup, 0);
      wire::write_u32(report, completion(wave) + wire::kSyncOccurrence, 1);
      wire::write_u32(report, completion(wave) + wire::kSyncParticipants, 2);
      wire::write_u32(report, completion(wave) + wire::kSyncSequence, wave == 0 ? 1 : 0);
    }
  }

  void many_barrier_sites(uint32_t count, uint32_t split, uint32_t waves = 2) {
    test::manifest(capture_.header).barriers.clear();
    test::manifest(capture_.header).scope = ScopeKind::NativeLdsNoSynchronization;
    many_sites(count);
    Manifest &facts = test::manifest(capture_.header);
    facts.target = 1250;
    facts.scope = ScopeKind::NativeLdsWorkgroupBarrier;
    for (uint32_t site = split; site < count; ++site) {
      facts.sites[site].original_offset += 16;
      facts.sites[site].prepared_guest_offset += 128;
      facts.sites[site].probe_offset += 128;
    }
    facts.barriers = {BarrierFacts{.original_drain = uint64_t{split} * 8,
                                   .original_signal = uint64_t{split} * 8 + 4,
                                   .original_wait = uint64_t{split} * 8 + 8,
                                   .prepared_signal = uint64_t{split} * 128,
                                   .prepared_wait = uint64_t{split} * 128 + 4,
                                   .completion_probe = uint64_t{split} * 128 + 8,
                                   .probe_size = 64}};
    capture_.executions.clear();
    append_execution({{32 * waves, 1, 1}, {32 * waves, 1, 1}});
    std::vector<uint8_t> &report = capture_.executions.front().report;
    for (uint32_t wave = 0; wave < waves; ++wave) {
      wire::write_u32(report, row(wave) + wire::kRowFinalPhase, 1);
      wire::write_u32(report, completion(wave) + wire::kSyncPhase, 1);
      wire::write_u32(report, completion(wave) + wire::kSyncCommit, wire::kBarrierCompleted);
      wire::write_u32(report, completion(wave) + wire::kSyncOccurrence, 1);
      wire::write_u32(report, completion(wave) + wire::kSyncParticipants, waves);
      wire::write_u32(report, completion(wave) + wire::kSyncSequence, split);
    }
  }

  void workgroups(uint32_t count = 4, uint32_t target = 1250, uint32_t wave_size = 32) {
    test::manifest(capture_.header).target = target;
    test::manifest(capture_.header).wave_size = wave_size;
    test::manifest(capture_.header).entry.workgroup_id_scalar =
        target == 1201 || target == 1250 ? 117 : 15;
    capture_.executions.clear();
    const uint32_t threads = 2 * wave_size + 1;
    append_execution({{threads * count, 1, 1}, {threads, 1, 1}});
  }

  void workgroup_barriers() {
    barrier();
    const FailureOr<LaunchLayout> layout =
        plan_launch(test::manifest(capture_.header), capture_.executions[0].dimensions,
                    capture_.executions[0].group_bytes, 0);
    ASSERT_TRUE(layout.succeeded());
    for (uint32_t index = 0; index < layout.value().wave_count; ++index) {
      std::vector<uint8_t> &report = capture_.executions[0].report;
      wire::write_u32(report, row(index) + wire::kRowFinalPhase, 1);
      wire::write_u32(report, completion(index) + wire::kSyncPhase, 1);
      wire::write_u32(report, completion(index) + wire::kSyncCommit, wire::kBarrierCompleted);
      wire::write_u32(report, completion(index) + wire::kSyncGroup, 0);
      wire::write_u32(report, completion(index) + wire::kSyncOccurrence, 1);
      wire::write_u32(report, completion(index) + wire::kSyncParticipants,
                      layout.value().waves_per_workgroup);
      wire::write_u32(report, completion(index) + wire::kSyncSequence,
                      index % layout.value().waves_per_workgroup == 0 ? 1 : 0);
    }
  }

  Capture capture_;
  util::StringDiagnostic error_;
  uint32_t stride_ = 0;
};

TEST_F(ConSanAnalysis, DualRangesPreserveTheGapAndAttributeSecondOrdinalConflicts) {
  Manifest &manifest = test::manifest(capture_.header);
  manifest.sites[0].width = 8;
  manifest.sites[0].lds_ranges = {.shape = LdsRangeShape::Dual, .offset0 = 3, .offset1 = 5};
  manifest.sites[1].width = 8;
  for (const uint32_t peer : {24u, 32u, 40u, 48u}) {
    SCOPED_TRACE(peer);
    capture_.executions.clear();
    append_execution();
    access(0, 0, 1, 0);
    access(1, 1, 1, peer);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_TRUE(result.value().has_dual_ranges);
    EXPECT_TRUE(execution.coverage_complete);
    EXPECT_EQ(execution.valid_observations, 3u);
    EXPECT_EQ(execution.cross_wave_pairs, peer == 24 || peer == 40 ? 1u : 0u);
    if (!execution.cross_wave_examples.empty()) {
      const LdsRaceFinding &finding = execution.cross_wave_examples.front();
      const LdsRaceParticipant &dual = finding.first.site == 0 ? finding.first : finding.second;
      EXPECT_EQ(dual.range_id, peer == 24 ? 0u : 1u);
      EXPECT_EQ(dual.offset, peer);
    }
  }
}

TEST_F(ConSanAnalysis, DualStoreCollisionsIncludeDifferentOrdinalsOfOneOccurrence) {
  test::manifest(capture_.header).sites[0].lds_ranges = {
      .shape = LdsRangeShape::Dual, .offset0 = 3, .offset1 = 5};
  access(0, 0, 1, 0, 1);
  access(0, 0, 1, 8, 2, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_TRUE(execution.coverage_complete);
  EXPECT_EQ(execution.valid_observations, 4u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_EQ(execution.collision_lane_pairs, 1u);
  ASSERT_EQ(execution.collision_examples.size(), 1u);
  const LdsRaceFinding &finding = execution.collision_examples.front();
  EXPECT_EQ(finding.first.offset, 20u);
  EXPECT_EQ(finding.second.offset, 20u);
  EXPECT_EQ(finding.first.range_id, 1u);
  EXPECT_EQ(finding.second.range_id, 0u);
  EXPECT_EQ(finding.first.mask, 1u);
  EXPECT_EQ(finding.second.mask, 2u);
  const std::string rendered = render_lds_race_analysis(result.value());
  EXPECT_NE(rendered.find("\"scope\":\"native_lds_multi_range_"), std::string::npos);
  EXPECT_NE(rendered.find("\"range_id\":1"), std::string::npos);
  EXPECT_NE(rendered.find("\"second\":"), std::string::npos);
}

TEST_F(ConSanAnalysis, DualStrideOffsetsDoNotTruncateOrInvalidateTheOtherValidRange) {
  Manifest &manifest = test::manifest(capture_.header);
  manifest.sites[0].width = 8;
  manifest.sites[0].lds_ranges = {
      .shape = LdsRangeShape::DualStride64, .offset0 = 0, .offset1 = 255};
  EXPECT_EQ(manifest.sites[0].lds_ranges.byte_offset(1, 8), 130560u);
  access(0, 0, 1, 16);
  access(1, 1, 1, 16);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_EQ(execution.rejected_records, 1u);
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_EQ(execution.cross_wave_pairs, 1u);
}

TEST_F(ConSanAnalysis, DualCoincidentOffsetsFollowTheConcreteTargetSemantics) {
  Manifest &manifest = test::manifest(capture_.header);
  for (const uint32_t target : {950u, 1100u}) {
    SCOPED_TRACE(target);
    manifest.target = target;
    manifest.wave_size = 64;
    manifest.sites[0].lds_ranges = {.shape = LdsRangeShape::Dual, .offset0 = 3, .offset1 = 3};
    capture_.executions.clear();
    append_execution({{128, 1, 1}, {128, 1, 1}});
    access(0, 0, 1, 0);
    access(1, 1, 1, 12);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_EQ(result.value().cross_wave_pairs, target == 950 ? 1u : 2u);
    EXPECT_EQ(result.value().collision_groups, 0u);
    EXPECT_TRUE(result.value().coverage_complete);
  }
}

TEST_F(ConSanAnalysis, DualCoincidentReadsCountConcreteRequestsAndAttributeRetainedOrdinals) {
  Manifest &manifest = test::manifest(capture_.header);
  for (const uint32_t target : {942u, 950u, 1100u}) {
    for (const uint32_t width : {4u, 8u}) {
      SCOPED_TRACE(::testing::Message() << "target=" << target << " width=" << width);
      manifest.target = target;
      manifest.wave_size = 64;
      manifest.sites[0].access = AccessKind::Read;
      manifest.sites[0].width = width;
      manifest.sites[0].lds_ranges = {.shape = LdsRangeShape::Dual, .offset0 = 3, .offset1 = 3};
      manifest.sites[1].access = AccessKind::Write;
      manifest.sites[1].width = width;
      capture_.executions.clear();
      append_execution({{128, 1, 1}, {128, 1, 1}});
      access(0, 0, 1, 0);
      access(1, 1, 1, 3 * width);
      const FailureOr<LdsRaceResult> result = analyze();
      ASSERT_TRUE(result.succeeded()) << error_.message();
      const LdsRaceExecutionResult &execution = result.value().executions.front();
      // CDNA3/4 suppress the coincident second read request; RDNA3 retains it.
      // Keep these expectations independent of the policy used by the analyzer.
      const uint64_t expected_pairs = target == 1100 ? 2u : 1u;
      const uint32_t expected_range_mask = target == 1100 ? 0b11u : 0b01u;
      EXPECT_TRUE(result.value().has_dual_ranges);
      EXPECT_TRUE(result.value().coverage_complete);
      EXPECT_TRUE(execution.coverage_complete);
      EXPECT_EQ(execution.valid_observations, expected_pairs + 1);
      EXPECT_EQ(execution.rejected_records, 0u);
      EXPECT_EQ(execution.integrity_events, 0u);
      EXPECT_EQ(execution.cross_wave_pairs, expected_pairs);
      EXPECT_EQ(execution.cross_wave_lane_pairs, expected_pairs);
      EXPECT_EQ(execution.collision_groups, 0u);
      EXPECT_EQ(execution.cross_wave_examples.size(), expected_pairs);
      uint32_t read_range_mask = 0;
      for (const LdsRaceFinding &finding : execution.cross_wave_examples) {
        const LdsRaceParticipant &read = finding.first.site == 0 ? finding.first : finding.second;
        const LdsRaceParticipant &write = finding.first.site == 0 ? finding.second : finding.first;
        EXPECT_EQ(read.site, 0u);
        EXPECT_EQ(read.access, AccessKind::Read);
        EXPECT_EQ(read.wave, 0u);
        EXPECT_EQ(read.offset, 3 * width);
        EXPECT_EQ(read.width, width);
        EXPECT_LE(read.range_id, 1u);
        if (read.range_id <= 1)
          read_range_mask |= 1u << read.range_id;
        EXPECT_EQ(write.site, 1u);
        EXPECT_EQ(write.access, AccessKind::Write);
        EXPECT_EQ(write.wave, 1u);
        EXPECT_EQ(write.offset, 3 * width);
        EXPECT_EQ(write.width, width);
        EXPECT_EQ(write.range_id, 0u);
        EXPECT_EQ(finding.lane_pairs, 1u);
      }
      EXPECT_EQ(read_range_mask, expected_range_mask);
    }
  }
}

TEST_F(ConSanAnalysis, DecoderControlUnequalLoopFinalPhasesCannotConfirmAnUnorderedPair) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  test::manifest(capture_.header).scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  std::vector<uint8_t> &report = capture_.executions[0].report;
  // Both endpoints precede the first barrier. One row claims a zero-iteration
  // exit. Its absence of a completion cannot establish absence of ordering.
  wire::write_u32(report, record(1, 1) + wire::kRecordPhase, 0);
  wire::write_u32(report, row(1) + wire::kRowFinalPhase, 0);
  std::fill_n(report.begin() + completion(1), kSyncRecordBytes, 0);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().ordering_unresolved_pairs, 1u);
  EXPECT_EQ(result.value().executions[0].integrity_events, 0u);
  EXPECT_FALSE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, DecoderControlLoopQuotaLossKeepsIndependentWitnessWithoutIntegrityError) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  barrier();
  test::manifest(capture_.header).scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, kSyncCapacity + 1);
    wire::write_u32(report, row(wave) + wire::kRowSyncLoss, 1);
    wire::write_u32(report, row(wave) + wire::kRowLoss, 5);
    wire::write_u32(report, row(wave) + wire::kRowFinalSequence, 6);
    wire::write_u32(report, row(wave) + wire::kRowStatus, 7);
    for (uint32_t slot = 1; slot < kSyncCapacity; ++slot) {
      std::copy_n(report.begin() + completion(wave), kSyncRecordBytes,
                  report.begin() + completion(wave, slot));
      wire::write_u32(report, completion(wave, slot) + wire::kSyncOccurrence, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncPhase, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncSequence, 6);
    }
  }
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions[0];
  EXPECT_EQ(execution.ordered_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_EQ(execution.lost_occurrences, 10u);
  EXPECT_EQ(execution.synchronization_loss, 2u);
  EXPECT_EQ(execution.integrity_events, 0u);
  EXPECT_FALSE(execution.coverage_complete);
}

TEST_F(ConSanAnalysis, SaturatedAccessPhasesAreCoverageLossWithoutInventingOrdering) {
  barrier();
  test::manifest(capture_.header).scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  test::manifest(capture_.header).sites[1].access = AccessKind::Write;
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    access(wave, 1, 1, 0, 3);
    wire::write_u32(report, record(wave, 1) + wire::kRecordPhase, UINT32_MAX);
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, UINT32_MAX);
    wire::write_u32(report, row(wave) + wire::kRowSyncLoss, UINT32_MAX - kSyncCapacity);
    wire::write_u32(report, row(wave) + wire::kRowStatus,
                    wire::kRowEntered | wire::kRowExited | wire::kRowExhausted);
    for (uint32_t slot = 0; slot < kSyncCapacity; ++slot) {
      wire::write_u32(report, completion(wave, slot) + wire::kSyncCommit, wire::kBarrierCompleted);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncGroup, 0);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncOccurrence, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncPhase, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncSequence, 0);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncParticipants, 2);
    }
  }
  // First reaching UINT32_MAX is exact. Further completions saturate the phase
  // and set exhaustion, while a site's first active access can still publish.
  for (const uint32_t exhausted : {0u, 1u}) {
    SCOPED_TRACE(exhausted);
    for (uint32_t wave = 0; wave < 2; ++wave) {
      wire::write_u32(report, row(wave) + wire::kRowCounterExhausted, exhausted);
      wire::write_u32(report, row(wave) + wire::kRowSyncLoss,
                      UINT32_MAX - kSyncCapacity + exhausted);
    }
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions[0];
    EXPECT_EQ(execution.valid_observations, 2u);
    EXPECT_EQ(execution.integrity_events, 0u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.collision_groups, 2u);
    EXPECT_FALSE(execution.coverage_complete);
  }
  // Saturation does not excuse an access beyond its row's final phase.
  wire::write_u32(report, row(0) + wire::kRowFinalPhase, UINT32_MAX - 1);
  const FailureOr<LdsRaceResult> contradictory = analyze();
  ASSERT_TRUE(contradictory.succeeded()) << error_.message();
  const LdsRaceExecutionResult &invalid = contradictory.value().executions[0];
  // Only the access contradicts its own row; unequal wave-local final phases
  // independently leave ordering coverage incomplete.
  EXPECT_EQ(invalid.integrity_events, 1u);
  EXPECT_FALSE(invalid.coverage_complete);
  for (const char *reason :
       {"access_phase_contradicts_qualified_program", "mismatched_workgroup_final_phases"}) {
    EXPECT_TRUE(std::ranges::any_of(invalid.issues, [reason](const LdsRaceIntegrityIssue &issue) {
      return issue.reason == reason;
    }));
  }
}

TEST_F(ConSanAnalysis, SaturatedPhaseResidueIsUnknownOnlyAfterCounterExhaustion) {
  barrier();
  Manifest &facts = test::manifest(capture_.header);
  facts.scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  facts.sites[1].access = AccessKind::Write;
  facts.barriers.push_back({.original_drain = 40,
                            .original_signal = 44,
                            .original_wait = 48,
                            .prepared_signal = 576,
                            .prepared_wait = 580,
                            .completion_probe = 584,
                            .probe_size = 64});
  facts.barriers.push_back({.original_drain = 56,
                            .original_signal = 60,
                            .original_wait = 64,
                            .prepared_signal = 656,
                            .prepared_wait = 660,
                            .completion_probe = 664,
                            .probe_size = 64});
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    access(wave, 1, 1, 0, 3);
    wire::write_u32(report, record(wave, 1) + wire::kRecordPhase, UINT32_MAX);
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, UINT32_MAX);
    wire::write_u32(report, row(wave) + wire::kRowStatus,
                    wire::kRowEntered | wire::kRowExited | wire::kRowExhausted);
    for (uint32_t slot = 0; slot < kSyncCapacity; ++slot) {
      wire::write_u32(report, completion(wave, slot) + wire::kSyncCommit, wire::kBarrierCompleted);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncGroup, slot % 3);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncOccurrence, slot / 3 + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncPhase, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncSequence, 0);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncParticipants, 2);
    }
  }
  // UINT32_MAX closes a three-barrier cycle, but site 1 lies at rank one.
  // The residue is contradictory while exact, and unknowable after saturation.
  for (const uint32_t exhausted : {0u, 1u}) {
    SCOPED_TRACE(exhausted);
    for (uint32_t wave = 0; wave < 2; ++wave) {
      wire::write_u32(report, row(wave) + wire::kRowCounterExhausted, exhausted);
      // A further complete cycle can contain the rank-one access while all
      // three additional completions leave the stored phase saturated.
      wire::write_u32(report, row(wave) + wire::kRowSyncLoss,
                      UINT32_MAX - kSyncCapacity + 3 * exhausted);
    }
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions[0];
    EXPECT_EQ(execution.valid_observations, 2u);
    EXPECT_EQ(execution.integrity_events, exhausted ? 0u : 2u);
    EXPECT_EQ(std::ranges::count_if(execution.issues,
                                    [](const LdsRaceIntegrityIssue &issue) {
                                      return issue.reason ==
                                             "access_phase_contradicts_qualified_program";
                                    }),
              exhausted ? 0u : 2u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.collision_groups, 2u);
    EXPECT_FALSE(execution.coverage_complete);
  }
}

TEST_F(ConSanAnalysis, DecoderControlSeededCounterLimitsKeepCandidatesBoundedAndCollisionsValid) {
  // Both sites are actually before the barrier in the qualified static order.
  many_barrier_sites(2, 2);
  test::manifest(capture_.header).sites[1].access = AccessKind::Read;
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  test::manifest(capture_.header).scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, UINT32_MAX);
    wire::write_u32(report, row(wave) + wire::kRowFinalSequence, UINT32_MAX);
    wire::write_u32(report, row(wave) + wire::kRowLoss, UINT32_MAX);
    wire::write_u32(report, row(wave) + wire::kRowSyncLoss, UINT32_MAX);
    wire::write_u32(report, row(wave) + wire::kRowCounterExhausted, 1);
    wire::write_u32(report, row(wave) + wire::kRowStatus, 7);
    wire::write_u32(report, record(wave, wave) + wire::kRecordPhase, 0);
    wire::write_u32(report, completion(wave) + wire::kSyncSequence, 1);
  }
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions[0];
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_EQ(execution.counter_exhausted_waves, 2u);
  // Both accesses precede the complete first barrier. Later exhaustion does not
  // erase that closed interval, even though the remaining journal has gaps.
  EXPECT_EQ(execution.cross_wave_pairs, 1u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_NE(render_lds_race_analysis(result.value()).find("\"dropped_representatives\":null"),
            std::string::npos);
  // Losing the required first completion does make this same pair unresolved;
  // the independent exact-mask collision remains confirmed in both cases.
  wire::write_u32(report, completion(1) + wire::kSyncCommit, 0);
  const FailureOr<LdsRaceResult> missing = analyze();
  ASSERT_TRUE(missing.succeeded()) << error_.message();
  EXPECT_EQ(missing.value().cross_wave_pairs, 0u);
  EXPECT_EQ(missing.value().ordering_unresolved_pairs, 1u);
  EXPECT_EQ(missing.value().collision_groups, 1u);
}

TEST_F(ConSanAnalysis, OverlappingWriteReadProducesAttributedUnorderedCrossWaveFinding) {
  access(0, 0, 1);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_TRUE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().collision_groups, 0u);
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_EQ(execution.cross_wave_lane_pairs, 1u);
  ASSERT_EQ(execution.cross_wave_examples.size(), 1u);
  const LdsRaceFinding &finding = execution.cross_wave_examples.front();
  EXPECT_EQ(finding.first.wave, 0u);
  EXPECT_EQ(finding.first.site, 0u);
  EXPECT_EQ(finding.first.original_offset, 0u);
  EXPECT_EQ(finding.second.wave, 1u);
  EXPECT_EQ(finding.second.site, 1u);
  EXPECT_EQ(finding.second.original_offset, 8u);
  EXPECT_EQ(finding.first.mask, 1u);
  EXPECT_FALSE(execution.ordering_uncertain);
}

TEST_F(ConSanAnalysis, UnequalStartMixedWidthRangesOverlapAndAdjacentRangesDoNot) {
  test::manifest(capture_.header).target = 1250;
  test::manifest(capture_.header).sites[0].width = 8;
  for (const AccessKind wide_access : {AccessKind::Write, AccessKind::Read}) {
    for (const AccessKind narrow_access : {AccessKind::Read, AccessKind::Write}) {
      for (const uint32_t offset : {4u, 8u}) {
        SCOPED_TRACE(::testing::Message()
                     << "wide_access=" << static_cast<uint32_t>(wide_access) << ", narrow_access="
                     << static_cast<uint32_t>(narrow_access) << ", narrow_offset=" << offset);
        test::manifest(capture_.header).sites[0].access = wide_access;
        test::manifest(capture_.header).sites[1].access = narrow_access;
        capture_.executions.clear();
        append_execution();
        access(0, 0, 1, 0);
        access(1, 1, 1, offset);
        const FailureOr<LdsRaceResult> result = analyze();
        ASSERT_TRUE(result.succeeded()) << error_.message();
        const bool conflict =
            offset == 4 && (wide_access == AccessKind::Write || narrow_access == AccessKind::Write);
        EXPECT_EQ(result.value().executions[0].valid_observations, 2u);
        EXPECT_EQ(result.value().cross_wave_pairs, conflict ? 1u : 0u);
        EXPECT_TRUE(result.value().coverage_complete);
      }
    }
  }
}

TEST_F(ConSanAnalysis, AtomicOverlapUsesOrdinaryPeersAndRetainsTheSavedAccessRole) {
  struct PeerRange {
    uint32_t offset, width;
    bool overlaps;
  };
  // The atomic covers [4, 8). Exercise the equal-range bucket and both sides
  // of the range sweep, including an adjacent ordinary access.
  for (const AccessKind ordinary : {AccessKind::Read, AccessKind::Write})
    for (const PeerRange peer : {PeerRange{4, 4, true}, {0, 8, true}, {6, 2, true}, {8, 4, false}})
      for (const uint32_t atomic_site : {0u, 1u}) {
        SCOPED_TRACE(::testing::Message()
                     << "atomic site " << atomic_site << " peer " << uint32_t(ordinary)
                     << " offset " << peer.offset << " width " << peer.width);
        Manifest &manifest = test::manifest(capture_.header);
        manifest.sites[atomic_site].access = AccessKind::Atomic;
        manifest.sites[atomic_site].width = 4;
        const uint32_t ordinary_site = 1 - atomic_site;
        manifest.sites[ordinary_site].access = ordinary;
        manifest.sites[ordinary_site].width = peer.width;
        workgroups(1, 950, 64);
        access(atomic_site, atomic_site, 1, 4, uint64_t{3} << 62);
        access(ordinary_site, ordinary_site, 1, peer.offset);
        const FailureOr<LdsRaceResult> result = analyze();
        ASSERT_TRUE(result.succeeded()) << error_.message();
        EXPECT_TRUE(result.value().coverage_complete);
        const LdsRaceExecutionResult &execution = result.value().executions.front();
        EXPECT_EQ(execution.valid_observations, 2u);
        EXPECT_EQ(execution.cross_wave_pairs, peer.overlaps ? 1u : 0u);
        EXPECT_EQ(execution.cross_wave_lane_pairs, peer.overlaps ? 2u : 0u);
        EXPECT_EQ(execution.collision_groups, 0u);
        if (peer.overlaps) {
          ASSERT_EQ(execution.cross_wave_examples.size(), 1u);
          const LdsRaceFinding &finding = execution.cross_wave_examples.front();
          const LdsRaceParticipant &atomic =
              finding.first.site == atomic_site ? finding.first : finding.second;
          EXPECT_EQ(atomic.access, AccessKind::Atomic);
          EXPECT_EQ(atomic.offset, 4u);
          EXPECT_EQ(atomic.mask, uint64_t{3} << 62);
          EXPECT_NE(render_analysis(result.value()).find("\"access\":\"atomic\""),
                    std::string::npos);
        }
      }
}

TEST_F(ConSanAnalysis, AtomicPeersAndSameWaveOrdinaryAccessDoNotFormCandidates) {
  Manifest &manifest = test::manifest(capture_.header);
  for (auto &site : manifest.sites)
    site.access = AccessKind::Atomic;
  workgroups(1, 950, 64);
  access(0, 0, 1, 4, 3);
  access(1, 1, 1, 4, 3);
  auto result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_TRUE(result.value().coverage_complete);
  EXPECT_EQ(result.value().executions.front().valid_observations, 2u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().collision_groups, 0u);

  manifest.sites[1].access = AccessKind::Write;
  workgroups(1, 950, 64);
  access(0, 0, 1, 4, 3);
  access(0, 1, 2, 4);
  result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_TRUE(result.value().coverage_complete);
  EXPECT_EQ(result.value().executions.front().valid_observations, 2u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().collision_groups, 0u);
}

TEST_F(ConSanAnalysis, QwordAtomicRequiresEightValidBytesAndOverlapsTheHighDword) {
  Manifest &manifest = test::manifest(capture_.header);
  manifest.sites.front().access = AccessKind::Atomic;
  manifest.sites.front().width = 8;
  // [120,128) is the last aligned qword inside this 128-byte LDS extent.
  for (const uint32_t offset : {120u, 116u, 128u}) {
    workgroups(1, 950, 64);
    access(0, 0, 1, offset, 3);
    access(1, 1, 1, 124);
    const auto result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const auto &execution = result.value().executions.front();
    const bool valid = offset == 120;
    EXPECT_EQ(execution.coverage_complete, valid);
    EXPECT_EQ(execution.valid_observations, valid ? 2u : 1u);
    EXPECT_EQ(execution.rejected_records, valid ? 0u : 1u);
    EXPECT_EQ(execution.cross_wave_pairs, valid ? 1u : 0u);
    EXPECT_EQ(execution.collision_groups, 0u);
  }
}

TEST_F(ConSanAnalysis, OrdinaryLdsProfileReplaysByteAccessesAndRejectsAtomicFacts) {
  Manifest &manifest = test::manifest(capture_.header);
  manifest.options = {
      0x0c, 0x9d, 0x4a, 0x7a, 0x78, 0x6b, 0xd8, 0x4a, 0xe6, 0x60, 0xc3,
      0xff, 0xa2, 0xcf, 0xe2, 0xf3, 0x2b, 0xb4, 0x29, 0xc7, 0x23, 0x4b,
      0x13, 0x17, 0x87, 0x26, 0x0f, 0x4d, 0xfa, 0x19, 0xe6, 0x15,
  };
  manifest.sites[0].width = 1;
  workgroups(1, 950, 64);
  access(0, 0, 1);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  manifest.sites[1].access = AccessKind::Atomic;
  EXPECT_TRUE(validate_manifest(manifest).failed());
}

TEST_F(ConSanAnalysis, ThreeDwordAccessRequiresNativeAlignmentAndCoversOnlyTwelveBytes) {
  test::manifest(capture_.header).sites[0].width = 12;
  for (uint32_t start : {12u, 16u}) {
    for (uint32_t peer : {24u, 28u, 32u}) {
      SCOPED_TRACE(::testing::Message() << start << ", " << peer);
      capture_.executions.clear();
      append_execution();
      access(0, 0, 1, start);
      access(1, 1, 1, peer);
      const FailureOr<LdsRaceResult> result = analyze();
      ASSERT_TRUE(result.succeeded()) << error_.message();
      EXPECT_EQ(result.value().coverage_complete, start == 16);
      EXPECT_EQ(result.value().executions[0].valid_observations, start == 16 ? 2u : 1u);
      EXPECT_EQ(result.value().cross_wave_pairs, start == 16 && peer == 24 ? 1u : 0u);
    }
  }
}

TEST_F(ConSanAnalysis, TransposeFactsRequireTheTargetReadWidthAndNoEncodedDualOffsets) {
  for (uint32_t target : {950u, 1250u}) {
    set_transpose_site(target);
    Manifest &facts = test::manifest(capture_.header);
    SiteFacts &site = facts.sites[1];
    for (const auto &[shape, width] : std::array<std::pair<LdsRangeShape, uint32_t>, 4>{
             {{LdsRangeShape::Transpose4, 8},
              {LdsRangeShape::Transpose6, 12},
              {LdsRangeShape::Transpose8, 8},
              {LdsRangeShape::Transpose16, target == 950 ? 8u : 16u}}}) {
      site.lds_ranges = {.shape = shape};
      site.width = width;
      const auto bytes = encode_manifest(facts);
      ASSERT_TRUE(bytes.succeeded());
      const auto decoded = decode_manifest(bytes.value());
      ASSERT_TRUE(decoded.succeeded());
      EXPECT_EQ(decoded.value().sites[1], site);
      site.width = 4;
      EXPECT_TRUE(validate_manifest(facts).failed());
    }
    site.width = target == 950 ? 8 : 16;
    site.lds_ranges.offset0 = 1;
    EXPECT_TRUE(validate_manifest(facts).failed());
    site.lds_ranges.offset0 = 0;
    site.lds_ranges.offset1 = 1;
    EXPECT_TRUE(validate_manifest(facts).failed());
    site.lds_ranges.offset1 = 0;
    site.access = AccessKind::Write;
    EXPECT_TRUE(validate_manifest(facts).failed());
    site.access = AccessKind::Read;
    facts.target = target == 950 ? 942 : 1201;
    EXPECT_TRUE(validate_manifest(facts).failed());
  }
}

TEST_F(ConSanAnalysis, PhysicalTransposeLanesAndGuestExecRemainDistinctInAShortWave) {
  set_transpose_site(1250);
  append_execution({{4, 1, 1}, {4, 1, 1}});
  transpose_access(UINT32_MAX, 0);
  const auto complete = analyze();
  ASSERT_TRUE(complete.succeeded()) << error_.message();
  EXPECT_TRUE(complete.value().coverage_complete);
  EXPECT_EQ(complete.value().executions[0].valid_observations, 1u);
  EXPECT_NE(render_analysis(complete.value()).find("transpose_no_sync"), std::string::npos);
  auto &report = capture_.executions[0].report;
  for (const auto &[field, value] :
       std::array<std::pair<uint32_t, uint64_t>, 5>{{{wire::kRecordGuestMask, 15},
                                                     {wire::kRecordGuestMask, uint64_t{1} << 32},
                                                     {wire::kRecordAnchorExec, uint64_t{1} << 32},
                                                     {wire::kRecordFlags, wire::kExactGuestMask},
                                                     {wire::kRecordAnchorExec + 8, 1}}}) {
    SCOPED_TRACE(::testing::Message() << field << ", " << value);
    const auto saved = report;
    wire::write_u64(report, record(0, 1) + field, value);
    const auto malformed = analyze();
    ASSERT_TRUE(malformed.succeeded()) << error_.message();
    EXPECT_FALSE(malformed.value().coverage_complete);
    EXPECT_EQ(malformed.value().executions[0].rejected_records, 1u);
    EXPECT_EQ(malformed.value().executions[0].valid_observations, 0u);
    report = saved;
  }
  access(0, 0, 1, 0, 16);
  const auto ordinary = analyze();
  ASSERT_TRUE(ordinary.succeeded());
  EXPECT_FALSE(ordinary.value().coverage_complete);
  EXPECT_EQ(ordinary.value().executions[0].rejected_records, 1u);
}

TEST_F(ConSanAnalysis, DisagreeingTransposeExecAcrossSourceGroupsCannotProduceAWitness) {
  set_transpose_site(1250);
  append_execution();
  access(1, 0, 1);
  transpose_access(1, 0);
  transpose_access(UINT32_MAX - 1, 1, 1, 8);
  const auto malformed = analyze();
  ASSERT_TRUE(malformed.succeeded()) << error_.message();
  EXPECT_EQ(malformed.value().executions[0].rejected_records, 2u);
  EXPECT_EQ(malformed.value().executions[0].valid_observations, 1u);
  EXPECT_EQ(malformed.value().cross_wave_pairs, 0u);
}

TEST_F(ConSanAnalysis, Cdna4TransposeRequiresFullExecAndOrdinarySlotsKeepReservedWordsZero) {
  set_transpose_site(950);
  append_execution();
  transpose_access(UINT64_MAX, UINT64_MAX);
  const auto complete = analyze();
  ASSERT_TRUE(complete.succeeded());
  EXPECT_TRUE(complete.value().coverage_complete);
  auto &report = capture_.executions[0].report;
  for (const uint64_t bad : std::array<uint64_t, 2>{0, 1}) {
    wire::write_u64(report, record(0, 1) + wire::kRecordAnchorExec, bad);
    const auto malformed = analyze();
    ASSERT_TRUE(malformed.succeeded());
    EXPECT_EQ(malformed.value().executions[0].rejected_records, 1u);
    EXPECT_EQ(malformed.value().executions[0].valid_observations, 0u);
  }
  test::manifest(capture_.header).sites[1].lds_ranges = {};
  access(0, 1, 1, 0, 1);
  const auto reserved = analyze();
  ASSERT_TRUE(reserved.succeeded());
  EXPECT_EQ(reserved.value().executions[0].rejected_records, 1u);
}

TEST_F(ConSanAnalysis, DualBaseGroupsRetainTheSharedStoreOccurrenceAndBothOrdinals) {
  test::manifest(capture_.header).sites[0].lds_ranges = {LdsRangeShape::Dual, 3, 5};
  access(0, 0, 1, 0, 1);
  access(0, 0, 1, 8, 2, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_TRUE(execution.coverage_complete);
  EXPECT_EQ(execution.valid_observations, 4u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_EQ(execution.collision_lane_pairs, 1u);
  ASSERT_EQ(execution.collision_examples.size(), 1u);
  const LdsRaceFinding &finding = execution.collision_examples.front();
  EXPECT_EQ(finding.first.offset, 20u);
  EXPECT_EQ(finding.second.offset, 20u);
  EXPECT_EQ(finding.first.range_id, 1u);
  EXPECT_EQ(finding.second.range_id, 0u);
  EXPECT_EQ(finding.first.mask, 1u);
  EXPECT_EQ(finding.second.mask, 2u);
}

TEST_F(ConSanAnalysis, InvalidSecondRangeRetainsTheFirstValidConflict) {
  test::manifest(capture_.header).sites[0].lds_ranges = {LdsRangeShape::DualStride64, 0, 255};
  access(0, 0, 1, 16);
  access(1, 1, 1, 16);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_EQ(execution.rejected_records, 1u);
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_EQ(execution.cross_wave_pairs, 1u);
}

TEST_F(ConSanAnalysis, CoincidentOffsetsFollowTargetRequestSemantics) {
  Manifest &manifest = test::manifest(capture_.header);
  for (uint32_t target : {950u, 1100u}) {
    manifest.target = target;
    manifest.wave_size = 64;
    manifest.sites[0].lds_ranges = {LdsRangeShape::Dual, 3, 3};
    capture_.executions.clear();
    append_execution({{128, 1, 1}, {128, 1, 1}});
    access(0, 0, 1, 0);
    access(1, 1, 1, 12);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_EQ(result.value().cross_wave_pairs, target == 950 ? 1u : 2u);
    EXPECT_EQ(result.value().collision_groups, 0u);
  }
}

TEST_F(ConSanAnalysis, EarlierThreeDwordProfileCannotAdvertiseByteSites) {
  // The selection identity emitted before whole-register byte loads were admitted.
  Manifest &manifest = test::manifest(capture_.header);
  manifest.options = {
      0x0f, 0xd8, 0x00, 0xf0, 0xff, 0xf8, 0x77, 0xb0, 0xf0, 0x51, 0x04,
      0x56, 0xfe, 0x42, 0x8d, 0x4c, 0xab, 0x4e, 0xe3, 0x50, 0x3b, 0xb6,
      0x11, 0xe3, 0x11, 0xda, 0xd5, 0x94, 0x33, 0x0b, 0xd5, 0xf5,
  };
  manifest.sites[0].width = 12;
  EXPECT_TRUE(validate_manifest(manifest).succeeded());
  manifest.sites[0].width = 1;
  EXPECT_TRUE(validate_manifest(manifest).failed());
}

TEST_F(ConSanAnalysis, DecoderControlWideBoundsCannotWrapOrSilentlyBecomeB32) {
  test::manifest(capture_.header).target = 1250;
  for (const uint32_t width : {2u, 8u, 12u, 16u}) {
    test::manifest(capture_.header).sites[0].width = width;
    for (const uint32_t offset : {1u, 128u - width + 1, UINT32_MAX - 7}) {
      capture_.executions.clear();
      append_execution();
      access(0, 0, 1, offset, 3);
      access(1, 1, 1);
      const FailureOr<LdsRaceResult> result = analyze();
      ASSERT_TRUE(result.succeeded()) << error_.message();
      EXPECT_EQ(result.value().executions[0].valid_observations, 1u);
      EXPECT_EQ(result.value().executions[0].rejected_records, 1u);
      EXPECT_EQ(result.value().cross_wave_pairs, 0u);
      EXPECT_EQ(result.value().collision_groups, 0u);
      EXPECT_FALSE(result.value().coverage_complete);
    }
  }
  for (const uint32_t width : {0u, 3u, 5u, 10u, 32u}) {
    test::manifest(capture_.header).sites[0].width = width;
    EXPECT_TRUE(validate_manifest(test::manifest(capture_.header)).failed());
  }
  test::manifest(capture_.header).sites[0].width = 8;
  test::manifest(capture_.header).target = 1201;
  EXPECT_TRUE(validate_manifest(test::manifest(capture_.header)).succeeded());
}

TEST_F(ConSanAnalysis, MixedWidthUnequalStartPairUsesTheQualifiedBarrierCertificate) {
  test::manifest(capture_.header).sites[0].width = 8;
  access(0, 0, 1, 0);
  access(1, 1, 1, 4);
  barrier();
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().executions[0].ordered_pairs, 1u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
}

TEST_F(ConSanAnalysis, IdenticalOffsetsInDifferentWorkgroupsNeverFormLdsCandidates) {
  workgroups();
  access(0, 0, 1);
  access(4, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().executions[0].valid_observations, 2u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_TRUE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, LocalConflictsAndReadReadExclusionApplyIndependentlyInEveryWorkgroup) {
  for (const bool read_read : {false, true}) {
    workgroups();
    test::manifest(capture_.header).sites[0].access =
        read_read ? AccessKind::Read : AccessKind::Write;
    for (uint32_t group = 0; group < 4; ++group) {
      access(3 * group, 0, 1);
      access(3 * group + 1, 1, 1);
    }
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_EQ(result.value().cross_wave_pairs, read_read ? 0u : 4u);
    EXPECT_EQ(result.value().executions[0].valid_observations, 8u);
    EXPECT_TRUE(result.value().coverage_complete);
    for (const LdsRaceFinding &finding : result.value().executions[0].cross_wave_examples) {
      EXPECT_EQ(finding.first.workgroup, finding.second.workgroup);
      EXPECT_EQ(finding.first.wave, 0u);
      EXPECT_EQ(finding.second.wave, 1u);
      EXPECT_EQ(finding.ordering, LdsRaceOrdering::Unordered);
    }
  }
}

TEST_F(ConSanAnalysis, EveryTargetKeepsNoSyncConflictsAndReadReadExclusionLocalToWorkgroups) {
  for (const TargetProfile &target : kNativeLdsTargets) {
    for (const uint32_t wave_size : {32u, 64u}) {
      if (!supports_wave(target, wave_size)) {
        continue;
      }
      SCOPED_TRACE(target.name);
      SCOPED_TRACE(wave_size);
      for (const bool read_read : {false, true}) {
        SCOPED_TRACE(read_read);
        ASSERT_NO_FATAL_FAILURE(workgroups(4, target.id, wave_size));
        test::manifest(capture_.header).sites[0].access =
            read_read ? AccessKind::Read : AccessKind::Write;
        for (uint32_t group = 0; group < 4; ++group) {
          access(3 * group, 0, 1);
          access(3 * group + 1, 1, 1);
          access(3 * group + 2, 1, 1);
        }
        const FailureOr<LdsRaceResult> result = analyze();
        ASSERT_TRUE(result.succeeded()) << error_.message();
        const LdsRaceExecutionResult &execution = result.value().executions.front();
        EXPECT_EQ(execution.workgroup_count, 4u);
        EXPECT_EQ(execution.waves_per_workgroup, 3u);
        EXPECT_EQ(execution.valid_observations, 12u);
        EXPECT_EQ(execution.cross_wave_pairs, read_read ? 0u : 8u);
        EXPECT_EQ(execution.ordered_pairs, 0u);
        EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
        EXPECT_EQ(execution.required_barrier_waves, 0u);
        EXPECT_TRUE(execution.coverage_complete);
        for (const LdsRaceFinding &finding : execution.cross_wave_examples) {
          EXPECT_EQ(finding.first.workgroup, finding.second.workgroup);
          EXPECT_EQ(finding.first.wave, 0u);
          EXPECT_TRUE(finding.second.wave == 1 || finding.second.wave == 2);
          EXPECT_EQ(finding.ordering, LdsRaceOrdering::Unordered);
        }
      }
    }
  }
}

TEST_F(ConSanAnalysis, EveryTargetNeverJoinsIdenticalLdsOffsetsAcrossWorkgroups) {
  for (const TargetProfile &target : kNativeLdsTargets) {
    for (const uint32_t wave_size : {32u, 64u}) {
      if (!supports_wave(target, wave_size)) {
        continue;
      }
      SCOPED_TRACE(target.name);
      SCOPED_TRACE(wave_size);
      ASSERT_NO_FATAL_FAILURE(workgroups(4, target.id, wave_size));
      access(0, 0, 1);
      for (uint32_t group = 1; group < 4; ++group) {
        access(3 * group + 1, 1, 1);
      }
      const FailureOr<LdsRaceResult> result = analyze();
      ASSERT_TRUE(result.succeeded()) << error_.message();
      const LdsRaceExecutionResult &execution = result.value().executions.front();
      EXPECT_EQ(execution.valid_observations, 4u);
      EXPECT_EQ(execution.cross_wave_pairs, 0u);
      EXPECT_EQ(execution.ordered_pairs, 0u);
      EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
      EXPECT_TRUE(execution.coverage_complete);
    }
  }
}

TEST_F(ConSanAnalysis, Wave64WorkgroupsKeepHighLanesAndRejectOnlyInvalidPartialWaveMasks) {
  for (const TargetProfile &target : kNativeLdsTargets) {
    if (!supports_wave(target, 64)) {
      continue;
    }
    SCOPED_TRACE(target.name);
    ASSERT_NO_FATAL_FAILURE(workgroups(4, target.id, 64));
    for (uint32_t group = 0; group < 4; ++group) {
      access(3 * group, 0, 1, 0, uint64_t{3} << 62);
      access(3 * group + 1, 1, 1, 0, uint64_t{1} << 63);
      access(3 * group + 2, 1, 1);
    }
    FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &complete = result.value().executions.front();
    EXPECT_EQ(complete.valid_observations, 12u);
    EXPECT_EQ(complete.cross_wave_pairs, 8u);
    EXPECT_EQ(complete.cross_wave_lane_pairs, 16u);
    EXPECT_EQ(complete.collision_groups, 4u);
    EXPECT_EQ(complete.collision_lane_pairs, 4u);
    EXPECT_TRUE(complete.coverage_complete);
    for (const LdsRaceFinding &finding : complete.cross_wave_examples) {
      EXPECT_EQ(finding.first.workgroup, finding.second.workgroup);
      EXPECT_EQ(finding.first.mask, uint64_t{3} << 62);
      EXPECT_EQ(finding.second.mask, finding.second.wave == 1 ? uint64_t{1} << 63 : 1u);
    }

    // WG2/W2 contains only lane zero; rejecting lane one must preserve other groups.
    access(8, 1, 1, 0, 2);
    result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &damaged = result.value().executions.front();
    EXPECT_EQ(damaged.valid_observations, 11u);
    EXPECT_EQ(damaged.rejected_records, 1u);
    EXPECT_EQ(damaged.cross_wave_pairs, 7u);
    EXPECT_EQ(damaged.cross_wave_lane_pairs, 14u);
    EXPECT_EQ(damaged.collision_groups, 4u);
    EXPECT_FALSE(damaged.coverage_complete);
    EXPECT_TRUE(std::ranges::any_of(damaged.issues, [](const LdsRaceIntegrityIssue &issue) {
      return issue.workgroup == 2 && issue.wave == 2 && issue.site == 1 &&
             issue.reason == "invalid_record_commit_identity_mask_or_sequence";
    }));
  }
}

TEST_F(ConSanAnalysis,
       DecoderControlMissingCertificateInOneWorkgroupKeepsOtherCertificatesAndCollisions) {
  workgroups();
  for (uint32_t group = 0; group < 4; ++group) {
    access(3 * group, 0, 1, 0, group == 0 ? 3 : 1);
    access(3 * group + 1, 1, 1);
  }
  workgroup_barriers();
  const std::vector<uint8_t> saved = capture_.executions[0].report;
  for (const uint32_t field :
       {static_cast<uint32_t>(sync_offset(test::manifest(capture_.header).sites.size()) +
                              wire::kSyncCommit),
        wire::kRowWorkgroupX}) {
    capture_.executions[0].report = saved;
    wire::write_u32(capture_.executions[0].report, row(8) + field, 0);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions[0];
    EXPECT_EQ(execution.valid_observations, 8u);
    EXPECT_EQ(execution.ordered_pairs, 3u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.collision_groups, 1u);
    EXPECT_FALSE(result.value().coverage_complete);
    ASSERT_EQ(execution.unresolved_examples.size(), 1u);
    EXPECT_EQ(execution.unresolved_examples[0].first.workgroup, 2u);
    EXPECT_EQ(execution.unresolved_examples[0].second.workgroup, 2u);
    EXPECT_EQ(execution.unresolved_examples[0].ordering, LdsRaceOrdering::Unresolved);
    for (const LdsRaceFinding &finding : execution.ordered_examples) {
      EXPECT_NE(finding.first.workgroup, 2u);
    }
  }
}

TEST_F(ConSanAnalysis, DecoderControlWrongWorkgroupRowIdentityDiscardsOnlyItsAccesses) {
  workgroups();
  for (uint32_t group = 0; group < 4; ++group) {
    access(3 * group, 0, 1);
    access(3 * group + 1, 1, 1);
  }
  wire::write_u32(capture_.executions[0].report, row(3) + wire::kRowWorkgroupX, 0);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().executions[0].valid_observations, 7u);
  EXPECT_EQ(result.value().cross_wave_pairs, 3u);
  EXPECT_FALSE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, MultiWorkgroupReusedStorageNeverJoinsDifferentDispatches) {
  workgroups();
  append_execution({{260, 1, 1}, {65, 1, 1}});
  access(9, 0, 1, 0, 1, 0, 0);
  access(10, 1, 1, 0, 1, 0, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().executions.size(), 2u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions[0].valid_observations, 1u);
  EXPECT_EQ(result.value().executions[1].valid_observations, 1u);
  EXPECT_TRUE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, AtomicOrdinaryOverlapRequiresACompleteBarrierWitnessToBeOrdered) {
  test::manifest(capture_.header).sites[0].access = AccessKind::Atomic;
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  for (const bool complete : {true, false}) {
    if (!complete)
      wire::write_u32(capture_.executions[0].report, completion(1) + wire::kSyncCommit, 0);
    const auto result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const auto &execution = result.value().executions.front();
    EXPECT_EQ(execution.ordered_pairs, complete ? 1u : 0u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, complete ? 0u : 1u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.coverage_complete, complete);
  }
}

TEST_F(ConSanAnalysis, FixedBarrierCertificateOrdersRetainedPairAndSurvivesUnrelatedLocalGap) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  for (const bool unrelated_gap : {false, true}) {
    if (unrelated_gap) {
      wire::write_u32(capture_.executions[0].report, record(0, 1) + wire::kRecordCommit, 7);
    }
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_EQ(execution.valid_observations, 2u);
    EXPECT_EQ(execution.ordered_pairs, 1u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
    EXPECT_EQ(execution.completed_barrier_waves, 2u);
    EXPECT_EQ(execution.coverage_complete, !unrelated_gap);
    ASSERT_EQ(execution.ordered_examples.size(), 1u);
    EXPECT_EQ(execution.ordered_examples[0].ordering, LdsRaceOrdering::Ordered);
    EXPECT_EQ(execution.ordered_examples[0].first.mask, 1u);
  }
}

TEST_F(ConSanAnalysis,
       DecoderControlMissingParticipantOccurrenceOrCompletionRetainsUnresolvedPair) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  barrier();
  const std::vector<uint8_t> saved = capture_.executions[0].report;
  constexpr std::array<std::pair<uint32_t, uint32_t>, 6> corruptions = {{
      {wire::kSyncCommit, 9},
      {wire::kSyncGroup, 9},
      {wire::kSyncOccurrence, 9},
      {wire::kSyncParticipants, 9},
      {wire::kSyncSequence, 9},
      // Zero stays below final_phase, isolating the slot's required phase.
      {wire::kSyncPhase, 0},
  }};
  for (const std::pair<uint32_t, uint32_t> &corruption : corruptions) {
    SCOPED_TRACE(corruption.first);
    capture_.executions[0].report = saved;
    wire::write_u32(capture_.executions[0].report, completion(1) + corruption.first,
                    corruption.second);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_EQ(execution.ordered_pairs, 0u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
    EXPECT_EQ(execution.collision_groups, 1u);
    EXPECT_FALSE(execution.coverage_complete);
    ASSERT_EQ(execution.unresolved_examples.size(), 1u);
    EXPECT_TRUE(execution.cross_wave_examples.empty());
    EXPECT_EQ(execution.unresolved_examples[0].ordering, LdsRaceOrdering::Unresolved);
    EXPECT_NE(render_lds_race_analysis(result.value()).find("\"ordering\":\"ordering_unresolved\""),
              std::string::npos);
  }
}

TEST_F(ConSanAnalysis, FixedBarrierDoesNotEraseIndependentSameInstructionCollision) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  barrier();
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.ordered_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_TRUE(execution.coverage_complete);
}

TEST_F(ConSanAnalysis, OrderedPairSummaryAccumulatesAcrossExecutionsWithoutExamples) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  append_execution();
  std::vector<uint8_t> &second = capture_.executions[1].report;
  second = capture_.executions[0].report;
  wire::write_u64(second, wire::kReportSerial, 2);
  for (const uint32_t wave : {0u, 1u}) {
    wire::write_u64(second, row(wave) + wire::kRowSerial, 2);
  }

  const FailureOr<LdsRaceResult> result = analyze({0});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_TRUE(result.value().coverage_complete);
  ASSERT_EQ(result.value().executions.size(), 2u);
  EXPECT_EQ(result.value().ordered_pairs, 2u);
  for (const LdsRaceExecutionResult &execution : result.value().executions) {
    EXPECT_EQ(execution.ordered_pairs, 1u);
    EXPECT_TRUE(execution.ordered_examples.empty());
  }
  const std::string json = render_analysis(result.value());
  const size_t executions = json.find("\"executions\":");
  ASSERT_NE(executions, std::string::npos);
  EXPECT_NE(json.substr(0, executions).find("\"ordered_pairs\":2"), std::string::npos);
  EXPECT_FALSE(has_findings(result.value()));
}

TEST_F(ConSanAnalysis, FixedBarrierReadReadAndDisjointKeepObservationsWithoutCandidates) {
  for (const bool read_read : {false, true}) {
    capture_.executions.clear();
    append_execution();
    test::manifest(capture_.header).sites[0].access =
        read_read ? AccessKind::Read : AccessKind::Write;
    access(0, 0, 1);
    access(1, 1, 1, read_read ? 0 : 4);
    barrier();
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_EQ(execution.valid_observations, 2u);
    EXPECT_EQ(execution.ordered_pairs, 0u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.collision_groups, 0u);
    EXPECT_TRUE(execution.coverage_complete);
  }
}

TEST_F(ConSanAnalysis, DecoderControlCompletionSequenceMustBracketTheActualEndpoints) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  // This otherwise valid completion wrongly places the consumer before wait.
  wire::write_u32(capture_.executions[0].report, completion(1) + wire::kSyncSequence, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.completed_barrier_waves, 1u);
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_GT(execution.integrity_events, 0u);
  EXPECT_EQ(execution.ordered_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
}

TEST_F(ConSanAnalysis, DecoderControlEmptyAccessWaveStillMustCompleteTheSameBarrier) {
  capture_.executions.clear();
  append_execution({{65, 1, 1}, {65, 1, 1}});
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  for (const uint32_t wave : {0u, 1u}) {
    wire::write_u32(capture_.executions[0].report, completion(wave) + wire::kSyncParticipants, 3);
  }
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_EQ(execution.required_barrier_waves, 3u);
  EXPECT_EQ(execution.completed_barrier_waves, 2u);
  EXPECT_EQ(execution.ordered_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
}

TEST_F(ConSanAnalysis, SamePhaseRequiresCompleteJournalWithoutUnexpectedTail) {
  // Place both endpoints after B0 so their interval needs a complete exit.
  // Before B0, entry and B0 already close the interval independently of any
  // unexpected later journal bytes, just as they do under loop journal loss.
  many_barrier_sites(2, 0);
  test::manifest(capture_.header).sites[1].access = AccessKind::Read;
  access(0, 0, 1);
  access(1, 1, 1);
  std::vector<uint8_t> &report = capture_.executions[0].report;
  const FailureOr<LdsRaceResult> baseline = analyze();
  ASSERT_TRUE(baseline.succeeded()) << error_.message();
  ASSERT_TRUE(baseline.value().coverage_complete);
  ASSERT_EQ(baseline.value().cross_wave_pairs, 1u);
  ASSERT_EQ(baseline.value().ordering_unresolved_pairs, 0u);
  const std::vector<uint8_t> saved = report;
  for (const bool unexpected_tail : {false, true}) {
    SCOPED_TRACE(unexpected_tail);
    report = saved;
    if (unexpected_tail) {
      std::copy_n(report.begin() + completion(1), kSyncRecordBytes,
                  report.begin() + completion(1, 1));
    } else {
      wire::write_u32(report, completion(1) + wire::kSyncCommit, 0);
    }
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_EQ(execution.valid_observations, 2u);
    EXPECT_FALSE(execution.coverage_complete);
    EXPECT_GT(execution.integrity_events, 0u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.ordered_pairs, 0u);
    EXPECT_EQ(execution.collision_groups, 0u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
    ASSERT_EQ(execution.unresolved_examples.size(), 1u);
    EXPECT_EQ(execution.unresolved_examples[0].ordering, LdsRaceOrdering::Unresolved);
    EXPECT_FALSE(has_findings(result.value()));
    if (unexpected_tail) {
      EXPECT_TRUE(std::ranges::any_of(execution.issues, [](const LdsRaceIntegrityIssue &issue) {
        return issue.reason == "unexpected_barrier_completion";
      }));
    }

    const std::string json = render_analysis(result.value());
    EXPECT_NE(json.find("\"unresolved_examples\":[{\"kind\":\"cross_wave\""), std::string::npos);
    EXPECT_NE(json.find("\"ordering\":\"ordering_unresolved\""), std::string::npos);
    const FailureOr<LdsRaceResult> hidden = analyze({.examples = 0});
    ASSERT_TRUE(hidden.succeeded()) << error_.message();
    LdsRaceResult expected = result.value();
    expected.executions.front().unresolved_examples.clear();
    // Compare the real analyzer/renderer output: hiding examples must preserve
    // the unresolved count, incomplete coverage, diagnostics, and verdict.
    EXPECT_EQ(render_analysis(hidden.value()), render_analysis(expected));
  }
}

TEST_F(ConSanAnalysis, StaticAccessPhaseCannotBeReplacedByAConsistentDynamicBoundary) {
  many_barrier_sites(2, 0);
  test::manifest(capture_.header).sites[1].access = AccessKind::Read;
  access(0, 0, 1);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> baseline = analyze();
  ASSERT_TRUE(baseline.succeeded()) << error_.message();
  ASSERT_TRUE(baseline.value().coverage_complete);
  ASSERT_EQ(baseline.value().cross_wave_pairs, 1u);

  std::vector<uint8_t> &report = capture_.executions[0].report;
  // These values agree with each other and fit the final counters, but the
  // qualified write site is after B0 and therefore cannot have phase zero.
  wire::write_u32(report, record(0, 0) + wire::kRecordPhase, 0);
  wire::write_u32(report, completion(0) + wire::kSyncSequence, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_GT(execution.integrity_events, 0u);
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
  EXPECT_EQ(execution.ordered_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
  EXPECT_TRUE(std::ranges::any_of(execution.issues, [](const LdsRaceIntegrityIssue &issue) {
    return issue.reason == "access_phase_contradicts_qualified_program";
  }));
}

TEST_F(ConSanAnalysis, DecoderControlBoundaryContradictionKeepsIndependentEndpointsAndCollisions) {
  many_barrier_sites(4, 1);
  Manifest &facts = test::manifest(capture_.header);
  facts.sites[3].original_offset = 64;
  facts.barriers.push_back({.original_drain = 40,
                            .original_signal = 44,
                            .original_wait = 48,
                            .prepared_signal = 700,
                            .prepared_wait = 704,
                            .completion_probe = 708,
                            .probe_size = 32});
  access(0, 1, 1, 0, 3);
  access(0, 2, 2, 4);
  access(1, 3, 1, 0);
  access(1, 3, 1, 4, 2, 1);
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, 2);
    wire::write_u32(report, completion(wave) + wire::kSyncSequence, 0);
    std::copy_n(report.begin() + completion(wave), kSyncRecordBytes,
                report.begin() + completion(wave, 1));
    wire::write_u32(report, completion(wave, 1) + wire::kSyncGroup, 1);
    wire::write_u32(report, completion(wave, 1) + wire::kSyncPhase, 2);
    wire::write_u32(report, completion(wave, 1) + wire::kSyncSequence, wave == 0 ? 2 : 0);
  }
  const FailureOr<LdsRaceResult> baseline = analyze();
  ASSERT_TRUE(baseline.succeeded()) << error_.message();
  ASSERT_TRUE(baseline.value().coverage_complete);
  ASSERT_EQ(baseline.value().executions[0].ordered_pairs, 2u);
  ASSERT_EQ(baseline.value().collision_groups, 1u);

  // Only site 1 contradicts B0. Site 2 in the same wave still has a consistent
  // B1 witness, and the exact store mask needs no cross-wave ordering proof.
  wire::write_u32(report, completion(0) + wire::kSyncSequence, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions[0];
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_EQ(execution.integrity_events, 1u);
  EXPECT_EQ(execution.completed_barrier_waves, 1u);
  EXPECT_EQ(execution.valid_observations, 4u);
  EXPECT_EQ(execution.rejected_records, 0u);
  EXPECT_EQ(execution.ordered_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_EQ(execution.collision_lane_pairs, 1u);
  ASSERT_EQ(execution.ordered_examples.size(), 1u);
  EXPECT_EQ(execution.ordered_examples[0].first.site, 2u);
  ASSERT_EQ(execution.unresolved_examples.size(), 1u);
  EXPECT_EQ(execution.unresolved_examples[0].first.site, 1u);
  EXPECT_TRUE(has_findings(result.value()));
}

TEST_F(ConSanAnalysis, DecoderControlContradictionIsLocalToOccurrenceAndWorkgroup) {
  many_barrier_sites(3, 1);
  Manifest &facts = test::manifest(capture_.header);
  facts.scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  facts.entry.workgroup_id_scalar = 117;
  facts.sites[2].original_offset = 64;
  facts.barriers.push_back({.original_drain = 40,
                            .original_signal = 44,
                            .original_wait = 48,
                            .prepared_signal = 700,
                            .prepared_wait = 704,
                            .completion_probe = 708,
                            .probe_size = 32});
  capture_.executions.clear();
  append_execution({{128, 1, 1}, {64, 1, 1}});
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t workgroup = 0; workgroup < 2; ++workgroup) {
    const uint32_t first = workgroup * 2;
    access(first, 1, 1);
    access(first, 1, 2, 0, 2, 1);
    wire::write_u32(report, record(first, 1, 1) + wire::kRecordPhase, 3);
    access(first + 1, 2, 1);
    wire::write_u32(report, record(first + 1, 2) + wire::kRecordPhase, 4);
    for (uint32_t wave = first; wave < first + 2; ++wave) {
      wire::write_u32(report, row(wave) + wire::kRowFinalPhase, 4);
      for (uint32_t slot = 0; slot < 4; ++slot) {
        wire::write_u32(report, completion(wave, slot) + wire::kSyncCommit,
                        wire::kBarrierCompleted);
        wire::write_u32(report, completion(wave, slot) + wire::kSyncGroup, slot % 2);
        wire::write_u32(report, completion(wave, slot) + wire::kSyncOccurrence, slot / 2 + 1);
        wire::write_u32(report, completion(wave, slot) + wire::kSyncParticipants, 2);
        wire::write_u32(report, completion(wave, slot) + wire::kSyncPhase, slot + 1);
        wire::write_u32(report, completion(wave, slot) + wire::kSyncSequence,
                        wave == first ? (slot + 1) / 2 : 0);
      }
    }
  }
  const FailureOr<LdsRaceResult> baseline = analyze();
  ASSERT_TRUE(baseline.succeeded()) << error_.message();
  ASSERT_TRUE(baseline.value().coverage_complete);
  ASSERT_EQ(baseline.value().executions[0].ordered_pairs, 4u);

  // B0's second occurrence contradicts only WG0's second retained operation.
  // The first operation at that static site and both WG1 operations stay valid.
  wire::write_u32(report, completion(0, 2) + wire::kSyncSequence, 2);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions[0];
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_EQ(execution.integrity_events, 1u);
  EXPECT_EQ(execution.completed_barrier_waves, 3u);
  EXPECT_EQ(execution.required_barrier_waves, 4u);
  EXPECT_EQ(execution.valid_observations, 6u);
  EXPECT_EQ(execution.rejected_records, 0u);
  EXPECT_EQ(execution.ordered_pairs, 3u);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
  ASSERT_EQ(execution.unresolved_examples.size(), 1u);
  EXPECT_EQ(execution.unresolved_examples[0].first.workgroup, 0u);
  EXPECT_EQ(execution.unresolved_examples[0].second.workgroup, 0u);
  EXPECT_EQ(execution.unresolved_examples[0].first.sequence, 2u);
  EXPECT_EQ(execution.unresolved_examples[0].first.phase, 3u);
  ASSERT_EQ(execution.ordered_examples.size(), 3u);
  for (const LdsRaceFinding &example : execution.ordered_examples) {
    EXPECT_EQ(example.first.workgroup, example.second.workgroup);
    if (example.first.workgroup == 0) {
      EXPECT_EQ(example.first.sequence, 1u);
    }
  }
  EXPECT_FALSE(has_findings(result.value()));
}

TEST_F(ConSanAnalysis, MultipleEpochsUseIndependentGroupWitnesses) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  Manifest &facts = test::manifest(capture_.header);
  facts.sites[1].original_offset = 64;
  facts.barriers.push_back({.original_drain = 48,
                            .original_signal = 52,
                            .original_wait = 56,
                            .prepared_signal = 400,
                            .prepared_wait = 404,
                            .completion_probe = 408,
                            .probe_size = 64});
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, 2);
    std::copy_n(report.begin() + completion(wave), kSyncRecordBytes,
                report.begin() + completion(wave, 1));
    wire::write_u32(report, completion(wave, 1) + wire::kSyncGroup, 1);
    wire::write_u32(report, completion(wave, 1) + wire::kSyncPhase, 2);
  }
  wire::write_u32(report, record(1, 1) + wire::kRecordPhase, 2);
  for (uint32_t missing = 0; missing < 3; ++missing) {
    if (missing) {
      wire::write_u32(report, completion(1, missing - 1) + wire::kSyncCommit, 0);
    }
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions[0];
    EXPECT_EQ(execution.ordered_pairs, missing < 2 ? 1u : 0u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, missing == 2 ? 1u : 0u);
    EXPECT_EQ(execution.coverage_complete, missing == 0);
    EXPECT_FALSE(has_findings(result.value()));
  }
}

TEST_F(ConSanAnalysis, AdjacentEffectiveByteRangesAreDisjointWithNonemptyObservations) {
  access(0, 0, 1, 0);
  access(1, 1, 1, 4);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 2u);
  EXPECT_TRUE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, OverlappingReadReadGroupsAreNotConflictCandidates) {
  test::manifest(capture_.header).sites[0].access = AccessKind::Read;
  access(0, 0, 1, 0, 3);
  access(1, 1, 1, 0, UINT32_MAX);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().collision_groups, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 2u);
}

TEST_F(ConSanAnalysis, DifferentInstructionsInOneWaveDoNotBecomeCrossWavePairs) {
  access(0, 0, 1);
  access(0, 1, 2);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().collision_groups, 0u);
  EXPECT_TRUE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, ReusedStorageAcrossDispatchesCannotJoinOtherwiseOverlappingGroups) {
  append_execution();
  access(0, 0, 1, 0, 1, 0, 0);
  access(1, 1, 1, 0, 1, 0, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(result.value().executions.size(), 2u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions[0].valid_observations, 1u);
  EXPECT_EQ(result.value().executions[1].valid_observations, 1u);
  EXPECT_EQ(result.value().executions[1].serial, 2u);
}

TEST_F(ConSanAnalysis, ExactOrdinaryWriteMaskReportsFullParticipatingLanesAndPairCount) {
  access(0, 0, 1, 0, 0x80000005u);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().collision_groups, 1u);
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.collision_lane_pairs, 3u);
  ASSERT_EQ(execution.collision_examples.size(), 1u);
  EXPECT_EQ(execution.collision_examples.front().first.mask, 0x80000005u);
  EXPECT_EQ(execution.valid_observations, 1u);
  EXPECT_EQ(render_analysis(result.value()).find("\"ordering\":"), std::string::npos);
  EXPECT_NE(render_analysis(result.value()).find("ordinary_store_exact_mask"), std::string::npos);
}

TEST_F(ConSanAnalysis, SingleLaneWriteAndMultiLaneReadNeverInventInstructionCollisions) {
  access(0, 0, 1, 0, 0x80000000u);
  access(0, 1, 2, 0, UINT32_MAX);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().collision_groups, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 2u);
}

TEST_F(ConSanAnalysis, BothChecksConsumeTheSameWriteObservation) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_EQ(execution.cross_wave_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_lane_pairs, 2u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_EQ(execution.cross_wave_examples.front().first,
            execution.collision_examples.front().first);
}

TEST_F(ConSanAnalysis, ExampleLimitsNeverChangeCompleteGroupOrLanePairCounts) {
  test::manifest(capture_.header).sites[1].access = AccessKind::Write;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    access(wave, 0, 1, 0, 3);
    access(wave, 1, 2, 0, 3);
  }
  for (const uint32_t limit : {0, 1, 3, 8}) {
    const FailureOr<LdsRaceResult> result = analyze({limit});
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_EQ(execution.cross_wave_pairs, 4u);
    EXPECT_EQ(execution.cross_wave_lane_pairs, 16u);
    EXPECT_EQ(execution.collision_groups, 4u);
    EXPECT_EQ(execution.collision_lane_pairs, 4u);
    EXPECT_EQ(execution.cross_wave_examples.size(), std::min(limit, 4u));
    EXPECT_EQ(execution.collision_examples.size(), std::min(limit, 4u));
  }
}

TEST_F(ConSanAnalysis, MalformedRowDoesNotEraseAnotherWavesIndependentCollision) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  wire::write_u32(capture_.executions[0].report, row(1) + wire::kRowStatus, UINT32_MAX);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().collision_groups, 1u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_TRUE(result.value().executions.front().ordering_uncertain);
}

TEST_F(ConSanAnalysis, NoSynchronizationKeepsConflictsDespiteUnrelatedRowDamage) {
  struct RowDamage {
    uint32_t target;
    uint32_t wave_size;
    uint32_t wave;
    uint32_t field;
    uint32_t value;
  };
  constexpr std::array<RowDamage, 3> cases = {{
      {.target = 1250, .wave_size = 32, .wave = 0, .field = wire::kRowStatus, .value = 0},
      {.target = 1100, .wave_size = 64, .wave = 1, .field = wire::kRowSerial, .value = 7},
      {.target = 1250, .wave_size = 32, .wave = 2, .field = wire::kRowFinalPhase, .value = 7},
  }};
  for (const RowDamage &damage : cases) {
    SCOPED_TRACE(damage.field);
    test::manifest(capture_.header).target = damage.target;
    test::manifest(capture_.header).wave_size = damage.wave_size;
    capture_.executions.clear();
    append_execution({{3 * damage.wave_size, 1, 1}, {3 * damage.wave_size, 1, 1}});
    access((damage.wave + 1) % 3, 0, 1);
    access((damage.wave + 2) % 3, 1, 1);
    const FailureOr<LdsRaceResult> baseline = analyze();
    ASSERT_TRUE(baseline.succeeded()) << error_.message();
    ASSERT_EQ(baseline.value().cross_wave_pairs, 1u);
    ASSERT_TRUE(baseline.value().coverage_complete);
    // The missing/malformed row has no endpoint and cannot introduce
    // synchronization into the statically qualified no-sync profile.
    wire::write_u32(capture_.executions[0].report, row(damage.wave) + damage.field, damage.value);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions[0];
    EXPECT_EQ(execution.valid_observations, 2u);
    EXPECT_EQ(execution.cross_wave_pairs, 1u);
    EXPECT_EQ(execution.cross_wave_lane_pairs, 1u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
    EXPECT_EQ(execution.required_barrier_waves, 0u);
    EXPECT_EQ(execution.completed_barrier_waves, 0u);
    EXPECT_GT(execution.integrity_events, 0u);
    EXPECT_FALSE(result.value().coverage_complete);
    EXPECT_TRUE(has_findings(result.value()));
  }
}

TEST_F(ConSanAnalysis, NoSynchronizationStillRejectsDamagedEndpointRows) {
  test::manifest(capture_.header).target = 1250;
  capture_.executions.clear();
  append_execution({{96, 1, 1}, {96, 1, 1}});
  access(0, 0, 1);
  access(1, 1, 1);
  auto &report = capture_.executions[0].report;
  const auto original = report;
  for (const uint32_t damaged_wave : {0u, 1u}) {
    report = original;
    wire::write_u32(report, row(damaged_wave) + wire::kRowStatus, 0);
    const auto result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_EQ(result.value().executions[0].valid_observations, 1u);
    EXPECT_EQ(result.value().cross_wave_pairs, 0u);
    EXPECT_EQ(result.value().ordering_unresolved_pairs, 0u);
    EXPECT_FALSE(result.value().coverage_complete);
    EXPECT_FALSE(has_findings(result.value()));
  }
}

TEST_F(ConSanAnalysis, NoSynchronizationStillRejectsContradictoryEndpointPhases) {
  access(0, 0, 1);
  access(1, 1, 1);
  auto &report = capture_.executions[0].report;
  const auto original = report;
  for (const uint32_t damaged_wave : {0u, 1u}) {
    report = original;
    wire::write_u32(report, record(damaged_wave, damaged_wave) + wire::kRecordPhase, 1);
    const auto result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_EQ(result.value().executions[0].valid_observations, 2u);
    EXPECT_GT(result.value().executions[0].integrity_events, 0u);
    EXPECT_EQ(result.value().cross_wave_pairs, 0u);
    EXPECT_EQ(result.value().ordering_unresolved_pairs, 1u);
    EXPECT_FALSE(result.value().coverage_complete);
    EXPECT_FALSE(has_findings(result.value()));
  }
}

TEST_F(ConSanAnalysis, BarrierProfileStillRequiresAnEmptyAccessParticipant) {
  many_barrier_sites(2, 2, 3);
  access(0, 0, 1);
  access(1, 1, 1);
  auto &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 3; ++wave) {
    wire::write_u32(report, completion(wave) + wire::kSyncSequence, wave < 2 ? 1 : 0);
  }
  const auto baseline = analyze();
  ASSERT_TRUE(baseline.succeeded()) << error_.message();
  ASSERT_EQ(baseline.value().cross_wave_pairs, 1u);
  ASSERT_TRUE(baseline.value().coverage_complete);
  wire::write_u32(report, row(2) + wire::kRowStatus, 0);
  const auto result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().executions[0].valid_observations, 2u);
  EXPECT_EQ(result.value().executions[0].required_barrier_waves, 3u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().ordering_unresolved_pairs, 1u);
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_FALSE(has_findings(result.value()));
}

TEST_F(ConSanAnalysis, MalformedLocalRecordPreservesAnIndependentCrossWavePair) {
  access(0, 0, 1);
  access(0, 1, 2);
  access(1, 1, 1);
  wire::write_u32(capture_.executions[0].report, record(0, 1) + wire::kRecordCommit, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().executions.front().rejected_records, 1u);
}

TEST_F(ConSanAnalysis, InvalidRecordIdentityMaskAndBoundsAreGapsBeyondTheContainerChecksum) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  std::vector<uint8_t> &report = capture_.executions.front().report;
  const std::vector<uint8_t> saved = report;
  for (const auto &[field, value] :
       std::array<std::pair<uint32_t, uint32_t>, 12>{{{wire::kRecordCommit, 1},
                                                      {wire::kRecordAnalysis, 2},
                                                      {wire::kRecordKind, 2},
                                                      {wire::kRecordSite, 99},
                                                      {wire::kRecordRange, 1},
                                                      {wire::kRecordSequence, 2},
                                                      {wire::kRecordWidth, 8},
                                                      {wire::kRecordGuestMask, 0},
                                                      {wire::kRecordFlags, 0},
                                                      {wire::kRecordFlags, 3},
                                                      {wire::kRecordLdsOffset, 128},
                                                      {wire::kRecordReservedBegin, 1}}}) {
    report = saved;
    wire::write_u32(report, record(1, 1) + field, value);
    // Production encoder makes a valid whole container and frame checksum.
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_FALSE(result.value().coverage_complete) << field;
    EXPECT_EQ(result.value().cross_wave_pairs, 0u) << field;
    EXPECT_EQ(result.value().collision_groups, 1u) << field;
  }
}

TEST_F(ConSanAnalysis, MissingStaticMappingIsAnInputErrorNotInstructionZero) {
  access(0, 0, 1);
  test::manifest(capture_.header).sites[0].original_size = 0;
  EXPECT_TRUE(analyze_typed(capture_, {}, error_.emitter()).failed());
  EXPECT_EQ(error_.message(), "invalid site identity, mapping or address group");
}

TEST_F(ConSanAnalysis, UnmodeledScopeCannotBeTreatedAsConfirmedUnorderedAccesses) {
  access(0, 0, 1);
  access(1, 1, 1);
  test::manifest(capture_.header).scope = static_cast<ScopeKind>(99);
  EXPECT_TRUE(analyze_typed(capture_, {}, error_.emitter()).failed());
  EXPECT_EQ(error_.message(), "empty image or unsupported target, ABI, scope or options");
}

TEST_F(ConSanAnalysis, ApplicationTransportAnalyzesLdsWithoutClaimingSignalForwarding) {
  access(0, 0, 1);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> controlled = analyze();
  ASSERT_TRUE(controlled.succeeded()) << error_.message();
  ASSERT_EQ(controlled.value().cross_wave_pairs, 1u);
  capture_.header.execution_conditions = capture_execution_conditions(ExecutionMode::Application);
  // A changed header cannot relabel evidence from a different transport.
  EXPECT_TRUE(analyze_typed(capture_, {}, error_.emitter()).failed());
  EXPECT_TRUE(encode().failed());
  capture_.executions.front().completion = completion_evidence(ExecutionMode::Application);
  const FailureOr<LdsRaceResult> application = analyze();
  ASSERT_TRUE(application.succeeded()) << error_.message();
  EXPECT_EQ(application.value().execution_conditions, kApplicationExecutionConditions);
  EXPECT_EQ(application.value().cross_wave_pairs, controlled.value().cross_wave_pairs);
  EXPECT_EQ(application.value().coverage_complete, controlled.value().coverage_complete);
  capture_.header.execution_conditions = capture_execution_conditions(ExecutionMode::Controlled);
  EXPECT_TRUE(analyze_typed(capture_, {}, error_.emitter()).failed());
  EXPECT_TRUE(encode().failed());
}

TEST_F(ConSanAnalysis, TypedHeaderConditionsAndIdentityCannotBypassFileValidation) {
  const CaptureHeader saved = capture_.header;
  for (const uint32_t unsupported : {0u, 14u, 13u, 11u, 7u, 31u}) {
    capture_.header.execution_conditions = unsupported;
    EXPECT_TRUE(analyze_typed(capture_, {}, error_.emitter()).failed());
    EXPECT_EQ(error_.message(), "unsupported capture execution conditions");
  }
  capture_.header = saved;
  capture_.header.run_identity = {};
  EXPECT_TRUE(analyze_typed(capture_, {}, error_.emitter()).failed());
  EXPECT_EQ(error_.message(), "missing capture run identity");
  capture_.header = saved;
  EXPECT_TRUE(analyze_typed(capture_).succeeded());
}

TEST_F(ConSanAnalysis, TypedAnalysisBudgetsRetainTheirIndependentLimits) {
  EXPECT_TRUE(analyze_typed(capture_, {kMaxFindingExamples}).succeeded());
  EXPECT_TRUE(analyze_typed(capture_, {kMaxFindingExamples + 1}, error_.emitter()).failed());
  EXPECT_EQ(error_.message(), "finding example budget exceeded");
  capture_.executions.resize(CaptureLimits{}.executions + 1);
  EXPECT_TRUE(analyze_typed(capture_, {}, error_.emitter()).failed());
  EXPECT_EQ(error_.message(), "capture execution budget exceeded");
}

TEST_F(ConSanAnalysis, InvalidReportParentPreservesFindingsFromAnotherExecution) {
  append_execution();
  access(0, 0, 1, 0, 1, 0, 1);
  access(1, 1, 1, 0, 1, 0, 1);
  wire::write_u32(capture_.executions[0].report, wire::kReportRowStride, UINT32_MAX);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().executions[0].valid_observations, 0u);
  EXPECT_EQ(result.value().executions[1].cross_wave_pairs, 1u);
}

TEST_F(ConSanAnalysis, RowIdentityPublicationAndFinalSequenceMustMatchTheirExecution) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  std::vector<uint8_t> &report = capture_.executions.front().report;
  const std::vector<uint8_t> saved = report;
  for (const auto &[field, value] :
       std::array<std::pair<uint32_t, uint32_t>, 8>{{{wire::kRowSerial, 2},
                                                     {wire::kRowEntry, 1},
                                                     {wire::kRowWave, 0},
                                                     {wire::kRowStatus, wire::kRowEntered},
                                                     {wire::kRowLoss, 1},
                                                     {wire::kRowFinalSequence, 0},
                                                     {wire::kRowFinalSequence, UINT32_MAX},
                                                     {wire::kRowReservedBegin, 1}}}) {
    report = saved;
    wire::write_u32(report, row(1) + field, value);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_FALSE(result.value().coverage_complete);
    EXPECT_EQ(result.value().cross_wave_pairs, 0u);
    EXPECT_EQ(result.value().collision_groups, 1u);
  }
}

TEST_F(ConSanAnalysis, InvertedSiteOccurrenceOrderCannotBeAValidForwardOnlyExecution) {
  access(0, 0, 2);
  access(0, 1, 1);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
}

TEST_F(ConSanAnalysis, ShortTypedReportCannotEscapeTheCoreBoundsCheck) {
  capture_.executions[0].report.resize(1);
  const FailureOr<LdsRaceResult> result = analyze_typed(capture_, {}, error_.emitter());
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().executions.front().valid_observations, 0u);
}

TEST_F(ConSanAnalysis, DuplicateOccurrencesAndOverlappingMasksCannotInflateFindings) {
  access(0, 0, 1);
  access(0, 1, 1);
  access(1, 1, 1);
  FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions[0].rejected_records, 2u);
  // Another address group in the same instruction cannot reuse lane zero.
  std::ranges::fill(
      std::span(capture_.executions[0].report).subspan(record(0, 1), kAccessRecordBytes), 0);
  access(0, 0, 1, 4, 1, 1);
  result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_FALSE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, ReservationLossRetainsValidFindingsButMakesCoverageIncomplete) {
  for (uint32_t address_group = 0; address_group < 4; ++address_group) {
    access(0, 0, 1, address_group * 4, 1u << address_group, address_group);
  }
  access(1, 1, 1);
  wire::write_u32(capture_.executions[0].report, row(0) + wire::kRowLoss, 1);
  wire::write_u32(capture_.executions[0].report, row(0) + wire::kRowStatus,
                  wire::kRowEntered | wire::kRowExited | wire::kRowExhausted);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().executions.front().lost_occurrences, 1u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 5u);
}

TEST_F(ConSanAnalysis, UnalignedOperandsNeedAlignmentModeEvidenceAndCannotProduceACleanVerdict) {
  access(0, 0, 1, 1, 3);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().collision_groups, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 1u);
  EXPECT_EQ(result.value().executions.front().issues.front().reason,
            "unaligned_lds_alignment_mode_unavailable");
}

TEST_F(ConSanAnalysis, ReaderOwnedEvidenceAndRenderedSemanticReplayAreDeterministic) {
  test::manifest(capture_.header).entry.name = "kernel\"\\\n";
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  FailureOr<std::vector<uint8_t>> bytes = encode();
  ASSERT_TRUE(bytes.succeeded());
  FailureOr<Capture> first = decode_capture(bytes.value());
  FailureOr<Capture> second = decode_capture(bytes.value());
  ASSERT_TRUE(first.succeeded());
  ASSERT_TRUE(second.succeeded());
  std::ranges::fill(bytes.value(), 0xcd);
  const FailureOr<LdsRaceResult> a = analyze_capture(first.value());
  const FailureOr<LdsRaceResult> b = analyze_capture(second.value());
  ASSERT_TRUE(a.succeeded());
  ASSERT_TRUE(b.succeeded());
  EXPECT_EQ(a.value(), b.value());
  EXPECT_EQ(render_analysis(a.value()), render_analysis(b.value()));
  EXPECT_NE(render_analysis(a.value()).find("kernel\\\"\\\\\\u000a"), std::string::npos);
}

TEST_F(ConSanAnalysis, AcceptedUtf8EntryNamesRetainTheirBytesInJson) {
  // Exercise two-, three- and four-byte UTF-8 alongside JSON escapes.
  const std::string name = "lds_producer_consum\xc3\xa9_\xe6\xa0\xb8_\xf0\x9f\x98\x80";
  test::manifest(capture_.header).entry.name = name + "\"\\\n\t";
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().entry, test::manifest(capture_.header).entry.name);
  EXPECT_NE(
      render_analysis(result.value()).find("\"entry\":\"" + name + "\\\"\\\\\\u000a\\u0009\""),
      std::string::npos);
}

TEST_F(ConSanAnalysis, InvalidUtf8EntryNamesAreAnalysisInputErrorsAfterBinaryRoundTrip) {
  for (const std::string_view invalid :
       {"\x80", "\xbf", "\xc0\x80", "\xc1\xbf", "\xc2", "\xc2\x7f", "\xdf\xff", "\xe0\x80\x80",
        "\xed\xa0\x80", "\xef\xbf", "\xf0\x80\x80\x80", "\xf4\x90\x80\x80", "\xf5\x80\x80\x80",
        "\xf8\x88\x80\x80\x80", "\xff"}) {
    test::manifest(capture_.header).entry.name = "kernel_" + std::string(invalid);
    // Binary capture remains byte preserving; only JSON-producing analysis
    // requires text. Re-encode each case so checksums cannot hide this check.
    const FailureOr<std::vector<uint8_t>> bytes = encode();
    ASSERT_TRUE(bytes.succeeded()) << error_.message();
    const FailureOr<Capture> decoded = decode_capture(bytes.value(), error_.emitter());
    ASSERT_TRUE(decoded.succeeded()) << error_.message();
    EXPECT_EQ(test::manifest(decoded.value().header).entry.name,
              test::manifest(capture_.header).entry.name);
    const FailureOr<LdsRaceResult> result = analyze_capture(decoded.value(), {}, error_.emitter());
    EXPECT_TRUE(result.failed());
    EXPECT_EQ(error_.message(), "entry name is not valid UTF-8 for JSON output");
  }
}

TEST_F(ConSanAnalysis, Utf8ScalarBoundariesRetainTheirBytesInJson) {
  const std::string name = "kernel_\xc2\x80\xdf\xbf\xe0\xa0\x80\xed\x9f\xbf"
                           "\xee\x80\x80\xef\xbf\xbf\xf0\x90\x80\x80\xf4\x8f\xbf\xbf";
  test::manifest(capture_.header).entry.name = name;
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_NE(render_analysis(result.value()).find("\"entry\":\"" + name + "\""), std::string::npos);
}

TEST_F(ConSanAnalysis, EmptyValidObservationsAndNoExecutionsNeverClaimRaceFreedom) {
  FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 0u);
  EXPECT_NE(render_analysis(result.value()).find("no_observed_findings"), std::string::npos);
  EXPECT_EQ(render_analysis(result.value()).find("race_free"), std::string::npos);
  capture_.executions.clear();
  result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, LargeCandidateCountsRemainCompleteWithOneExample) {
  constexpr uint32_t sites = 1024;
  many_sites(sites);
  for (uint32_t wave = 0; wave < 2; ++wave) {
    for (uint32_t site = 0; site < sites; ++site) {
      access(wave, site, site + 1, 0, 3);
    }
  }
  const FailureOr<LdsRaceResult> result = analyze({1});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_TRUE(execution.coverage_complete);
  EXPECT_EQ(execution.cross_wave_pairs, uint64_t{sites} * sites);
  EXPECT_EQ(execution.cross_wave_lane_pairs, 4 * uint64_t{sites} * sites);
  EXPECT_EQ(execution.collision_groups, 2 * sites);
  EXPECT_EQ(execution.cross_wave_examples.size(), 1u);
  EXPECT_EQ(execution.collision_examples.size(), 1u);
}

TEST_F(ConSanAnalysis, LargeBarrierCountsRetainAllPairsWithBoundedExamples) {
  constexpr uint32_t sites = 1024;
  for (const uint32_t split : {0u, sites / 2, sites}) {
    for (const bool missing : {false, true}) {
      many_barrier_sites(sites, split);
      for (uint32_t wave = 0; wave < 2; ++wave) {
        for (uint32_t site = 0; site < sites; ++site) {
          access(wave, site, site + 1, 0, wave == 0 ? 3 : 7);
        }
      }
      if (missing) {
        wire::write_u32(capture_.executions[0].report, completion(1) + wire::kSyncCommit, 0);
      }
      for (const uint32_t examples : {0u, 1u, 7u}) {
        SCOPED_TRACE(split);
        SCOPED_TRACE(missing);
        SCOPED_TRACE(examples);
        const FailureOr<LdsRaceResult> result = analyze({examples});
        ASSERT_TRUE(result.succeeded()) << error_.message();
        const LdsRaceExecutionResult &execution = result.value().executions.front();
        const uint64_t separated = 2 * uint64_t{split} * (sites - split);
        const uint64_t ordered = missing ? 0 : separated;
        const uint64_t retained = missing ? 0 : uint64_t{sites} * sites - ordered;
        EXPECT_EQ(execution.valid_observations, 2u * sites);
        EXPECT_EQ(execution.ordered_pairs, ordered);
        EXPECT_EQ(execution.cross_wave_pairs, retained);
        EXPECT_EQ(execution.cross_wave_lane_pairs, 6 * retained);
        EXPECT_EQ(execution.ordering_unresolved_pairs, missing ? uint64_t{sites} * sites : 0);
        EXPECT_EQ(execution.coverage_complete, !missing);
        EXPECT_EQ(execution.collision_groups, 2u * sites);
        EXPECT_EQ(execution.ordered_examples.size(), std::min(uint64_t{examples}, ordered));
        EXPECT_EQ(execution.cross_wave_examples.size(), std::min(uint64_t{examples}, retained));
        for (const LdsRaceFinding &finding : execution.ordered_examples) {
          EXPECT_EQ(finding.ordering, LdsRaceOrdering::Ordered);
          EXPECT_NE(finding.first.site < split, finding.second.site < split);
        }
      }
    }
  }
}

TEST_F(ConSanAnalysis, BarrierCountsAndExampleOrderMatchExhaustiveCandidatePairs) {
  constexpr uint32_t sites = 8, split = 4, waves = 4;
  const std::array<uint32_t, waves> boundaries{2, 6, 0, 8};
  for (const bool missing : {false, true}) {
    many_barrier_sites(sites, split, waves);
    for (uint32_t wave = 0; wave < waves; ++wave) {
      wire::write_u32(capture_.executions[0].report, completion(wave) + wire::kSyncSequence,
                      boundaries[wave]);
      for (uint32_t site = 0; site < sites; ++site) {
        access(wave, site, site + 1, 0, (uint64_t{1} << (wave + 1)) - 1);
      }
    }
    if (missing) {
      wire::write_u32(capture_.executions[0].report, completion(3) + wire::kSyncCommit, 0);
    }
    for (const uint32_t examples : {0u, 1u, 7u, 64u}) {
      std::vector<LdsRaceFinding> ordered, retained, unresolved;
      uint64_t ordered_count = 0, unresolved_count = 0, retained_count = 0, lane_pairs = 0;
      // Brute-force the original candidate order independently of classification
      // buckets, including both static sides and contradictory dynamic boundaries.
      for (uint32_t second_wave = 1; second_wave < waves; ++second_wave) {
        for (uint32_t first_wave = 0; first_wave < second_wave; ++first_wave) {
          for (uint32_t first_site = 0; first_site < sites; ++first_site) {
            for (uint32_t second_site = 0; second_site < sites; ++second_site) {
              const bool first_valid =
                  (first_site + 1 <= boundaries[first_wave]) == (first_site < split);
              const bool second_valid =
                  (second_site + 1 <= boundaries[second_wave]) == (second_site < split);
              LdsRaceOrdering order =
                  missing || !first_valid || !second_valid        ? LdsRaceOrdering::Unresolved
                  : (first_site < split) != (second_site < split) ? LdsRaceOrdering::Ordered
                                                                  : LdsRaceOrdering::Unordered;
              const auto participant = [&](uint32_t wave, uint32_t site) {
                return LdsRaceParticipant{
                    wave,
                    site,
                    site + 1,
                    (uint64_t{1} << (wave + 1)) - 1,
                    0,
                    4,
                    test::manifest(capture_.header).sites[site].original_offset,
                    AccessKind::Write,
                    site < split ? 0u : 1u};
              };
              const uint64_t lanes = uint64_t{first_wave + 1} * (second_wave + 1);
              LdsRaceFinding finding{LdsRaceFindingKind::CrossWave,
                                     participant(first_wave, first_site),
                                     participant(second_wave, second_site), lanes, order};
              if (order == LdsRaceOrdering::Ordered) {
                ++ordered_count;
                if (ordered.size() < examples) {
                  ordered.push_back(finding);
                }
              } else if (order == LdsRaceOrdering::Unresolved) {
                ++unresolved_count;
                if (unresolved.size() < examples) {
                  unresolved.push_back(finding);
                }
              } else {
                ++retained_count;
                lane_pairs += lanes;
                if (retained.size() < examples) {
                  retained.push_back(finding);
                }
              }
            }
          }
        }
      }
      const FailureOr<LdsRaceResult> result = analyze({examples});
      ASSERT_TRUE(result.succeeded()) << error_.message();
      const LdsRaceExecutionResult &execution = result.value().executions.front();
      EXPECT_EQ(execution.ordered_pairs, ordered_count);
      EXPECT_EQ(execution.cross_wave_pairs, retained_count);
      EXPECT_EQ(execution.cross_wave_lane_pairs, lane_pairs);
      EXPECT_EQ(execution.ordering_unresolved_pairs, unresolved_count);
      EXPECT_EQ(execution.ordered_examples, ordered);
      EXPECT_EQ(execution.cross_wave_examples, retained);
      EXPECT_EQ(execution.unresolved_examples, unresolved);
    }
  }
}

TEST_F(ConSanAnalysis, CorruptRecordDiagnosticsAreBoundedWithoutLosingIntegrityCounts) {
  constexpr uint32_t sites = 256;
  many_sites(sites);
  for (uint32_t wave = 0; wave < 2; ++wave) {
    for (uint32_t site = 0; site < sites; ++site) {
      access(wave, site, site + 1);
      wire::write_u32(capture_.executions[0].report, record(wave, site) + wire::kRecordCommit, 1);
    }
  }
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_EQ(execution.rejected_records, 2 * sites);
  EXPECT_EQ(execution.integrity_events, 2 * sites + 2);
  EXPECT_EQ(execution.issues.size(), kMaxIntegrityIssueExamples);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
}

TEST_F(ConSanAnalysis, Wave64HighLanesAndFullMasksRetainExactLanePairCounts) {
  test::manifest(capture_.header).target = 942;
  test::manifest(capture_.header).wave_size = 64;
  capture_.executions.clear();
  append_execution({{128, 1, 1}, {128, 1, 1}});
  access(0, 0, 1, 0, uint64_t{3} << 62);
  access(1, 1, 1, 0, uint64_t{1} << 63);
  FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_TRUE(result.value().coverage_complete);
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.cross_wave_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_lane_pairs, 2u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_EQ(execution.collision_lane_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_examples.front().first.mask, uint64_t{3} << 62);
  EXPECT_NE(render_analysis(result.value()).find("13835058055282163712"), std::string::npos);
  access(0, 0, 1, 0, UINT64_MAX);
  access(1, 1, 1, 0, UINT64_MAX);
  result = analyze();
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.value().executions.front().collision_lane_pairs, 2016u);
  EXPECT_EQ(result.value().executions.front().cross_wave_lane_pairs, 4096u);
}

TEST_F(ConSanAnalysis, MoreThanTwoWavesShareCandidatesAndPartialWaveGapsStayLocal) {
  capture_.executions.clear();
  append_execution({{97, 1, 1}, {97, 1, 1}});
  access(0, 0, 1);
  access(1, 1, 1);
  access(2, 1, 1);
  access(3, 1, 1, 0, 2); // Only lane zero exists in the final wave.
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.valid_observations, 3u);
  EXPECT_EQ(execution.rejected_records, 1u);
  EXPECT_EQ(execution.cross_wave_pairs, 2u);
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_EQ(execution.cross_wave_examples.back().second.wave, 2u);
}

TEST_F(ConSanAnalysis, Wave32CannotAttributeHighMaskBitsToExistingLanes) {
  access(0, 0, 1, 0, uint64_t{1} << 32);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.value().executions.front().rejected_records, 1u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_FALSE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, DecoderControlDisjointBoundaryContradictionIsAnIntegrityIssue) {
  access(0, 0, 1);
  access(1, 1, 1, 4);
  barrier();
  const auto baseline = analyze();
  ASSERT_TRUE(baseline.succeeded()) << error_.message();
  ASSERT_TRUE(baseline.value().coverage_complete);
  wire::write_u32(capture_.executions[0].report, completion(1) + wire::kSyncSequence, 1);
  const auto result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const auto &execution = result.value().executions[0];
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_GT(execution.integrity_events, 0u);
  EXPECT_EQ(execution.completed_barrier_waves, 1u);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
}

TEST_F(ConSanAnalysis, DecoderControlReadReadBoundaryContradictionIsAnIntegrityIssue) {
  test::manifest(capture_.header).sites[0].access = AccessKind::Read;
  access(0, 0, 1);
  access(1, 1, 1, 0);
  barrier();
  const auto baseline = analyze();
  ASSERT_TRUE(baseline.succeeded()) << error_.message();
  ASSERT_TRUE(baseline.value().coverage_complete);
  wire::write_u32(capture_.executions[0].report, completion(1) + wire::kSyncSequence, 1);
  const auto result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const auto &execution = result.value().executions[0];
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_GT(execution.integrity_events, 0u);
  EXPECT_EQ(execution.completed_barrier_waves, 1u);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
}

TEST_F(ConSanAnalysis, DecoderControlRepeatedStaticPhaseContradictionMustNotConfirmARace) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  Manifest &facts = test::manifest(capture_.header);
  facts.scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  facts.barriers.push_back({.original_drain = 48,
                            .original_signal = 52,
                            .original_wait = 56,
                            .prepared_signal = 400,
                            .prepared_wait = 404,
                            .completion_probe = 408,
                            .probe_size = 64});
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, 2);
    std::copy_n(report.begin() + completion(wave), kSyncRecordBytes,
                report.begin() + completion(wave, 1));
    wire::write_u32(report, completion(wave, 1) + wire::kSyncGroup, 1);
    wire::write_u32(report, completion(wave, 1) + wire::kSyncPhase, 2);
    wire::write_u32(report, completion(wave, 1) + wire::kSyncSequence, 1);
  }
  const FailureOr<LdsRaceResult> baseline = analyze();
  ASSERT_TRUE(baseline.succeeded()) << error_.message();
  ASSERT_EQ(baseline.value().executions[0].ordered_pairs, 1u);
  ASSERT_TRUE(baseline.value().coverage_complete);
  const std::vector<uint8_t> original = report;
  // A between-barriers read has an odd phase. Corrupt both its phase and the
  // corresponding sequence boundaries, so numeric agreement cannot mask the
  // static contradiction. Integrity validation also covers non-candidates.
  for (const uint32_t phase : {0u, 2u}) {
    for (const uint32_t mode : {0u, 1u, 2u}) {
      SCOPED_TRACE(phase);
      SCOPED_TRACE(mode);
      report = original;
      facts.sites[0].access = mode == 2 ? AccessKind::Read : AccessKind::Write;
      wire::write_u32(report, record(1, 1) + wire::kRecordLdsOffset, mode == 1 ? 4 : 0);
      wire::write_u32(report, record(1, 1) + wire::kRecordPhase, phase);
      wire::write_u32(report, completion(1) + wire::kSyncSequence, phase == 0 ? 1 : 0);
      wire::write_u32(report, completion(1, 1) + wire::kSyncSequence, phase == 0 ? 1 : 0);
      const FailureOr<LdsRaceResult> result = analyze();
      ASSERT_TRUE(result.succeeded()) << error_.message();
      EXPECT_EQ(result.value().cross_wave_pairs, 0u);
      EXPECT_EQ(result.value().ordering_unresolved_pairs, mode == 0 ? 1u : 0u);
      EXPECT_GT(result.value().executions[0].integrity_events, 0u);
      EXPECT_EQ(result.value().executions[0].completed_barrier_waves, 1u);
      EXPECT_FALSE(result.value().coverage_complete);
    }
  }
}

TEST_F(ConSanAnalysis, DecoderControlLaterJournalLossKeepsEarlierUnorderedPair) {
  many_barrier_sites(2, 2);
  test::manifest(capture_.header).scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  access(0, 0, 1);
  access(1, 1, 1);
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    wire::write_u32(report, completion(wave) + wire::kSyncSequence, 1);
  }
  const FailureOr<LdsRaceResult> baseline = analyze();
  ASSERT_TRUE(baseline.succeeded()) << error_.message();
  ASSERT_EQ(baseline.value().cross_wave_pairs, 1u);
  for (uint32_t wave = 0; wave < 2; ++wave) {
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, kSyncCapacity + 1);
    wire::write_u32(report, row(wave) + wire::kRowSyncLoss, 1);
    wire::write_u32(report, row(wave) + wire::kRowStatus, 7);
    for (uint32_t slot = 1; slot < kSyncCapacity; ++slot) {
      std::copy_n(report.begin() + completion(wave), kSyncRecordBytes,
                  report.begin() + completion(wave, slot));
      wire::write_u32(report, completion(wave, slot) + wire::kSyncOccurrence, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncPhase, slot + 1);
    }
  }
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().ordering_unresolved_pairs, 0u);
  EXPECT_FALSE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, DecoderControlUnorderedIntervalNeedsBothAdjacentOccurrences) {
  many_barrier_sites(2, 0);
  test::manifest(capture_.header).scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  access(0, 0, 1);
  access(1, 1, 1);
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, kSyncCapacity + 1);
    wire::write_u32(report, row(wave) + wire::kRowSyncLoss, 1);
    wire::write_u32(report, row(wave) + wire::kRowStatus, 7);
    for (uint32_t slot = 0; slot < kSyncCapacity; ++slot) {
      if (slot) {
        std::copy_n(report.begin() + completion(wave), kSyncRecordBytes,
                    report.begin() + completion(wave, slot));
      }
      wire::write_u32(report, completion(wave, slot) + wire::kSyncPhase, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncOccurrence, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncSequence, slot ? 1 : 0);
    }
  }
  const std::vector<uint8_t> original = report;
  // Both endpoints are in phase 1, between occurrences 1 and 2. A missing
  // unrelated later occurrence must not revoke this interval's proof.
  for (const uint32_t missing_slot : {0u, 1u, kSyncCapacity - 1, kSyncCapacity}) {
    SCOPED_TRACE(missing_slot);
    report = original;
    if (missing_slot < kSyncCapacity) {
      wire::write_u32(report, completion(1, missing_slot) + wire::kSyncCommit, 0);
    }
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_EQ(result.value().cross_wave_pairs, missing_slot < 2 ? 0u : 1u);
    EXPECT_EQ(result.value().ordering_unresolved_pairs, missing_slot < 2 ? 1u : 0u);
    EXPECT_FALSE(result.value().coverage_complete);
  }
}

TEST_F(ConSanAnalysis, DecoderControlRepeatedCycleEdgesRespectFinalPhaseAndZeroTrip) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  test::manifest(capture_.header).scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  std::vector<uint8_t> &report = capture_.executions[0].report;
  const std::vector<uint8_t> original = report;
  for (const uint32_t wave : {0u, 1u}) {
    SCOPED_TRACE(wave);
    report = original;
    // A prefix cannot run at the final positive phase and then skip its scalar
    // barrier. A suffix cannot run before the first completion of a nonempty loop.
    wire::write_u32(report, record(wave, wave) + wire::kRecordPhase, wave == 0 ? 1 : 0);
    wire::write_u32(report, completion(wave) + wire::kSyncSequence, wave == 0 ? 0 : 1);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_EQ(result.value().cross_wave_pairs, 0u);
    EXPECT_EQ(result.value().ordering_unresolved_pairs, 1u);
    EXPECT_GT(result.value().executions[0].integrity_events, 0u);
    EXPECT_FALSE(result.value().coverage_complete);
  }
  // With no iterations, preheader and post-loop sites may both run in phase 0.
  report = original;
  for (const uint32_t wave : {0u, 1u}) {
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, 0);
    wire::write_u32(report, record(wave, wave) + wire::kRecordPhase, 0);
    std::fill_n(report.begin() + completion(wave), kSyncRecordBytes, 0);
  }
  const FailureOr<LdsRaceResult> zero = analyze();
  ASSERT_TRUE(zero.succeeded()) << error_.message();
  EXPECT_EQ(zero.value().cross_wave_pairs, 1u);
  EXPECT_EQ(zero.value().ordering_unresolved_pairs, 0u);
  EXPECT_TRUE(zero.value().coverage_complete);
}

TEST_F(ConSanAnalysis, DecoderControlUnorderedIntervalRequiresTheEmptyAccessWave) {
  many_barrier_sites(2, 2, 3);
  test::manifest(capture_.header).scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  access(0, 0, 1);
  access(1, 1, 1);
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 3; ++wave) {
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, kSyncCapacity + 1);
    wire::write_u32(report, row(wave) + wire::kRowSyncLoss, 1);
    wire::write_u32(report, row(wave) + wire::kRowStatus, 7);
    for (uint32_t slot = 0; slot < kSyncCapacity; ++slot) {
      if (slot) {
        std::copy_n(report.begin() + completion(wave), kSyncRecordBytes,
                    report.begin() + completion(wave, slot));
      }
      wire::write_u32(report, completion(wave, slot) + wire::kSyncPhase, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncOccurrence, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncSequence, wave < 2 ? 1 : 0);
    }
  }
  const FailureOr<LdsRaceResult> complete = analyze();
  ASSERT_TRUE(complete.succeeded()) << error_.message();
  EXPECT_EQ(complete.value().cross_wave_pairs, 1u);
  EXPECT_EQ(complete.value().ordering_unresolved_pairs, 0u);
  wire::write_u32(report, completion(2) + wire::kSyncCommit, 0);
  const FailureOr<LdsRaceResult> missing = analyze();
  ASSERT_TRUE(missing.succeeded()) << error_.message();
  EXPECT_EQ(missing.value().cross_wave_pairs, 0u);
  EXPECT_EQ(missing.value().ordering_unresolved_pairs, 1u);
  EXPECT_FALSE(missing.value().coverage_complete);
}

TEST_F(ConSanAnalysis, LargeTruncatedLoopCountsKeepDistinctLatePhasesAggregated) {
  constexpr uint32_t sites = 1024, final_phase = sites + kSyncCapacity + 1;
  many_barrier_sites(sites, 0);
  test::manifest(capture_.header).scope = ScopeKind::NativeLdsWorkgroupBarrierLoop;
  std::vector<uint8_t> &report = capture_.executions[0].report;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    for (uint32_t site = 0; site < sites; ++site) {
      access(wave, site, site + 1);
      wire::write_u32(report, record(wave, site) + wire::kRecordPhase, kSyncCapacity + site + 1);
    }
    wire::write_u32(report, row(wave) + wire::kRowFinalPhase, final_phase);
    wire::write_u32(report, row(wave) + wire::kRowSyncLoss, final_phase - kSyncCapacity);
    wire::write_u32(report, row(wave) + wire::kRowStatus, 7);
    for (uint32_t slot = 1; slot < kSyncCapacity; ++slot) {
      std::copy_n(report.begin() + completion(wave), kSyncRecordBytes,
                  report.begin() + completion(wave, slot));
      wire::write_u32(report, completion(wave, slot) + wire::kSyncPhase, slot + 1);
      wire::write_u32(report, completion(wave, slot) + wire::kSyncOccurrence, slot + 1);
    }
  }
  for (const uint32_t examples : {0u, 1u, 7u}) {
    const FailureOr<LdsRaceResult> result = analyze({examples});
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const LdsRaceExecutionResult &execution = result.value().executions[0];
    EXPECT_EQ(execution.valid_observations, 2 * sites);
    EXPECT_EQ(execution.integrity_events, 0u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.ordered_pairs, 0u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, uint64_t{sites} * sites);
    EXPECT_EQ(execution.unresolved_examples.size(), examples);
    EXPECT_FALSE(execution.coverage_complete);
  }
}

TEST_F(ConSanAnalysis, SecondSelectedEntryContributesDualAndTransposeMetadata) {
  capture_.header = test::multi_entry_header();
  capture_.executions.clear();
  Manifest &second = *capture_.header.image.entries[1].selected;
  test::manifest(capture_.header).target = 1250;
  second.target = 1250;
  second.sites[1].width = 8;
  second.sites[1].lds_ranges.shape = LdsRangeShape::Transpose8;
  second.sites[0].lds_ranges = {LdsRangeShape::Dual, 3, 5};
  CapturedExecution execution = test::execution(second, 1);
  for (uint32_t wave = 0; wave < 2; ++wave) {
    wire::write_u32(execution.report, row(wave) + wire::kRowStatus,
                    wire::kRowEntered | wire::kRowExited);
  }
  capture_.executions.push_back(std::move(execution));
  access(0, 0, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_TRUE(result.value().has_dual_ranges);
  EXPECT_TRUE(result.value().has_transpose);
  ASSERT_EQ(result.value().executions.size(), 1u);
  EXPECT_EQ(result.value().executions.front().entry_key, second.entry.original_descriptor_offset);
  EXPECT_EQ(result.value().executions.front().valid_observations, 2u);
}

#if defined(CONSAN_ANALYZER_FIXTURE)
TEST_F(ConSanAnalysis, FreshCliDistinguishesAbsentExecutionsFromZeroObservationAnalysis) {
  std::string pattern = (std::filesystem::temp_directory_path() / "consan-lds-cli-XXXXXX").string();
  std::vector<char> temporary(pattern.begin(), pattern.end());
  temporary.push_back(0);
  ASSERT_NE(::mkdtemp(temporary.data()), nullptr);
  const std::filesystem::path directory = temporary.data();
  const std::filesystem::path capture_path = directory / "capture.bin";
  const std::filesystem::path output_path = directory / "analysis.json";
  const auto execute = [&](std::string_view performed, bool covered) {
    const FailureOr<std::vector<uint8_t>> encoded = encode();
    ASSERT_TRUE(encoded.succeeded()) << error_.message();
    std::ofstream file(capture_path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(encoded.value().data()), encoded.value().size());
    file.close();
    ASSERT_TRUE(file);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      const int output = ::open(output_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
      if (output < 0 || ::dup2(output, STDOUT_FILENO) < 0 || ::chdir(directory.c_str()) != 0)
        std::_Exit(90);
      ::close(output);
      ::execl(CONSAN_ANALYZER_FIXTURE, CONSAN_ANALYZER_FIXTURE, "capture.bin", nullptr);
      std::_Exit(91);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    std::ifstream output(output_path);
    const std::string rendered{std::istreambuf_iterator<char>(output),
                               std::istreambuf_iterator<char>()};
    EXPECT_NE(rendered.find(std::string("\"lds_analysis\":\"") + std::string(performed) + "\""),
              std::string::npos);
    EXPECT_NE(rendered.find(std::string("\"provided_entry_coverage_complete\":") +
                            (covered ? "true" : "false")),
              std::string::npos);
    if (covered)
      EXPECT_NE(rendered.find("\"valid_observations\":0"), std::string::npos);
    else
      EXPECT_NE(rendered.find("\"executions\":[]"), std::string::npos);
  };
  // An executed dispatch with no committed LDS observations still ran analysis.
  execute("performed", true);
  capture_.executions.clear();
  execute("not_performed", false);
  std::filesystem::remove_all(directory);
}
#endif

TEST_F(ConSanAnalysis, NativeNoProbesProofClosesInventoryGapWithoutInventingAnExecution) {
  ImageEntryFacts native;
  native.key = 192;
  native.original_sgprs = 16;
  native.original_vgprs = 8;
  native.binding = {.name = "no_lds_native",
                    .original_descriptor_offset = 192,
                    .prepared_descriptor_offset = 864,
                    .original_entry_offset = 96,
                    .prepared_entry_offset = 640,
                    .original_kernarg_bytes = 16,
                    .kernarg_alignment = 8,
                    .prepared_sgprs = 16,
                    .prepared_vgprs = 8};
  native.native_code = {{.original_offset = 96, .prepared_offset = 640, .bytes = 4}};
  capture_.header.image.entries.push_back(native);
  const FailureOr<LdsRaceResult> uncovered = analyze();
  ASSERT_TRUE(uncovered.succeeded()) << error_.message();
  EXPECT_TRUE(uncovered.value().coverage_complete);
  EXPECT_FALSE(uncovered.value().provided_entry_coverage_complete);

  NoProbesNeededProof proof = {.instruction_count = 1, .code_bytes = 4};
  proof.code_sha256[0] = 1;
  capture_.header.image.entries.back().no_probes_needed = proof;
  const FailureOr<LdsRaceResult> covered = analyze();
  ASSERT_TRUE(covered.succeeded()) << error_.message();
  EXPECT_TRUE(covered.value().coverage_complete);
  EXPECT_TRUE(covered.value().provided_entry_coverage_complete);
  EXPECT_EQ(covered.value().executions.size(), 1u);
  EXPECT_EQ(covered.value().entries.back().no_probes_needed, proof);
  const std::string rendered = render_analysis(covered.value());
  EXPECT_NE(rendered.find("\"disposition\":\"no_probes_needed\""), std::string::npos);
  EXPECT_NE(rendered.find("\"provided_entry_coverage_complete\":true"), std::string::npos);

  capture_.executions.clear();
  const FailureOr<LdsRaceResult> absent = analyze();
  ASSERT_TRUE(absent.succeeded()) << error_.message();
  EXPECT_FALSE(absent.value().coverage_complete);
  EXPECT_FALSE(absent.value().provided_entry_coverage_complete);
  EXPECT_TRUE(absent.value().executions.empty());
}

TEST_F(ConSanAnalysis, UnexecutedSelectedSiblingKeepsProvidedEntryCoverageIncomplete) {
  capture_.header = test::multi_entry_header();
  capture_.executions.clear();
  append_execution();
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_TRUE(result.value().coverage_complete);
  EXPECT_FALSE(result.value().provided_entry_coverage_complete);
  ASSERT_EQ(result.value().executions.size(), 1u);
}

TEST_F(ConSanAnalysis, SameImageEntriesUseTheirOwnFactsAndRepeatAttribution) {
  capture_.header = test::multi_entry_header();
  capture_.executions.clear();
  const std::array<EntryKey, 3> keys = {128, 192, 128};
  for (uint32_t index = 0; index < keys.size(); ++index) {
    const Manifest *facts = selected_manifest(capture_.header.image, keys[index]);
    ASSERT_NE(facts, nullptr);
    CapturedExecution execution = test::execution(*facts, index + 1, index + 1);
    for (uint32_t wave = 0; wave < 2; ++wave) {
      wire::write_u32(execution.report, row(wave) + wire::kRowStatus,
                      wire::kRowEntered | wire::kRowExited);
    }
    capture_.executions.push_back(std::move(execution));
    access(0, 0, 1, 0, 1, 0, index);
    access(1, 1, 1, 0, 1, 0, index);
  }
  util::StringDiagnostic error;
  CaptureEncoder encoder;
  const FailureOr<std::vector<uint8_t>> prefix = encoder.start(capture_.header, error.emitter());
  ASSERT_TRUE(prefix.succeeded()) << error.message();
  std::vector<uint8_t> bytes = prefix.value();
  for (const CapturedExecution &execution : capture_.executions) {
    const FailureOr<std::vector<uint8_t>> frame = encoder.append(execution, error.emitter());
    ASSERT_TRUE(frame.succeeded()) << error.message();
    bytes.insert(bytes.end(), frame.value().begin(), frame.value().end());
  }
  const FailureOr<std::vector<uint8_t>> end = encoder.finish(error.emitter());
  ASSERT_TRUE(end.succeeded()) << error.message();
  bytes.insert(bytes.end(), end.value().begin(), end.value().end());
  const FailureOr<Capture> decoded = decode_capture(bytes, error.emitter());
  ASSERT_TRUE(decoded.succeeded()) << error.message();
  const FailureOr<LdsRaceResult> result = analyze_capture(decoded.value(), {}, error.emitter());
  ASSERT_TRUE(result.succeeded()) << error.message();
  ASSERT_EQ(result.value().executions.size(), 3u);
  EXPECT_EQ(result.value().cross_wave_pairs, 2u);
  EXPECT_TRUE(result.value().coverage_complete);
  EXPECT_EQ(result.value().executions[0].cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().executions[1].cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions[2].cross_wave_pairs, 1u);
  for (uint32_t index = 0; index < keys.size(); ++index) {
    EXPECT_EQ(result.value().executions[index].entry_key, keys[index]);
    EXPECT_EQ(result.value().executions[index].valid_observations, 2u);
  }
  EXPECT_EQ(result.value().executions[1].entry, "second_codec_fixture");
  capture_.executions[0].entry_key = 999;
  EXPECT_TRUE(analyze_typed(capture_, {}, error.emitter()).failed());
  capture_.executions[0].entry_key = 192;
  EXPECT_TRUE(analyze_typed(capture_, {}, error.emitter()).failed());
}

} // namespace
} // namespace rocjitsu::consan
