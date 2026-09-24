// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/hooks/consan/rj_hsa_dbi_conflict_analysis.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_publication.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <vector>

namespace rocjitsu::consan::hook {
namespace {
PublicationPoint point(uint32_t owner, uint64_t sequence) {
  return {.domain = {.generation = 7, .dispatch = 19}, .owner = owner, .sequence = sequence};
}
PublicationEvent rmw(uint32_t owner, uint64_t seq, uint64_t old, uint64_t next, bool release = true,
                     bool acquire = true, uint64_t address = 0x1000) {
  return {.point = point(owner, seq),
          .address = address,
          .bytes = 4,
          .observed = old,
          .written = next,
          .release = release,
          .acquire = acquire,
          .covers_workgroup = true,
          .observation_valid = true};
}
using Result = PublicationOrdering;
TEST(ConSanPublicationTest, OnePublicationOrdersEveryPrecedingWriteAndFollowingRead) {
  for (uint32_t producer : {0u, 1u}) {
    uint32_t consumer = producer ^ 1;
    std::array events{rmw(producer, 20, 0, 1), rmw(consumer, 20, 1, 2)};
    for (uint64_t write = 1; write <= 8; ++write)
      for (uint64_t read = 21; read <= 28; ++read)
        EXPECT_EQ(publication_orders(point(producer, write), point(consumer, read), events, true),
                  Result::Ordered);
    std::reverse(events.begin(), events.end());
    EXPECT_EQ(publication_orders(point(producer, 1), point(consumer, 30), events, true),
              Result::Ordered);
  }
}
TEST(ConSanPublicationTest, TreeReleaseSequenceIncludesAllProducersAndRelaxedRmw) {
  // Bits 1,2,4 are published by separate producers, then consumer sets bit 8.
  std::array events{rmw(0, 10, 0, 1, true, false), rmw(1, 10, 1, 3, true, false),
                    rmw(2, 10, 3, 7, true, false), rmw(3, 10, 7, 15)};
  for (uint32_t producer = 0; producer < 3; ++producer)
    EXPECT_EQ(publication_orders(point(producer, 1), point(3, 20), events, true), Result::Ordered);
  events[1].release = false;
  EXPECT_EQ(publication_orders(point(0, 1), point(3, 20), events, true), Result::Ordered);
  EXPECT_EQ(publication_orders(point(1, 1), point(3, 20), events, true), Result::Unordered);
}
TEST(ConSanPublicationTest, AcquiredPublicationCanBeForwardedThroughAnotherObject) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 10, 1, 2), rmw(1, 20, 0, 1, true, false, 0x2000),
                    rmw(2, 10, 1, 2, false, true, 0x2000)};
  EXPECT_EQ(publication_orders(point(0, 1), point(2, 20), events, true), Result::Ordered);
  events[1].acquire = false;
  EXPECT_EQ(publication_orders(point(0, 1), point(2, 20), events, true), Result::Unordered);
}
TEST(ConSanPublicationTest, UnrelatedBookkeepingDoesNotOverwriteApplicationPublication) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 10, 1, 2), rmw(0, 11, 0, 1, true, true, 0x2000),
                    rmw(1, 11, 0, 1, true, true, 0x3000)};
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Ordered);
}
TEST(ConSanPublicationTest, RelaxedAndNarrowScopeDoNotSynchronizeWaves) {
  std::array events{rmw(0, 10, 0, 1, false, false), rmw(1, 10, 1, 2, false, false)};
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Unordered);
  events[0].release = true;
  events[1].acquire = true;
  events[0].covers_workgroup = false;
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Unordered);
}
TEST(ConSanPublicationTest, AcquireOfInitialValueDoesNotObserveFutureRelease) {
  auto initial = rmw(1, 10, 0, 0, false, true);
  initial.operation = PublicationOperation::Read;
  std::array events{rmw(0, 10, 0, 1), initial};
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Unordered);
  events[1].observed = 1;
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Ordered);
}
TEST(ConSanPublicationTest, FailedCasCanAcquireButNeverPublishesRelease) {
  auto failed = rmw(1, 10, 1, 0, false, true);
  failed.operation = PublicationOperation::Read;
  std::array events{rmw(0, 10, 0, 1), failed};
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Ordered);
  events[1].release = true;
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Incomplete);
}
TEST(ConSanPublicationTest, ProgramOrderPreventsRetroactivePublication) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 10, 1, 2)};
  EXPECT_EQ(publication_orders(point(0, 11), point(1, 20), events, true), Result::Unordered);
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 9), events, true), Result::Unordered);
}
TEST(ConSanPublicationTest, RepeatedValuesAndAbaAreIncomplete) {
  for (auto events : {std::vector{rmw(0, 10, 0, 1), rmw(1, 10, 1, 1)},
                      std::vector{rmw(0, 10, 0, 1), rmw(1, 10, 1, 0)},
                      std::vector{rmw(0, 10, 0, 1), rmw(1, 10, 0, 1)}})
    EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Incomplete);
}
TEST(ConSanPublicationTest, MissingEventsAndObservationsAreIncomplete) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 10, 2, 3)};
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Incomplete);
  events[1].observed = 1;
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, false), Result::Incomplete);
  events[0].observation_valid = false;
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Incomplete);
}
TEST(ConSanPublicationTest, ReversedSameOwnerModificationOrderIsIncomplete) {
  std::array events{rmw(0, 20, 0, 1, false, false), rmw(0, 10, 1, 2, false, false)};
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 30), events, true), Result::Incomplete);
}
TEST(ConSanPublicationTest, NarrowRmwDoesNotCarryCrossWaveReleaseSequence) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 10, 1, 2, false, false), rmw(2, 10, 2, 3)};
  events[1].covers_workgroup = false;
  EXPECT_EQ(publication_orders(point(0, 1), point(2, 20), events, true), Result::Unordered);
}
TEST(ConSanPublicationTest, IterationsUseDynamicSequenceAndObservedTransition) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 10, 1, 2), rmw(0, 30, 2, 3), rmw(1, 30, 3, 4)};
  EXPECT_EQ(publication_orders(point(0, 20), point(1, 20), events, true), Result::Unordered);
  EXPECT_EQ(publication_orders(point(0, 20), point(1, 40), events, true), Result::Ordered);
}
TEST(ConSanPublicationTest, UnknownReadValueIsIncomplete) {
  auto read = rmw(1, 10, 9, 0, false, true);
  read.operation = PublicationOperation::Read;
  std::array events{rmw(0, 10, 0, 1), read};
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Incomplete);
}
TEST(ConSanPublicationTest, DomainsDoNotJoin) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 10, 1, 2)};
  const auto before = point(0, 1);
  for (unsigned field = 0; field < 6; ++field) {
    auto after = point(1, 20);
    if (field == 0)
      ++after.domain.generation;
    if (field == 1)
      ++after.domain.dispatch;
    if (field == 2)
      ++after.domain.workgroup_x;
    if (field == 3)
      ++after.domain.workgroup_y;
    if (field == 4)
      ++after.domain.workgroup_z;
    if (field == 5)
      ++after.domain.cluster_workgroup;
    EXPECT_EQ(publication_orders(before, after, events, true), Result::Unordered);
  }
}
TEST(ConSanPublicationTest, OverlappingAtomicObjectsAreIncomplete) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 10, 1, 2, true, true, 0x1002)};
  EXPECT_EQ(publication_orders(point(0, 1), point(1, 20), events, true), Result::Incomplete);
}
TEST(ConSanPublicationTest, ProgramOrderDoesNotCrossLanesOfOneWave) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 20, 1, 2)};
  auto before = point(0, 1);
  auto after = point(1, 30);
  EXPECT_EQ(publication_orders(before, after, events, true), Result::Ordered);
  before.lane = 1;
  EXPECT_EQ(publication_orders(before, after, events, true), Result::Unordered);
  before.lane = 0;
  after.lane = 1;
  EXPECT_EQ(publication_orders(before, after, events, true), Result::Unordered);
  after.lane = 0;
  events[0].point.lane = 1;
  EXPECT_EQ(publication_orders(before, after, events, true), Result::Unordered);
}
TEST(ConSanPublicationTest, TransitivePublicationRequiresSameIntermediateLane) {
  std::array events{rmw(0, 10, 0, 1), rmw(1, 20, 1, 2), rmw(1, 30, 0, 1, true, true, 0x2000),
                    rmw(2, 40, 1, 2, true, true, 0x2000)};
  EXPECT_EQ(publication_orders(point(0, 1), point(2, 50), events, true), Result::Ordered);
  events[2].point.lane = 1;
  EXPECT_EQ(publication_orders(point(0, 1), point(2, 50), events, true), Result::Unordered);
}
TEST(ConSanPublicationTest, ConflictAnalysisUsesObservationProofAndFailsClosed) {
  Evidence writer;
  writer.generation = 7;
  writer.dispatch_id = 19;
  writer.entry.valid = true;
  writer.entry.kind = ShadowAccessKind::Write;
  writer.entry.byte_count = 4;
  writer.entry.owner_id = 0;
  writer.publication_sequence = 1;
  writer.exact_lane_mask = 1;
  Evidence reader = writer;
  reader.entry.kind = ShadowAccessKind::Read;
  reader.entry.owner_id = 1;
  reader.publication_sequence = 30;
  std::array accesses{writer, reader};
  DecodedPublications publications{.status = PublicationDecodeStatus::Complete,
                                   .events = {rmw(0, 10, 0, 1), rmw(1, 20, 1, 2)}};
  const auto ordered = analyze_conflicts(accesses, true, 8, false, &publications);
  EXPECT_EQ(ordered.conflict_count, 0);
  EXPECT_EQ(ordered.ordered_publication_pairs, 1);
  for (const uint64_t lanes : {uint64_t{0}, uint64_t{3}}) {
    accesses[1].exact_lane_mask = lanes;
    const auto missing_lane = analyze_conflicts(accesses, true, 8, false, &publications);
    EXPECT_EQ(missing_lane.conflict_count, 1);
    EXPECT_EQ(missing_lane.incomplete_publication_pairs, 1);
  }
  accesses[1].exact_lane_mask = 2;
  EXPECT_EQ(analyze_conflicts(accesses, true, 8, false, &publications).conflict_count, 1);
  accesses[1].exact_lane_mask = 1;
  publications.events[1].acquire = false;
  EXPECT_EQ(analyze_conflicts(accesses, true, 8, false, &publications).conflict_count, 1);
  publications.events[1].acquire = true;
  publications.status = PublicationDecodeStatus::Incomplete;
  const auto incomplete = analyze_conflicts(accesses, true, 8, false, &publications);
  EXPECT_EQ(incomplete.conflict_count, 1);
  EXPECT_EQ(incomplete.incomplete_publication_pairs, 1);
  ReportSummary summary;
  accumulate_analysis(summary, incomplete);
  EXPECT_FALSE(evaluate_report_trust(summary, false).dynamic_complete);
  publications.status = PublicationDecodeStatus::Complete;
  EXPECT_EQ(analyze_conflicts(accesses, false, 8, false, &publications).conflict_count, 1);
}

PublicationRecord record(uint32_t owner, uint64_t sequence, uint64_t observed, uint64_t written) {
  return {.generation = 7,
          .dispatch_id = 19,
          .sequence = sequence,
          .address = 0x1000,
          .observed = observed,
          .written = written,
          .owner_id = owner,
          .byte_count = 4,
          .roles = kPublicationRelease | kPublicationAcquire | kPublicationObserved,
          .scope = 3,
          .operation = PublicationRecordOperation::Rmw,
          .state = kPublicationReady};
}
ReportHeader publication_header() {
  ReportHeader header;
  header.generation = 7;
  header.dispatch_id = 19;
  header.publication_clock = 30;
  header.publication_event_capacity = 2;
  header.publication_event_count = 2;
  header.publication_flags = kPublicationTraceEnabled | kPublicationTraceComplete;
  return header;
}
TEST(ConSanPublicationTest, DecodeRetainsGuestObservationAndIdentity) {
  std::array records{record(0, 10, 0, 1), record(1, 20, 1, 2)};
  records[0].workgroup_x = records[1].workgroup_x = 13;
  records[0].cluster_workgroup_id = records[1].cluster_workgroup_id = 4;
  const auto decoded = decode_publications(publication_header(), records);
  ASSERT_EQ(decoded.status, PublicationDecodeStatus::Complete);
  ASSERT_EQ(decoded.events.size(), 2);
  EXPECT_EQ(decoded.events[0].point.domain.workgroup_x, 13);
  EXPECT_EQ(decoded.events[0].point.domain.cluster_workgroup, 4);
  auto before = point(0, 1), after = point(1, 30);
  before.domain.workgroup_x = after.domain.workgroup_x = 13;
  before.domain.cluster_workgroup = after.domain.cluster_workgroup = 4;
  EXPECT_EQ(publication_orders(before, after, decoded.events, true), Result::Ordered);
}
TEST(ConSanPublicationTest, DecodeRejectsStalePartialAndMissingObservations) {
  for (unsigned defect = 0; defect < 12; ++defect) {
    SCOPED_TRACE(defect);
    std::array records{record(0, 10, 0, 1), record(1, 20, 1, 2)};
    auto &r = records[0];
    if (defect == 0)
      ++r.generation;
    if (defect == 1)
      ++r.dispatch_id;
    if (defect == 2)
      r.state = 0;
    if (defect == 3)
      r.sequence = 0;
    if (defect == 4)
      r.sequence = 31;
    if (defect == 5)
      r.sequence = records[1].sequence;
    if (defect == 6)
      r.roles &= ~kPublicationObserved;
    if (defect == 7)
      r.roles |= 8;
    if (defect == 8)
      r.owner_id = 32;
    if (defect == 9)
      r.lane_id = 64;
    if (defect == 10)
      r.byte_count = 3;
    if (defect == 11)
      r.observed = uint64_t{1} << 32;
    const auto decoded = decode_publications(publication_header(), records);
    EXPECT_NE(decoded.status, PublicationDecodeStatus::Complete);
    EXPECT_TRUE(decoded.events.empty());
  }
}
TEST(ConSanPublicationTest, DecodeRequiresCompleteBoundedTrace) {
  std::array records{record(0, 10, 0, 1), record(1, 20, 1, 2)};
  for (unsigned defect = 0; defect < 5; ++defect) {
    auto header = publication_header();
    if (defect == 0)
      header.publication_dropped_count = 1;
    if (defect == 1)
      header.publication_event_count = 3;
    if (defect == 2)
      header.publication_event_capacity = 3;
    if (defect == 3)
      header.publication_flags &= ~kPublicationTraceComplete;
    if (defect == 4)
      header.publication_flags |= 4;
    EXPECT_NE(decode_publications(header, records).status, PublicationDecodeStatus::Complete);
  }
  EXPECT_EQ(decode_publications(ReportHeader{}, {}).status, PublicationDecodeStatus::Disabled);
  ReportHeader disabled;
  disabled.publication_clock = 123;
  const auto decoded = decode_publications(disabled, {});
  EXPECT_EQ(decoded.status, PublicationDecodeStatus::Disabled);
  EXPECT_TRUE(decoded.events.empty());
  auto header = publication_header();
  header.publication_flags = 0;
  EXPECT_EQ(decode_publications(header, records).status, PublicationDecodeStatus::Malformed);
}
} // namespace
} // namespace rocjitsu::consan::hook
