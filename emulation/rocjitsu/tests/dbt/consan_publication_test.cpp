// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
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
} // namespace
} // namespace rocjitsu::consan::hook
