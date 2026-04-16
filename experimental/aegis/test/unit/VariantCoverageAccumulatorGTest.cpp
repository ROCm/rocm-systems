//===-- VariantCoverageAccumulatorGTest.cpp --------------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for `VariantCoverageAccumulator`: empty state, overlapping
/// adds, plateau (`lastAdded == 0`), and the covered-set snapshot that the
/// variant-build loop feeds back as `ExcludedPCs`.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/VariantCoverageAccumulator.h"
#include <gtest/gtest.h>

using namespace aegisbit;

TEST(VariantCoverageAccumulator, EmptyStartsEmpty) {
  VariantCoverageAccumulator A;
  EXPECT_EQ(A.coveredCount(), 0u);
  EXPECT_EQ(A.lastAdded(), 0u);
  EXPECT_TRUE(A.needsCoverage(0x1000));
}

TEST(VariantCoverageAccumulator, MarksCoveredAddsNewPCs) {
  VariantCoverageAccumulator A;
  uint64_t PCs[] = {0x1000, 0x1004, 0x1008};
  EXPECT_EQ(A.markCovered(PCs), 3u);
  EXPECT_EQ(A.coveredCount(), 3u);
  EXPECT_EQ(A.lastAdded(), 3u);
  EXPECT_FALSE(A.needsCoverage(0x1000));
  EXPECT_TRUE(A.needsCoverage(0x1100));
}

TEST(VariantCoverageAccumulator, OverlappingAddsCountOnlyNew) {
  VariantCoverageAccumulator A;
  uint64_t V0[] = {0x1000, 0x1004, 0x1008};
  A.markCovered(V0);

  // Variant 1 overlaps PC 0x1004 with V0 but adds two fresh PCs.
  uint64_t V1[] = {0x1004, 0x2000, 0x2004};
  std::size_t Added = A.markCovered(V1);
  EXPECT_EQ(Added, 2u);
  EXPECT_EQ(A.coveredCount(), 5u);
  EXPECT_EQ(A.lastAdded(), 2u);
}

TEST(VariantCoverageAccumulator, PlateauReturnsZero) {
  VariantCoverageAccumulator A;
  uint64_t V0[] = {0x1000, 0x1004};
  A.markCovered(V0);

  // Variant 1 is a pure subset of V0 — nothing new.
  uint64_t V1[] = {0x1000, 0x1004};
  std::size_t Added = A.markCovered(V1);
  EXPECT_EQ(Added, 0u);
  EXPECT_EQ(A.lastAdded(), 0u);
  EXPECT_EQ(A.coveredCount(), 2u);
}

TEST(VariantCoverageAccumulator, CoveredSnapshotMatches) {
  VariantCoverageAccumulator A;
  uint64_t V0[] = {0x1000, 0x1004, 0x1008};
  A.markCovered(V0);
  const auto &Set = A.covered();
  EXPECT_TRUE(Set.count(0x1000));
  EXPECT_TRUE(Set.count(0x1004));
  EXPECT_TRUE(Set.count(0x1008));
  EXPECT_FALSE(Set.count(0x200c));
  EXPECT_EQ(Set.size(), 3u);
}

TEST(VariantCoverageAccumulator, RepeatedPCsInSameVariantCollapse) {
  VariantCoverageAccumulator A;
  // Defensive: a malformed caller that feeds the same PC twice in one
  // variant should not double-count it.
  uint64_t V[] = {0x2000, 0x2000, 0x2004};
  EXPECT_EQ(A.markCovered(V), 2u);
}
