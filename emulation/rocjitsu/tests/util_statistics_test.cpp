// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/statistics.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

TEST(StatCounterTest, SaturatesAndRetainsOverflowAcrossMerges) {
  stats::Counter counter;
  counter.add(std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(counter.value(), std::numeric_limits<uint64_t>::max());
  EXPECT_FALSE(counter.overflowed());
  counter.add(0);
  EXPECT_FALSE(counter.overflowed());
  counter.increment();
  EXPECT_EQ(counter.value(), std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(counter.overflowed());

  stats::Counter merged;
  merged.merge(counter);
  EXPECT_EQ(merged.snapshot().value, std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(merged.snapshot().overflowed);
}

TEST(StatCounterTest, MergesIndependentCountsAndSnapshotsByValue) {
  stats::Counter left;
  stats::Counter right;
  left.add(17);
  right.add(25);
  const stats::CounterSnapshot before = left.snapshot();
  left.merge(right);
  EXPECT_EQ(before.value, 17u);
  EXPECT_EQ(left.value(), 42u);
  EXPECT_EQ(right.value(), 25u);
  EXPECT_FALSE(left.overflowed());
}

TEST(StatGaugeTest, DistinguishesAnUnsetGaugeFromZero) {
  stats::Gauge gauge;
  EXPECT_FALSE(gauge.snapshot().available);
  gauge.set(0);
  EXPECT_TRUE(gauge.available());
  EXPECT_EQ(gauge.value(), 0);
  gauge.set(-7);
  EXPECT_EQ(gauge.snapshot().value, -7);
}

TEST(StatSummaryTest, ComputesKnownPopulationMoments) {
  stats::Summary summary;
  EXPECT_EQ(summary.snapshot().count, 0u);
  for (double value : {2., 4., 4., 4., 5., 5., 7., 9.})
    summary.record(value);
  const stats::SummarySnapshot result = summary.snapshot();
  EXPECT_EQ(result.count, 8u);
  EXPECT_DOUBLE_EQ(result.sum, 40.);
  EXPECT_DOUBLE_EQ(result.minimum, 2.);
  EXPECT_DOUBLE_EQ(result.maximum, 9.);
  EXPECT_DOUBLE_EQ(result.mean, 5.);
  EXPECT_DOUBLE_EQ(result.variance, 4.);
  EXPECT_FALSE(result.invalid);
}

TEST(StatSummaryTest, MergesUnequalShardsWithEmptyInputs) {
  stats::Summary left;
  stats::Summary right;
  for (double value : {2., 4., 4.})
    left.record(value);
  for (double value : {4., 5., 5., 7., 9.})
    right.record(value);
  left.merge(stats::Summary{});
  left.merge(right);
  stats::Summary copied;
  copied.merge(left);
  const stats::SummarySnapshot result = copied.snapshot();
  EXPECT_EQ(result.count, 8u);
  EXPECT_DOUBLE_EQ(result.sum, 40.);
  EXPECT_DOUBLE_EQ(result.minimum, 2.);
  EXPECT_DOUBLE_EQ(result.maximum, 9.);
  EXPECT_DOUBLE_EQ(result.mean, 5.);
  EXPECT_NEAR(result.variance, 4., 1e-12);
  EXPECT_FALSE(result.invalid);
}

TEST(StatSummaryTest, MergesWidelySeparatedSamplesWithoutIntermediateOverflow) {
  for (double first : {-7.5e153, 7.5e153}) {
    SCOPED_TRACE(first);
    stats::Summary sequential;
    sequential.record(first);
    sequential.record(-first);
    const stats::SummarySnapshot expected = sequential.snapshot();
    EXPECT_FALSE(expected.invalid);
    EXPECT_DOUBLE_EQ(expected.variance, 5.625e307);

    stats::Summary left;
    stats::Summary right;
    left.record(first);
    right.record(-first);
    left.merge(right);
    const stats::SummarySnapshot result = left.snapshot();
    EXPECT_EQ(result.count, 2u);
    EXPECT_FALSE(result.invalid);
    EXPECT_TRUE(std::isfinite(result.variance));
    EXPECT_DOUBLE_EQ(result.variance, 5.625e307);
    EXPECT_DOUBLE_EQ(result.variance, expected.variance);
  }
}

TEST(StatSummaryTest, AvoidsCancellationInVarianceForNearbyLargeValues) {
  stats::Summary summary;
  for (double value : {1e12 + 1, 1e12 + 2, 1e12 + 3})
    summary.record(value);
  EXPECT_DOUBLE_EQ(summary.snapshot().mean, 1e12 + 2);
  EXPECT_NEAR(summary.snapshot().variance, 2. / 3., 1e-12);
}

TEST(StatSummaryTest, RejectsNonfiniteSamplesWithoutPublishingNonfiniteValues) {
  stats::Summary summary;
  summary.record(std::numeric_limits<double>::quiet_NaN());
  summary.record(std::numeric_limits<double>::infinity());
  summary.record(-std::numeric_limits<double>::infinity());
  summary.record(8);
  EXPECT_EQ(summary.count(), 1u);
  EXPECT_TRUE(summary.invalid());
  EXPECT_DOUBLE_EQ(summary.sum(), 8.);
  stats::Summary destination;
  destination.merge(summary);
  EXPECT_TRUE(destination.snapshot().invalid);
  EXPECT_EQ(destination.snapshot().count, 1u);
}

TEST(StatSummaryTest, MarksArithmeticAndCountOverflowWithoutWrapping) {
  stats::Summary summary;
  summary.record(std::numeric_limits<double>::max());
  summary.record(std::numeric_limits<double>::max());
  const stats::SummarySnapshot result = summary.snapshot();
  EXPECT_EQ(result.count, 2u);
  EXPECT_TRUE(result.invalid);
  EXPECT_FALSE(result.count_overflowed);
  EXPECT_TRUE(std::isfinite(result.sum));
  EXPECT_TRUE(std::isfinite(result.mean));
  EXPECT_TRUE(std::isfinite(result.variance));

  stats::Summary count_overflow;
  count_overflow.record(0);
  for (unsigned iteration = 0; iteration < 63; ++iteration)
    count_overflow.merge(count_overflow);
  EXPECT_EQ(count_overflow.count(), uint64_t{1} << 63);
  EXPECT_FALSE(count_overflow.invalid());
  EXPECT_FALSE(count_overflow.snapshot().count_overflowed);
  count_overflow.merge(count_overflow);
  EXPECT_EQ(count_overflow.count(), std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(count_overflow.invalid());
  EXPECT_TRUE(count_overflow.snapshot().count_overflowed);
  stats::Summary copied;
  copied.merge(count_overflow);
  EXPECT_TRUE(copied.snapshot().count_overflowed);
  count_overflow.record(0);
  EXPECT_EQ(count_overflow.count(), std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(count_overflow.snapshot().count_overflowed);
}

TEST(StatHistogramTest, RecordsInclusiveBoundariesAndAnUnboundedFinalBucket) {
  const std::array bounds{0., 4., 8.};
  util::FailureOr<stats::Histogram> histogram = stats::Histogram::create(bounds);
  ASSERT_TRUE(histogram.succeeded());
  for (double value : {-2., 0., 1., 4., 7., 8., 12.})
    histogram.value().record(value);
  const stats::HistogramSnapshot result = histogram.value().snapshot();
  EXPECT_EQ(result.upper_bounds, (std::vector<double>{0., 4., 8.}));
  EXPECT_EQ(result.counts, (std::vector<uint64_t>{2, 2, 2, 1}));
  EXPECT_EQ(result.summary.count, 7u);
  EXPECT_DOUBLE_EQ(result.summary.sum, 30.);
  EXPECT_FALSE(result.summary.invalid);
  EXPECT_FALSE(result.overflowed);
  histogram.value().record(100);
  EXPECT_EQ(result.counts.back(), 1u);
}

TEST(StatHistogramTest, ValidatesBoundsBeforeAllocatingARecorder) {
  util::StringDiagnostic diagnostic;
  for (const std::vector<double> &bounds :
       {std::vector<double>{1., 1.}, std::vector<double>{2., 1.},
        std::vector<double>{std::numeric_limits<double>::infinity()},
        std::vector<double>{std::numeric_limits<double>::quiet_NaN()}}) {
    EXPECT_TRUE(stats::Histogram::create(bounds, diagnostic.emitter()).failed());
    EXPECT_NE(diagnostic.message().find("strictly increasing"), std::string::npos);
  }
  util::FailureOr<stats::Histogram> unbounded = stats::Histogram::create({});
  ASSERT_TRUE(unbounded.succeeded());
  unbounded.value().record(-19);
  unbounded.value().record(19);
  EXPECT_EQ(unbounded.value().snapshot().counts, (std::vector<uint64_t>{2}));
}

TEST(StatHistogramTest, RejectsNonfiniteSamplesAndPreservesFiniteBucketCounts) {
  const std::array bounds{4.};
  util::FailureOr<stats::Histogram> histogram = stats::Histogram::create(bounds);
  ASSERT_TRUE(histogram.succeeded());
  histogram.value().record(std::numeric_limits<double>::infinity());
  histogram.value().record(std::numeric_limits<double>::quiet_NaN());
  histogram.value().record(2);
  const stats::HistogramSnapshot result = histogram.value().snapshot();
  EXPECT_EQ(result.counts, (std::vector<uint64_t>{1, 0}));
  EXPECT_EQ(result.summary.count, 1u);
  EXPECT_TRUE(result.summary.invalid);
}

TEST(StatHistogramTest, MergesOnlyCompatibleLayoutsWithoutPartialChanges) {
  const std::array bounds{2., 4.};
  const std::array other_bounds{2., 5.};
  util::FailureOr<stats::Histogram> left = stats::Histogram::create(bounds);
  util::FailureOr<stats::Histogram> right = stats::Histogram::create(bounds);
  util::FailureOr<stats::Histogram> incompatible = stats::Histogram::create(other_bounds);
  ASSERT_TRUE(left.succeeded());
  ASSERT_TRUE(right.succeeded());
  ASSERT_TRUE(incompatible.succeeded());
  left.value().record(1);
  right.value().record(3);
  right.value().record(8);
  incompatible.value().record(2);
  ASSERT_TRUE(left.value().merge(right.value()).succeeded());
  EXPECT_EQ(left.value().snapshot().counts, (std::vector<uint64_t>{1, 1, 1}));
  EXPECT_TRUE(left.value().merge(incompatible.value()).failed());
  EXPECT_EQ(left.value().snapshot().counts, (std::vector<uint64_t>{1, 1, 1}));
  EXPECT_EQ(left.value().snapshot().summary.count, 3u);
  EXPECT_DOUBLE_EQ(left.value().snapshot().summary.sum, 12.);
}

TEST(StatHistogramTest, SaturatesBucketCountsAndMarksTheSummaryInvalid) {
  util::FailureOr<stats::Histogram> histogram = stats::Histogram::create({});
  ASSERT_TRUE(histogram.succeeded());
  histogram.value().record(0);
  for (unsigned iteration = 0; iteration < 64; ++iteration)
    ASSERT_TRUE(histogram.value().merge(histogram.value()).succeeded());
  const stats::HistogramSnapshot result = histogram.value().snapshot();
  EXPECT_EQ(result.counts, (std::vector<uint64_t>{std::numeric_limits<uint64_t>::max()}));
  EXPECT_EQ(result.summary.count, std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(result.overflowed);
  EXPECT_TRUE(result.summary.invalid);
}

} // namespace
