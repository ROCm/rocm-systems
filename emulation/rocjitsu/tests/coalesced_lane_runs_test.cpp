// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/coalesced_lane_runs.h"
#include "rocjitsu/vm/amdgpu/strided_lane_addresses.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace {

using namespace rocjitsu;

using LaneRuns = std::vector<amdgpu::CoalescedLaneRun>;

LaneRuns coalesced_runs(std::span<const uint64_t> addrs, uint64_t lane_mask, uint32_t stride,
                        size_t stop_after = SIZE_MAX, bool *completed = nullptr) {
  LaneRuns runs;
  const bool done = amdgpu::for_each_coalesced_lane_run(
      addrs.data(), lane_mask, static_cast<uint32_t>(addrs.size()), stride,
      amdgpu::MemoryTransactionPolicy{}, [&](const amdgpu::CoalescedLaneRun &run) {
        runs.push_back(run);
        return runs.size() < stop_after;
      });
  if (completed)
    *completed = done;
  return runs;
}

LaneRuns lane_ordered_runs(std::span<const uint64_t> addrs, uint64_t lane_mask, uint32_t stride) {
  LaneRuns runs;
  for (uint32_t lane = 0; lane < addrs.size(); ++lane) {
    if (!(lane_mask >> lane & 1))
      continue;
    const bool extends =
        !runs.empty() && lane == runs.back().first_lane + runs.back().size / stride &&
        addrs[lane - 1] <= UINT64_MAX - stride && addrs[lane] == addrs[lane - 1] + stride;
    if (extends) {
      runs.back().lane_mask |= uint64_t{1} << lane;
      runs.back().size += stride;
    } else {
      runs.push_back({.address = addrs[lane],
                      .lane_mask = uint64_t{1} << lane,
                      .first_lane = lane,
                      .size = stride,
                      .lane_ordered = true});
    }
  }
  return runs;
}

bool operands_overlap_or_wrap(std::span<const uint64_t> addrs, uint64_t lane_mask,
                              uint32_t stride) {
  for (uint32_t lane_a = 0; lane_a < addrs.size(); ++lane_a) {
    if (!(lane_mask >> lane_a & 1))
      continue;
    if (addrs[lane_a] > UINT64_MAX - (stride - 1))
      return true;
    for (uint32_t lane_b = lane_a + 1; lane_b < addrs.size(); ++lane_b) {
      if ((lane_mask >> lane_b & 1) &&
          (addrs[lane_a] > addrs[lane_b] ? addrs[lane_a] - addrs[lane_b]
                                         : addrs[lane_b] - addrs[lane_a]) < stride)
        return true;
    }
  }
  return false;
}

void expect_same_runs(const LaneRuns &actual, const LaneRuns &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i].address, expected[i].address);
    EXPECT_EQ(actual[i].lane_mask, expected[i].lane_mask);
    EXPECT_EQ(actual[i].first_lane, expected[i].first_lane);
    EXPECT_EQ(actual[i].size, expected[i].size);
    EXPECT_EQ(actual[i].lane_ordered, expected[i].lane_ordered);
  }
}

TEST(CoalescedLaneRunsTest, LaneOrderedScanMatchesMaximalRuns) {
  constexpr uint32_t kStride = 16;
  for (const uint32_t wf_size : {32u, 64u}) {
    const uint64_t full = wf_size == 64 ? UINT64_MAX : UINT32_MAX;
    // Contiguous, permuted, descending and wrapping addresses all retain lane order.
    std::array<std::vector<uint64_t>, 4> patterns;
    for (auto &addrs : patterns)
      addrs.resize(wf_size);
    for (uint32_t lane = 0; lane < wf_size; ++lane) {
      patterns[0][lane] = 0x1000 + uint64_t{lane} * kStride;
      patterns[1][lane] = 0x1000 + uint64_t{lane ^ 4u} * kStride;
      patterns[2][lane] = 0x1000 + uint64_t{wf_size - 1 - lane} * kStride;
      patterns[3][lane] = UINT64_MAX - 127 + uint64_t{lane} * kStride;
    }
    for (const auto &addrs : patterns) {
      for (const uint64_t mask :
           {full, full & ~uint64_t{0x100}, full & 0x5555555555555555, uint64_t{0}}) {
        SCOPED_TRACE(::testing::Message() << wf_size << ' ' << mask << ' ' << addrs[0]);
        LaneRuns actual;
        EXPECT_TRUE(amdgpu::for_each_lane_ordered_run(addrs.data(), mask, wf_size, kStride,
                                                      [&](const amdgpu::CoalescedLaneRun &run) {
                                                        actual.push_back(run);
                                                        return true;
                                                      }));
        expect_same_runs(actual, lane_ordered_runs(addrs, mask, kStride));
      }
    }
  }
}

TEST(CoalescedLaneRunsTest, LaneOrderedScanStopsWhenCallbackReturnsFalse) {
  for (const uint32_t wf_size : {32u, 64u}) {
    SCOPED_TRACE(wf_size);
    std::vector<uint64_t> addrs(wf_size);
    for (uint32_t lane = 0; lane < wf_size; ++lane)
      addrs[lane] = 0x1000 + uint64_t{lane ^ 4u} * 16;
    const uint64_t full = wf_size == 64 ? UINT64_MAX : UINT32_MAX;
    LaneRuns actual;
    EXPECT_FALSE(amdgpu::for_each_lane_ordered_run(addrs.data(), full, wf_size, 16,
                                                   [&](const amdgpu::CoalescedLaneRun &run) {
                                                     actual.push_back(run);
                                                     return false;
                                                   }));
    auto expected = lane_ordered_runs(addrs, full, 16);
    ASSERT_GT(expected.size(), 1u);
    expected.resize(1);
    expect_same_runs(actual, expected);
  }
}

TEST(CoalescedLaneRunsTest, BroadcastLaneOperandsKeepLaneOrderedRuns) {
  constexpr uint64_t kBase = 0x7ffe8e600000;
  constexpr uint32_t kStride = 16;
  for (const uint32_t wf_size : {32u, 64u}) {
    SCOPED_TRACE(wf_size);
    const uint64_t full = wf_size == 64 ? UINT64_MAX : (uint64_t{1} << wf_size) - 1;
    // Each group of eight lanes repeats the same tiled operands. The first
    // order break repeats lane 0, while the group itself may be contiguous.
    for (const uint32_t spacing : {64u, kStride}) {
      SCOPED_TRACE(spacing);
      std::vector<uint64_t> addrs(wf_size);
      for (uint32_t lane = 0; lane < wf_size; ++lane)
        addrs[lane] = kBase + (lane % 8) * spacing;
      for (const uint64_t mask : {full, full & ~uint64_t{0x100}, full & 0xf0f0f0f0f0f0f0f0}) {
        SCOPED_TRACE(mask);
        const LaneRuns runs = coalesced_runs(addrs, mask, kStride);
        expect_same_runs(runs, lane_ordered_runs(addrs, mask, kStride));
        if (spacing == kStride && mask == full) {
          ASSERT_EQ(runs.size(), wf_size / 8);
          EXPECT_EQ(runs[1].lane_mask, uint64_t{0xff00});
          EXPECT_EQ(runs[1].size, 8 * kStride);
        }
      }
      bool completed = true;
      EXPECT_EQ(coalesced_runs(addrs, full, kStride, 3, &completed).size(), 3u);
      EXPECT_FALSE(completed);
    }
  }
}

TEST(CoalescedLaneRunsTest, OverlapFoundAfterSortingKeepsLaneOrderedRuns) {
  // Lane 2 breaks address order without overlapping lanes 0 or 1. Lane 3
  // repeats lane 0, so only the sorted scan finds the overlap.
  const std::array<uint64_t, 4> addrs{0x1000, 0x1100, 0x1080, 0x1000};
  expect_same_runs(coalesced_runs(addrs, 0xf, 16), lane_ordered_runs(addrs, 0xf, 16));
  // A partial overlap with an earlier lane is proven at the first order break.
  const std::array<uint64_t, 4> partial{0x2000, 0x2010, 0x2020, 0x2008};
  expect_same_runs(coalesced_runs(partial, 0xf, 16), lane_ordered_runs(partial, 0xf, 16));
}

TEST(CoalescedLaneRunsTest, ReversedDisjointLanesStillCoalesce) {
  constexpr uint64_t kBase = 0x4000;
  std::array<uint64_t, 32> reversed{};
  for (uint32_t lane = 0; lane < reversed.size(); ++lane)
    reversed[lane] = kBase + (31 - lane) * sizeof(uint32_t);
  expect_same_runs(coalesced_runs(reversed, 0xffffffff, sizeof(uint32_t)),
                   {{.address = kBase,
                     .lane_mask = 0xffffffff,
                     .first_lane = 31,
                     .size = 32 * sizeof(uint32_t),
                     .lane_ordered = false}});
  const std::array<uint64_t, 2> swapped{kBase + 16, kBase};
  expect_same_runs(
      coalesced_runs(swapped, 0x3, 16),
      {{.address = kBase, .lane_mask = 0x3, .first_lane = 1, .size = 32, .lane_ordered = false}});
}

TEST(CoalescedLaneRunsTest, DenseStridePredicateHandlesEveryCountAndUnsignedLimit) {
  for (uint32_t count = 1; count <= 64; ++count) {
    for (const uint32_t stride : {1u, 16u, 129u, UINT32_MAX}) {
      const uint64_t extent = uint64_t{count - 1} * stride;
      for (const uint64_t base : {uint64_t{0}, uint64_t{1} << 63, UINT64_MAX - extent}) {
        SCOPED_TRACE(::testing::Message() << count << ' ' << stride << ' ' << base);
        std::vector<uint64_t> addrs(count);
        for (uint32_t lane = 0; lane < count; ++lane)
          addrs[lane] = base + uint64_t{lane} * stride;
        ASSERT_TRUE(amdgpu::detail::is_strided_lane_addresses(addrs.data(), count, stride));
        for (uint32_t lane = 1; lane < count; ++lane) {
          for (const uint64_t difference : {uint64_t{1}, uint64_t{1} << 32, uint64_t{1} << 63}) {
            addrs[lane] ^= difference;
            EXPECT_FALSE(amdgpu::detail::is_strided_lane_addresses(addrs.data(), count, stride));
            addrs[lane] ^= difference;
          }
        }
        if (count > 1 && base == UINT64_MAX - extent) {
          for (auto &address : addrs)
            ++address;
          EXPECT_FALSE(amdgpu::detail::is_strided_lane_addresses(addrs.data(), count, stride));
        }
      }
    }
  }
}

TEST(CoalescedLaneRunsTest, DenseStridePredicateRespectsUnalignedExactStorage) {
  for (const uint32_t count : {7u, 33u}) {
    // A prefix keeps only natural alignment, with no spare element at the end.
    const auto storage = std::make_unique<uint64_t[]>(count + 1);
    uint64_t *const addrs = storage.get() + 1;
    for (uint32_t lane = 0; lane < count; ++lane)
      addrs[lane] = 0xfffffff0ULL + uint64_t{lane} * 16;
    EXPECT_TRUE(amdgpu::detail::is_strided_lane_addresses(addrs, count, 16));
    addrs[count - 1] ^= uint64_t{1} << 32;
    EXPECT_FALSE(amdgpu::detail::is_strided_lane_addresses(addrs, count, 16));
  }
}

TEST(CoalescedLaneRunsTest, DenseStridedWindowsPreserveRunsAndEarlyStops) {
  for (const uint32_t wf_size : {32u, 64u}) {
    for (const uint32_t first : {0u, wf_size / 4, wf_size - 1}) {
      for (const uint32_t count : {1u, std::min(32u, wf_size - first), wf_size - first}) {
        const uint64_t mask = (count == 64 ? UINT64_MAX : (uint64_t{1} << count) - 1) << first;
        for (const uint32_t stride : {1u, 16u, 129u}) {
          for (const uint64_t base :
               {uint64_t{0x1070}, UINT64_MAX - (uint64_t{wf_size} * stride - 1)}) {
            SCOPED_TRACE(::testing::Message() << wf_size << ' ' << first << ' ' << count << ' '
                                              << stride << ' ' << base);
            std::vector<uint64_t> addrs(wf_size);
            for (uint32_t lane = 0; lane < wf_size; ++lane)
              addrs[lane] = base + uint64_t{lane} * stride;
            const LaneRuns expected{{.address = addrs[first],
                                     .lane_mask = mask,
                                     .first_lane = first,
                                     .size = count * stride,
                                     .lane_ordered = true}};
            for (const auto policy : {amdgpu::MemoryTransactionPolicy{},
                                      amdgpu::MemoryTransactionPolicy::unqualified_64()}) {
              for (const bool stop : {false, true}) {
                LaneRuns actual;
                const bool completed =
                    amdgpu::for_each_coalesced_lane_run(addrs.data(), mask, wf_size, stride, policy,
                                                        [&](const amdgpu::CoalescedLaneRun &run) {
                                                          actual.push_back(run);
                                                          return !stop;
                                                        });
                EXPECT_EQ(completed, !stop);
                expect_same_runs(actual, expected);
              }
            }
            if (count > 1) {
              addrs[first + 1] = addrs[first];
              expect_same_runs(coalesced_runs(addrs, mask, stride),
                               lane_ordered_runs(addrs, mask, stride));
            }
          }
        }
      }
    }
    std::vector<uint64_t> wrapping(wf_size);
    for (uint32_t lane = 0; lane < wf_size; ++lane)
      wrapping[lane] = UINT64_MAX - 128 + uint64_t{lane} * 16;
    const uint64_t full = wf_size == 64 ? UINT64_MAX : UINT32_MAX;
    expect_same_runs(coalesced_runs(wrapping, full, 16), lane_ordered_runs(wrapping, full, 16));
  }
}

TEST(CoalescedLaneRunsTest, WrappingOperandsKeepLaneOrderedRuns) {
  const std::array<uint64_t, 3> wrapping{0x1000, UINT64_MAX - 7, 0x2000};
  expect_same_runs(coalesced_runs(wrapping, 0x7, 16), lane_ordered_runs(wrapping, 0x7, 16));
  const std::array<uint64_t, 2> wrapping_last{0x1000, UINT64_MAX - 7};
  expect_same_runs(coalesced_runs(wrapping_last, 0x3, 16),
                   lane_ordered_runs(wrapping_last, 0x3, 16));
  // The last operand ending exactly at the top does not wrap. These disjoint
  // lanes keep the reordered path and first-participating-lane order.
  const std::array<uint64_t, 2> top{UINT64_MAX - 15, 0};
  expect_same_runs(coalesced_runs(top, 0x3, 16),
                   {{.address = UINT64_MAX - 15, .lane_mask = 0x1, .first_lane = 0, .size = 16},
                    {.address = 0, .lane_mask = 0x2, .first_lane = 1, .size = 16}});
}

TEST(CoalescedLaneRunsTest, SeededLaneRunsMatchOverlapAndCoverageContracts) {
  std::mt19937_64 random(0x5eed1a7e);
  for (uint32_t iteration = 0; iteration < 20000; ++iteration) {
    SCOPED_TRACE(iteration);
    const uint32_t wf_size = random() & 1 ? 64 : 32;
    constexpr std::array<uint32_t, 6> kStrides{1, 2, 4, 8, 12, 16};
    const uint32_t stride = kStrides[random() % kStrides.size()];
    const uint64_t base = random() % 8 == 0 ? UINT64_MAX - 64 * stride : (random() >> 20) << 4;
    const uint32_t slots = 1 + random() % (2 * wf_size);
    std::vector<uint64_t> addrs(wf_size);
    for (uint32_t lane = 0; lane < wf_size; ++lane) {
      const uint32_t slot = random() % 4 == 0 ? lane % slots : random() % slots;
      const uint64_t jitter = random() % 16 == 0 ? random() % stride : 0;
      addrs[lane] = base + uint64_t{slot} * stride + jitter;
    }
    if (random() % 4 == 0)
      std::ranges::sort(addrs);
    uint64_t mask = random();
    if (wf_size == 32)
      mask &= 0xffffffff;
    if (random() % 2)
      mask |= wf_size == 64 ? UINT64_MAX : 0xffffffff;
    const LaneRuns runs = coalesced_runs(addrs, mask, stride);
    if (operands_overlap_or_wrap(addrs, mask, stride)) {
      expect_same_runs(runs, lane_ordered_runs(addrs, mask, stride));
      continue;
    }
    // Disjoint operands: each active byte appears exactly once, every run is
    // fully supplied by its lanes, and runs follow first-participating lanes.
    std::vector<uint32_t> covered(wf_size);
    int previous_first = -1;
    for (const auto &run : runs) {
      ASSERT_EQ(run.lane_mask & ~mask, 0u);
      ASSERT_TRUE(run.lane_mask >> run.first_lane & 1);
      if (run.lane_ordered) {
        EXPECT_EQ(run.first_lane, static_cast<uint32_t>(std::countr_zero(run.lane_mask)));
      }
      EXPECT_LE(previous_first, std::countr_zero(run.lane_mask));
      previous_first = std::countr_zero(run.lane_mask);
      uint64_t supplied = 0;
      for (uint64_t lanes = run.lane_mask; lanes; lanes &= lanes - 1) {
        const uint32_t lane = std::countr_zero(lanes);
        const uint64_t begin = std::max(addrs[lane], run.address);
        const uint64_t end = std::min(addrs[lane] + (stride - 1), run.address + (run.size - 1));
        ASSERT_LE(begin, end);
        covered[lane] += end - begin + 1;
        supplied += end - begin + 1;
      }
      EXPECT_EQ(supplied, run.size);
    }
    for (uint32_t lane = 0; lane < wf_size; ++lane)
      EXPECT_EQ(covered[lane], (mask >> lane & 1) ? stride : 0u) << lane;
  }
}

TEST(CoalescedLaneRunsTest, WrappingOperandRetainsLaterLaneOrderAndCallbackStops) {
  const std::array<uint64_t, 8> addresses{UINT64_MAX - 7, 0x300, 0x310, 0x320,
                                          0x200,          0x210, 0x410, 0x400};
  const std::array<amdgpu::CoalescedLaneRun, 5> expected{
      amdgpu::CoalescedLaneRun{.address = UINT64_MAX - 7,
                               .lane_mask = 1,
                               .first_lane = 0,
                               .size = 16,
                               .lane_ordered = true},
      amdgpu::CoalescedLaneRun{
          .address = 0x300, .lane_mask = 0xe, .first_lane = 1, .size = 48, .lane_ordered = true},
      amdgpu::CoalescedLaneRun{
          .address = 0x200, .lane_mask = 0x30, .first_lane = 4, .size = 32, .lane_ordered = true},
      amdgpu::CoalescedLaneRun{
          .address = 0x410, .lane_mask = 0x40, .first_lane = 6, .size = 16, .lane_ordered = true},
      amdgpu::CoalescedLaneRun{
          .address = 0x400, .lane_mask = 0x80, .first_lane = 7, .size = 16, .lane_ordered = true}};
  for (auto policy :
       {amdgpu::MemoryTransactionPolicy{}, amdgpu::MemoryTransactionPolicy::unqualified_64()}) {
    LaneRuns actual;
    // Observe request metadata only: a wrapping operand cannot be dereferenced.
    EXPECT_TRUE(amdgpu::for_each_coalesced_lane_run(addresses.data(), 0xff, addresses.size(), 16,
                                                    policy,
                                                    [&](const amdgpu::CoalescedLaneRun &run) {
                                                      actual.push_back(run);
                                                      return true;
                                                    }));
    expect_same_runs(actual, LaneRuns(expected.begin(), expected.end()));
    for (size_t stop = 1; stop <= expected.size(); ++stop) {
      size_t calls = 0;
      EXPECT_FALSE(amdgpu::for_each_coalesced_lane_run(
          addresses.data(), 0xff, addresses.size(), 16, policy,
          [&](const amdgpu::CoalescedLaneRun &) { return ++calls != stop; }));
      EXPECT_EQ(calls, stop);
    }
  }
}

} // namespace
