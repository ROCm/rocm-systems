// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <gtest/gtest.h>
#include <random>
#include <utility>
#include <vector>

using namespace rocjitsu::plugins::race_detector;
namespace amdgpu = rocjitsu::amdgpu;

namespace {

// Model only the retained LDS history, independently of chunk ownership,
// interval merging and EventRegistry storage. Completion and retirement are
// explicit inputs here; counter scheduling is tested separately.
class LdsHistoryTest : public ::testing::Test {
protected:
  static constexpr int kNumWaves = 4;
  static constexpr int kLdsBytes = 256;
  static constexpr std::array kOrders{MemoryOrderClass::LDS, MemoryOrderClass::VMEM,
                                      MemoryOrderClass::UNORDERED};

  struct Query {
    int wave;
    bool write;
    MemoryOrderClass order;
  };

  static Query sampleQuery(std::mt19937 &random) {
    return {int(random() % kNumWaves), bool(random() % 2), kOrders[random() % kOrders.size()]};
  }

  struct Event {
    EventId id;
    int wave;
    bool write;
    MemoryOrderClass order;
    amdgpu::WaitCounterType counter;
    std::bitset<kLdsBytes> bytes;
    bool complete = false;
  };

  void add(int wave, bool write, MemoryOrderClass order, const std::vector<Interval> &ranges) {
    const auto counter = order == MemoryOrderClass::VMEM ? amdgpu::WaitCounterType::VMCNT
                                                         : amdgpu::WaitCounterType::LGKMCNT;
    Event event{{}, wave, write, order, counter, {}};
    IntervalSet intervals;
    for (auto range : ranges) {
      intervals.append(range.start, range.end);
      for (int byte = range.start; byte < range.end; ++byte)
        event.bytes.set(byte);
    }
    intervals.finalize();
    const auto type = !write                            ? MemoryEventType::LDS_TO_VGPR
                      : order == MemoryOrderClass::VMEM ? MemoryEventType::GLOBAL_TO_LDS
                                                        : MemoryEventType::VGPR_TO_LDS;
    event.id = detector_.allocateEventId(WaveId{wave}, /*pc=*/0, type, {}, /*execMask=*/1,
                                         /*byteMask=*/0xF, std::move(intervals), counter, order);
    events_.push_back(event);
  }

  void complete(size_t index) {
    auto &event = events_.at(index);
    if (!event.complete) {
      ASSERT_TRUE(detector_.satisfyEventWaitCounter(event.id, event.counter));
      detector_.markEventWaveComplete(event.id);
      event.complete = true;
    }
  }

  void retire(size_t index) {
    ASSERT_NO_FATAL_FAILURE(complete(index));
    detector_.retireEvent(events_.at(index).id);
    events_.erase(events_.begin() + index);
  }

  void checkRange(int addr, int bytes, Query access) {
    SCOPED_TRACE(::testing::Message()
                 << "wave=" << access.wave << " order=" << int(access.order)
                 << " write=" << access.write << " range=[" << addr << "," << addr + bytes << ")");
    std::bitset<kLdsBytes> query;
    for (int byte = addr; byte < addr + bytes; ++byte)
      query.set(byte);
    std::vector<int> expected;
    for (const auto &event : events_) {
      if (event.write == access.write || (event.bytes & query).none())
        continue;
      // A completed event is safe only for its own wave. Otherwise,
      // that wave needs both accesses to share an ordered class.
      if (event.wave == access.wave &&
          (event.complete ||
           (access.order != MemoryOrderClass::UNORDERED && event.order == access.order)))
        continue;
      expected.push_back(event.id.value);
    }
    observed_.clear();
    if (access.write)
      detector_.validateWrite(addr, WaveId{access.wave}, /*lane=*/0, bytes, access.order);
    else
      detector_.validateRead(addr, WaveId{access.wave}, /*lane=*/0, bytes, access.order);
    // Retirement may reorder the detector's lists. Compare exact event
    // identities, so a missing report cannot cancel out an extra report.
    std::ranges::sort(expected);
    std::ranges::sort(observed_);
    ASSERT_EQ(observed_, expected);
  }

  std::vector<Event> events_;
  std::vector<int> observed_;
  RaceDetector detector_{kNumWaves,
                         /*vgprCount=*/8,
                         /*sgprCount=*/8,
                         Dim3d(0),
                         [this](RaceViolation v) { observed_.push_back(v.conflictingEvent.value); },
                         CounterCapacities{}};
};

TEST_F(LdsHistoryTest, DeterministicSoak) {
  // Use raw mt19937 output so the fixed seed also fixes the sequence across
  // standard library implementations. The bounds keep this in the unit suite.
  std::mt19937 random(12581);
  constexpr std::array boundaries{0, 12, 15, 16, 31, 32, 63, 64, 127, 128, 255};
  for (int round = 0; round < 8; ++round) {
    for (int step = 0; step < 256; ++step) {
      SCOPED_TRACE(::testing::Message() << "seed=12581 round=" << round << " step=" << step);
      // Start each round with sole owners. Later, mix waves and classes and
      // interleave completion, retirement and reuse, instead of leaving every
      // populated chunk permanently mixed throughout the test.
      const bool privateRegions = step < 32;
      if (privateRegions || events_.empty() || (events_.size() < 64 && random() % 3 == 0)) {
        const int wave = random() % kNumWaves;
        const bool write = random() % 2;
        const auto order =
            privateRegions ? MemoryOrderClass::LDS : kOrders[random() % kOrders.size()];
        std::vector<Interval> ranges;
        const int count = 1 + random() % 3;
        for (int i = 0; i < count; ++i) {
          const int base = privateRegions ? wave * 64 : 0;
          const int limit = privateRegions ? base + 64 : kLdsBytes;
          const int addr = base + random() % (limit - base);
          const int bytes = 1 + random() % 33;
          ranges.push_back({addr, std::min(addr + bytes, limit)});
        }
        // Include duplicate, overlapping and disjoint input intervals; the
        // byte-set oracle never calls the detector's interval-merging code.
        if (random() % 4 == 0)
          ranges.push_back(ranges.front());
        add(wave, write, order, ranges);
      } else {
        const size_t index = random() % events_.size();
        if (random() % 2)
          ASSERT_NO_FATAL_FAILURE(complete(index));
        else
          ASSERT_NO_FATAL_FAILURE(retire(index));
      }
      const int addr = step % 2 ? random() % kLdsBytes : boundaries[random() % boundaries.size()];
      const int bytes = std::min(1 + int(random() % 64), kLdsBytes - addr);
      ASSERT_NO_FATAL_FAILURE(checkRange(addr, bytes, sampleQuery(random)));
    }
    // Retire in a different order from allocation, checking the remaining
    // history after every removal. The next round reuses the emptied chunks.
    while (!events_.empty()) {
      SCOPED_TRACE(::testing::Message()
                   << "seed=12581 round=" << round << " remaining=" << events_.size());
      const size_t index = random() % events_.size();
      ASSERT_NO_FATAL_FAILURE(retire(index));
      ASSERT_NO_FATAL_FAILURE(checkRange(/*addr=*/0, kLdsBytes, sampleQuery(random)));
    }
  }
}

} // namespace
