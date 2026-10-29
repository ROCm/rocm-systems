// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file simdojo_pipe_latch_test.cpp
/// @brief Tests for simdojo::PipeLatch: the clock edge, and use between the
/// stages of a clocked component.

#include "simdojo/components/pipe_latch.h"
#include "simdojo/sim/clock_domain.h"
#include "simdojo/sim/clocked.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace simdojo;

namespace {

TEST(PipeLatchTest, WrittenValueIsValidAfterTheClockEdge) {
  PipeLatch<int> latch;
  EXPECT_TRUE(latch.empty());
  latch.write(7);
  EXPECT_FALSE(latch.valid());
  EXPECT_FALSE(latch.empty());
  latch.clock();
  ASSERT_TRUE(latch.valid());
  EXPECT_EQ(latch.take(), 7);
  EXPECT_FALSE(latch.valid());
  EXPECT_TRUE(latch.empty());
}

TEST(PipeLatchTest, EdgeWithoutAWriteLatchesNothing) {
  PipeLatch<int> latch;
  latch.write(1);
  latch.clock();
  latch.clock();
  EXPECT_FALSE(latch.valid());
  EXPECT_TRUE(latch.empty());
}

TEST(PipeLatchTest, WriteAndTakeInTheSameCycle) {
  PipeLatch<int> latch;
  latch.write(1);
  latch.clock();
  latch.write(2);
  EXPECT_EQ(latch.take(), 1);
  EXPECT_FALSE(latch.valid());
  latch.clock();
  EXPECT_EQ(latch.take(), 2);
}

TEST(PipeLatchTest, CarriesMoveOnlyValues) {
  PipeLatch<std::unique_ptr<int>> latch;
  latch.write(std::make_unique<int>(3));
  latch.clock();
  ASSERT_TRUE(latch.valid());
  auto value = latch.take();
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 3);
}

/// A three-stage pipeline (source -> double -> sink) built from two latches.
/// Each stage runs once per cycle, front-to-back or back-to-front, and the
/// latches are clocked after all of them.
class ThreeStagePipeline : public Clocked<Component> {
public:
  ThreeStagePipeline(const ClockDomain &domain, int count, bool back_to_front)
      : Clocked<Component>("pipeline", domain), count_(count), back_to_front_(back_to_front) {}

  bool advance(Tick) override {
    if (back_to_front_) {
      sink();
      double_stage();
      source();
    } else {
      source();
      double_stage();
      sink();
    }
    source_to_double_.clock();
    double_to_sink_.clock();
    ++cycle_;
    return next_ < count_ || !source_to_double_.empty() || !double_to_sink_.empty();
  }

  std::vector<std::pair<uint64_t, int>> arrivals; ///< (cycle, value) at the sink.

private:
  void source() {
    if (next_ < count_)
      source_to_double_.write(next_++);
  }
  void double_stage() {
    if (source_to_double_.valid())
      double_to_sink_.write(2 * source_to_double_.take());
  }
  void sink() {
    if (double_to_sink_.valid())
      arrivals.emplace_back(cycle_, double_to_sink_.take());
  }

  PipeLatch<int> source_to_double_;
  PipeLatch<int> double_to_sink_;
  int count_;
  int next_ = 0;
  uint64_t cycle_ = 0;
  bool back_to_front_;
};

std::vector<std::pair<uint64_t, int>> run_pipeline(int count, bool back_to_front) {
  auto maybe_domain = ClockDomain::create("clk", 1'000'000'000);
  EXPECT_TRUE(maybe_domain.succeeded());
  if (maybe_domain.failed())
    return {};
  const auto &domain = maybe_domain.value();
  SimulationEngine engine({.num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");
  auto *pipeline = static_cast<ThreeStagePipeline *>(
      root->add_child(std::make_unique<ThreeStagePipeline>(domain, count, back_to_front)));
  engine.topology().set_root(std::move(root));
  const auto created = engine.create();
  EXPECT_TRUE(created.succeeded());
  if (created.failed())
    return {};
  const ExitStatus exit = engine.run();
  EXPECT_EQ(exit.reason, ExitReason::COMPLETED);
  return pipeline->arrivals;
}

TEST(PipeLatchTest, ClockedPipelineMovesOneStagePerCycleInEitherOrder) {
  // Written in cycle c, a value is doubled in c + 1 and sunk in c + 2.
  const std::vector<std::pair<uint64_t, int>> expected{{2, 0}, {3, 2}, {4, 4}, {5, 6}};
  EXPECT_EQ(run_pipeline(4, false), expected);
  EXPECT_EQ(run_pipeline(4, true), expected);
}

} // namespace
