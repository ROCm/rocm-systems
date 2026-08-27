// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_model_test.cpp
/// @brief Behavioural tests for the timing plane.
///
/// @details Every assertion here is about a *relation* that has to hold
/// whatever the tuning says, never about a number the tuning supplies. That is
/// deliberate and it is the only way these can live in CI: the calibrated
/// values -- issue occupancies, cache latencies, launch costs -- are shipped
/// separately from this repository, and a test that pinned one would either
/// fail everywhere they are absent or leak one where they are not.
///
/// So the tests here ask questions like: does more work cost more? Does the
/// same input give the same answer? Does an unnamed parameter make the run
/// slower rather than faster? Is a divergent access more expensive than a
/// coalesced one? Those hold for any tuning that is internally consistent, and
/// they are what actually breaks when the model is changed carelessly.

#include "rocjitsu/vm/timing/cache_des.h"
#include "rocjitsu/vm/timing/coalesce.h"
#include "rocjitsu/vm/timing/cu_des.h"
#include "rocjitsu/vm/timing/engine.h"
#include "rocjitsu/vm/timing/inst_class.h"
#include "rocjitsu/vm/timing/timing_plane.h"
#include "rocjitsu/vm/timing/tuning.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace rocjitsu::timing {
namespace {

/// @brief A small, entirely made-up machine.
///
/// @details Not a model of any part. Four compute units and round numbers, so
/// that a test runs in milliseconds and so that nothing here can be mistaken
/// for a measurement of hardware.
std::string tiny_config(const std::string &extra = {}) {
  return R"({
    "timing": {
      "enabled": true,
      "clock_mhz": 1000,
      "machine": {
        "compute_units": 4,
        "xcds": 1,
        "simd_lanes": 16,
        "wave_slots_per_cu": 8,
        "vector_registers_per_cu": 512,
        "scalar_registers_per_cu": 800,
        "lds_bytes_per_cu": 65536,
        "memory_channels": 2,
        "vector_alu.issue_cycles": 4,
        "vector_alu.ports": 4,
        "scalar_alu.issue_cycles": 1,
        "vector_memory_read.issue_cycles": 4,
        "vector_memory_write.issue_cycles": 4,
        "vector_memory.ports": 1,
        "l1_vector.hit_cycles": 10,
        "l1_vector.line_bytes": 64,
        "l1_vector.lines_per_cycle": 1,
        "l1_vector.sets": 4,
        "l1_vector.ways": 4,
        "l2.hit_cycles": 40,
        "l2.line_bytes": 64,
        "l2.lines_per_cycle": 4,
        "l2.sets": 16,
        "l2.ways": 4,
        "mall.hit_cycles": 80,
        "mall.line_bytes": 64,
        "mall.bytes_per_cycle": 128,
        "mall.sets": 64,
        "mall.ways": 4,
        "dram.latency_cycles": 200,
        "dram.bytes_per_cycle": 64,
        "fabric.bytes_per_cycle": 128,
        "fabric_request_bytes": 64,
        "dispatch.start_cycles": 100,
        "dispatch.end_cycles": 50,
        "dispatch.invalidate_cycles": 10,
        "dispatch.workgroups_per_cycle": 1,
        "front_end.issue_cycles": 1,
        "miss_status_registers_per_cu": 16,
        "wavefront_issue_cycles": 4)" +
         extra + R"(
      }
    }
  })";
}

DispatchShape shape_for(std::uint32_t workgroups, std::uint32_t waves_per_workgroup) {
  DispatchShape shape;
  shape.dispatch_id = 1;
  shape.queue_id = 0;
  shape.kernel_name = "test_kernel";
  shape.workgroup_count = workgroups;
  shape.waves_per_workgroup = waves_per_workgroup;
  shape.vector_registers_per_wave = 32;
  shape.scalar_registers_per_wave = 16;
  shape.lds_bytes_per_workgroup = 0;
  shape.wave_size = 64;
  return shape;
}

RetiredInstruction alu(std::uint64_t pc) {
  RetiredInstruction retired;
  retired.pc = pc;
  retired.inst_class = InstClass::VectorAlu;
  retired.active_lanes = 64;
  retired.wave_lanes = 64;
  retired.wait.fill(kUnconstrained);
  return retired;
}

RetiredInstruction load(std::uint64_t pc, const std::vector<std::uint64_t> &addresses) {
  RetiredInstruction retired;
  retired.pc = pc;
  retired.inst_class = InstClass::VectorMemoryRead;
  retired.active_lanes = static_cast<std::uint32_t>(addresses.size());
  retired.wave_lanes = 64;
  retired.wait.fill(kUnconstrained);
  retired.lane_addresses = &addresses;
  retired.bytes_per_lane = 4;
  retired.space = MemorySpace::Global;
  retired.is_load = true;
  return retired;
}

/// @brief Run one dispatch and report what it cost.
///
/// @param instructions Instructions each wavefront retires.
/// @param addresses When non-empty, every wavefront also performs one vector
///        load over these lane addresses.
std::uint64_t run_dispatch(TimingPlane &plane, const DispatchShape &shape,
                           std::uint32_t instructions,
                           const std::vector<std::uint64_t> &addresses = {}) {
  plane.dispatch_begin(shape);
  const std::uint64_t before = plane.current_cycles();
  const std::uint32_t waves =
      shape.workgroup_count * std::max<std::uint32_t>(1, shape.waves_per_workgroup);
  for (std::uint32_t wave = 0; wave < waves; ++wave) {
    const std::uint32_t unit = wave % 4;
    const std::uint32_t slot = wave / 4;
    plane.wave_begin(unit, slot, shape.dispatch_id, shape.queue_id);
    for (std::uint32_t index = 0; index < instructions; ++index)
      plane.instruction(unit, slot, alu(0x1000 + 4 * index));
    if (!addresses.empty())
      plane.instruction(unit, slot, load(0x2000, addresses));
    plane.wave_end(unit, slot);
  }
  plane.dispatch_end(shape.dispatch_id, shape.queue_id);
  return plane.current_cycles() - before;
}

std::vector<std::uint64_t> contiguous_lanes(std::uint64_t base) {
  std::vector<std::uint64_t> addresses;
  for (std::uint64_t lane = 0; lane < 64; ++lane)
    addresses.push_back(base + lane * 4);
  return addresses;
}

std::vector<std::uint64_t> divergent_lanes(std::uint64_t base) {
  std::vector<std::uint64_t> addresses;
  for (std::uint64_t lane = 0; lane < 64; ++lane)
    addresses.push_back(base + lane * 4096);
  return addresses;
}

// -- Tuning ------------------------------------------------------------------

TEST(TimingTuning, ReadsWhatTheConfigNames) {
  const Tuning tuning = Tuning::parse(tiny_config());
  EXPECT_TRUE(tuning.enabled);
  EXPECT_EQ(tuning.compute_units, 4u);
  EXPECT_EQ(tuning.l1_vector.line_bytes, 64u);
  EXPECT_FALSE(tuning.resolved.empty());
}

TEST(TimingTuning, DisabledWithoutATimingBlock) {
  const Tuning tuning = Tuning::parse(R"({"soc": {"name": "whatever"}})");
  EXPECT_FALSE(tuning.enabled);
}

TEST(TimingTuning, RecordsEveryParameterItHadToGuess) {
  // The ledger is the whole fail-slow contract: a run whose config is missing
  // parameters has to be able to say which, or an incomplete config is
  // indistinguishable from a complete one.
  const Tuning tuning = Tuning::parse(tiny_config());
  EXPECT_FALSE(tuning.fell_back.empty());
  for (const std::string &name : tuning.fell_back)
    EXPECT_FALSE(name.empty());
}

TEST(TimingTuning, UnnamedInstructionClassCostsAsMuchAsTheDearestNamedOne) {
  // Fail slow. An opcode nobody classified has to make the run read slow and
  // look suspicious, never fast and look accurate.
  const Tuning tuning = Tuning::parse(tiny_config(R"(,
        "matrix_multiply.issue_cycles": 32)"));
  const std::uint64_t unknown = tuning.issue_cycles[static_cast<std::size_t>(InstClass::Unknown)];
  for (std::size_t index = 0; index < kNumInstClasses; ++index)
    EXPECT_GE(unknown, tuning.issue_cycles[index]) << "class " << index;
}

TEST(TimingTuning, PerClassFrontEndDefaultsToTheSharedFigure) {
  // A config from before the front end was split by class has to behave
  // exactly as it did.
  const Tuning tuning = Tuning::parse(tiny_config());
  for (std::size_t index = 0; index < kNumInstClasses; ++index)
    EXPECT_EQ(tuning.front_end_sixteenths[index], tuning.front_end_cycles * kFrontEndScale)
        << inst_class_name(static_cast<InstClass>(index));
}

TEST(TimingTuning, PerClassFrontEndOverridesTheSharedFigure) {
  const Tuning tuning = Tuning::parse(tiny_config(R"(,
        "front_end.matrix_multiply.issue_cycles": 2.5)"));
  EXPECT_EQ(tuning.front_end_sixteenths[static_cast<std::size_t>(InstClass::MatrixMultiply)],
            static_cast<std::uint64_t>(2.5 * kFrontEndScale));
  EXPECT_EQ(tuning.front_end_sixteenths[static_cast<std::size_t>(InstClass::VectorAlu)],
            tuning.front_end_cycles * kFrontEndScale);
}

TEST(TimingTuning, PortCountsMayBeFractional) {
  // A unit's sustained rate over a mixed instruction stream is not an integer
  // even though every pipeline stage is.
  const Tuning tuning = Tuning::parse(tiny_config(R"(,
        "local_data_share.ports": 2.5)"));
  EXPECT_DOUBLE_EQ(tuning.ports[static_cast<std::size_t>(FunctionalUnit::LocalDataShare)], 2.5);
}

// -- Coalescing --------------------------------------------------------------

TEST(TimingCoalesce, ContiguousCostsFewerLinesThanDivergent) {
  std::vector<std::uint64_t> pool;
  const std::uint32_t together = coalesce_lines(contiguous_lanes(0x10000), 6, 4, pool);
  pool.clear();
  const std::uint32_t apart = coalesce_lines(divergent_lanes(0x10000), 6, 4, pool);
  EXPECT_LT(together, apart);
  EXPECT_EQ(apart, 64u) << "one line per lane is the worst case for a wave64";
}

TEST(TimingCoalesce, AnAccessStraddlingALineCountsBoth) {
  // Off by one byte from the line, so the last lane's four bytes run over.
  std::vector<std::uint64_t> pool;
  const std::uint32_t aligned = coalesce_lines({{0x1000}}, 6, 64, pool);
  pool.clear();
  const std::uint32_t straddling = coalesce_lines({{0x1001}}, 6, 64, pool);
  EXPECT_EQ(aligned, 1u);
  EXPECT_EQ(straddling, 2u);
}

TEST(TimingCoalesce, RecoalescingToACoarserLineNeverGrows) {
  std::vector<std::uint64_t> pool;
  const std::uint32_t fine = coalesce_lines(contiguous_lanes(0x20000), 6, 4, pool);
  const std::uint32_t coarse = recoalesce_lines(pool.data(), fine, 7);
  EXPECT_LE(coarse, fine);
  EXPECT_GE(coarse, 1u);
}

TEST(TimingLds, BroadcastIsFreeAndConflictsSerialise) {
  std::vector<std::uint64_t> same(64, 0x400);
  const std::uint64_t broadcast = lds_conflict_cycles(same, 4, 64, 4, 64);
  std::vector<std::uint64_t> strided;
  for (std::uint64_t lane = 0; lane < 64; ++lane)
    strided.push_back(lane * 64 * 4); // every lane on bank zero, all distinct
  const std::uint64_t conflicting = lds_conflict_cycles(strided, 4, 64, 4, 64);
  EXPECT_EQ(broadcast, 1u);
  EXPECT_GT(conflicting, broadcast);
  EXPECT_LE(conflicting, 64u) << "a wave64 cannot conflict worse than one lane per pass";
}

TEST(TimingLds, EveryAccessCostsAtLeastOneCycle) {
  EXPECT_GE(lds_conflict_cycles({{0x100}}, 4, 64, 4, 64), 1u);
}

// -- Tag array ---------------------------------------------------------------

TEST(TimingCache, RepeatedAccessHitsAndCapacityEvicts) {
  TagArray tags;
  tags.configure(2, 2, 64);
  EXPECT_FALSE(tags.access(0x0000, 1)) << "cold";
  EXPECT_TRUE(tags.access(0x0000, 2)) << "warm";
  // Four distinct lines into the same two-way set: the first is gone.
  tags.access(0x0080, 3);
  tags.access(0x0100, 4);
  tags.access(0x0180, 5);
  EXPECT_FALSE(tags.access(0x0000, 6));
}

TEST(TimingCache, InvalidateLosesEverything) {
  TagArray tags;
  tags.configure(4, 4, 64);
  tags.access(0x0000, 1);
  ASSERT_TRUE(tags.access(0x0000, 2));
  tags.invalidate();
  EXPECT_FALSE(tags.access(0x0000, 3));
}

// -- Composition -------------------------------------------------------------

TEST(TimingDispatch, SameInputGivesTheSameAnswer) {
  TimingPlane first{Tuning::parse(tiny_config())};
  TimingPlane second{Tuning::parse(tiny_config())};
  const DispatchShape shape = shape_for(4, 1);
  EXPECT_EQ(run_dispatch(first, shape, 64), run_dispatch(second, shape, 64));
}

TEST(TimingDispatch, TimeOnlyMovesForward) {
  TimingPlane plane{Tuning::parse(tiny_config())};
  std::uint64_t previous = plane.current_cycles();
  for (std::uint32_t round = 0; round < 4; ++round) {
    DispatchShape shape = shape_for(2, 1);
    shape.dispatch_id = round + 1;
    run_dispatch(plane, shape, 16);
    EXPECT_GE(plane.current_cycles(), previous);
    previous = plane.current_cycles();
  }
}

TEST(TimingDispatch, ADispatchThatRanNothingStillCostsItsLaunch) {
  TimingPlane plane{Tuning::parse(tiny_config())};
  const DispatchShape shape = shape_for(1, 1);
  plane.dispatch_begin(shape);
  const std::uint64_t before = plane.current_cycles();
  plane.dispatch_end(shape.dispatch_id, shape.queue_id);
  EXPECT_GT(plane.current_cycles() - before, 0u);
}

TEST(TimingDispatch, MoreInstructionsNeverCostLess) {
  std::uint64_t previous = 0;
  for (std::uint32_t instructions : {8u, 64u, 512u, 4096u}) {
    // A fresh machine each time. Sharing one would compare a cold run against
    // a warm one and measure the caches rather than the work.
    TimingPlane plane{Tuning::parse(tiny_config())};
    DispatchShape shape = shape_for(4, 1);
    shape.dispatch_id = instructions;
    const std::uint64_t cost = run_dispatch(plane, shape, instructions);
    EXPECT_GE(cost, previous) << "at " << instructions << " instructions";
    previous = cost;
  }
}

TEST(TimingDispatch, MoreWorkgroupsNeverCostLess) {
  std::uint64_t previous = 0;
  for (std::uint32_t workgroups : {1u, 2u, 8u, 32u}) {
    TimingPlane plane{Tuning::parse(tiny_config())};
    DispatchShape shape = shape_for(workgroups, 1);
    shape.dispatch_id = 100 + workgroups;
    const std::uint64_t cost = run_dispatch(plane, shape, 32);
    EXPECT_GE(cost, previous) << "at " << workgroups << " workgroups";
    previous = cost;
  }
}

TEST(TimingDispatch, DivergentAccessCostsMoreThanCoalesced) {
  // The single relation the whole memory side exists to produce, and the one
  // that no static analysis of the source could recover.
  TimingPlane plane{Tuning::parse(tiny_config())};
  DispatchShape shape = shape_for(4, 1);
  const std::uint64_t together = run_dispatch(plane, shape, 8, contiguous_lanes(0x100000));
  shape.dispatch_id = 2;
  const std::uint64_t apart = run_dispatch(plane, shape, 8, divergent_lanes(0x200000));
  EXPECT_GT(apart, together);
}

TEST(TimingDispatch, ReReadingWhatIsAlreadyResidentIsCheaper) {
  TimingPlane plane{Tuning::parse(tiny_config())};
  const std::vector<std::uint64_t> lanes = divergent_lanes(0x300000);
  DispatchShape shape = shape_for(1, 1);
  const std::uint64_t cold = run_dispatch(plane, shape, 4, lanes);
  shape.dispatch_id = 2;
  const std::uint64_t warm = run_dispatch(plane, shape, 4, lanes);
  EXPECT_LE(warm, cold);
}

TEST(TimingDispatch, ASlowerFrontEndNeverMakesADispatchFaster) {
  // Relations between tunings, not values: whatever the shipped numbers are,
  // charging the front end more cannot produce a shorter dispatch.
  const DispatchShape shape = shape_for(4, 2);
  TimingPlane quick{Tuning::parse(tiny_config())};
  TimingPlane slow{Tuning::parse(tiny_config(R"(,
        "front_end.vector_alu.issue_cycles": 4)"))};
  EXPECT_GE(run_dispatch(slow, shape, 256), run_dispatch(quick, shape, 256));
}

TEST(TimingDispatch, TheStragglerTermGrowsWithTheWavefrontCountAndIsOffByDefault) {
  const std::string with = R"(,
        "straggler_cycles": 64)";
  TimingPlane without{Tuning::parse(tiny_config())};
  TimingPlane charged{Tuning::parse(tiny_config(with))};
  const DispatchShape small = shape_for(1, 1);
  DispatchShape large = shape_for(32, 4);
  large.dispatch_id = 2;

  const std::uint64_t plain_small = run_dispatch(without, small, 4);
  const std::uint64_t plain_large = run_dispatch(without, large, 4);
  const std::uint64_t straggling_small = run_dispatch(charged, small, 4);
  const std::uint64_t straggling_large = run_dispatch(charged, large, 4);

  // One wavefront has no straggler; many wavefronts do, and it grows.
  EXPECT_EQ(plain_small, straggling_small);
  EXPECT_GT(straggling_large, plain_large);
  EXPECT_GT(straggling_large - plain_large, straggling_small - plain_small);
}

TEST(TimingDispatch, ExposingLessOfAStallNeverCostsMore) {
  const DispatchShape shape = shape_for(2, 1);
  TimingPlane full{Tuning::parse(tiny_config())};
  TimingPlane halved{Tuning::parse(tiny_config(R"(,
        "stall_exposed_fraction": 0.5)"))};
  const std::vector<std::uint64_t> lanes = divergent_lanes(0x400000);
  EXPECT_LE(run_dispatch(halved, shape, 8, lanes), run_dispatch(full, shape, 8, lanes));
}

// -- Engine ------------------------------------------------------------------

TEST(TimingEngine, RunsToIdleAndStaysThere) {
  Tuning tuning = Tuning::parse(tiny_config());
  TimingPlane plane{std::move(tuning)};
  const DispatchShape shape = shape_for(2, 1);
  run_dispatch(plane, shape, 32);
  const std::uint64_t settled = plane.current_cycles();
  EXPECT_EQ(plane.current_cycles(), settled);
}

} // namespace
} // namespace rocjitsu::timing
