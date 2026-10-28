// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "execution_fuzz_core.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <thread>
#include <vector>

namespace rocjitsu::execution_fuzz {
namespace {

TEST(ExecutionFuzzTargetTest, NamesCoverEveryExecutionTarget) {
  constexpr std::array<std::string_view, static_cast<size_t>(FuzzTarget::Count)> expected = {
      "gfx908",  "gfx90a",  "gfx942",  "gfx950",  "gfx1010", "gfx1030",
      "gfx1100", "gfx1151", "gfx1200", "gfx1201", "gfx1250", "gfx1251"};
  for (uint8_t index = 0; index < static_cast<uint8_t>(FuzzTarget::Count); ++index) {
    const FuzzTarget target = *indexed_target(index);
    EXPECT_EQ(target_name(target), expected[index]);
    EXPECT_EQ(parse_target(expected[index]), target);
  }
  EXPECT_FALSE(parse_target("unknown"));
  EXPECT_TRUE(targets_cover_amdgpu_execution_registry());
}

TEST(ExecutionFuzzTargetTest, InputSelectorExecutesOnEveryTarget) {
  for (uint8_t index = 0; index < static_cast<uint8_t>(FuzzTarget::Count); ++index) {
    SCOPED_TRACE(target_name(*indexed_target(index)));
    const std::array<uint8_t, 1> input = {index};
    const RunStats stats = run_input(input);
    EXPECT_EQ(stats.executed, 1u);
    EXPECT_EQ(stats.waves_halted, 1u);
    EXPECT_EQ(stats.inputs_executed, 1u);
  }
}

TEST(ExecutionFuzzTargetTest, UnassignedSelectorsDoNotWrapToAnotherTarget) {
  for (unsigned selector = static_cast<uint8_t>(FuzzTarget::Count); selector < 256; ++selector) {
    const auto byte = static_cast<uint8_t>(selector);
    EXPECT_FALSE(indexed_target(byte));
    const RunStats stats = run_input(std::array<uint8_t, 1>{byte});
    EXPECT_EQ(stats.inputs, 1u);
    EXPECT_EQ(stats.executed, 0u);
    EXPECT_EQ(stats.inputs_executed, 0u);
  }
  const RunStats overridden = run_input(std::array<uint8_t, 1>{255}, FuzzTarget::Gfx1250);
  EXPECT_EQ(overridden.waves_halted, 1u);
}

TEST(ExecutionFuzzTargetTest, GeneratedEncodingsReachExecutionAcrossTargets) {
  for (uint8_t index = 0; index < static_cast<uint8_t>(FuzzTarget::Count); ++index) {
    SCOPED_TRACE(target_name(*indexed_target(index)));
    const FuzzTarget target = *indexed_target(index);
    const RunStats stats = run_input(random_input(17 + index * 4, 64, target), target);
    EXPECT_EQ(stats.inputs, 1u);
    EXPECT_GT(stats.executed, 0u);
    EXPECT_EQ(stats.inputs_executed, 1u);
  }
}

TEST(ExecutionFuzzInputTest, InitialAddressPairsReachMappedMemory) {
  for (const bool scalar : {false, true}) {
    for (const uint8_t reg : {0, 6, 14}) {
      SCOPED_TRACE(scalar);
      SCOPED_TRACE(reg);
      const FuzzTarget target = scalar ? FuzzTarget::Gfx1250 : FuzzTarget::Gfx942;
      std::vector<uint8_t> input{static_cast<uint8_t>(target), 0};
      const auto append = [&](auto words) {
        for (const uint32_t word : words)
          for (unsigned byte = 0; byte < sizeof(word); ++byte)
            input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
      };
      if (scalar) {
        append(cdna5::build_smem(
            cdna5::kSLoadB32Smem,
            {.sbase = static_cast<uint8_t>(reg / 2), .sdata = 20, .soffset = 124}));
        append(cdna5::build_sopp(cdna5::kSWaitKmcntSopp));
        append(cdna5::build_sopp(cdna5::kSNopSopp));
      } else {
        append(cdna3::build_flat(cdna3::kFlatLoadDwordFlat, {.addr = reg, .vdst = 20}));
        append(cdna3::build_sopp(cdna3::kSWaitcntSopp));
        append(cdna3::build_sopp(cdna3::kSNopSopp));
      }
      const RunStats stats = run_input(input, target, nullptr, AsyncMode::Disabled, 1);
      EXPECT_EQ(stats.executed, 4u);
      EXPECT_EQ(stats.inputs_halted, 1u);
      EXPECT_EQ(stats.inputs_stopped, 0u);
      EXPECT_EQ(stats.budget_exhausted, 0u);
    }
  }
}

TEST(ExecutionFuzzInputTest, GeneratedInputsSelectSyncAsyncAndConcurrentExecution) {
  auto resources = make_async_execution_resources(2);
  struct Case {
    uint64_t seed;
    bool async;
    bool concurrent;
  };
  constexpr std::array cases = {
      Case{1, false, false},
      Case{17, true, false},
      Case{97, false, true},
      Case{113, true, true},
  };
  for (const auto &[seed, async, concurrent] : cases) {
    SCOPED_TRACE(seed);
    const RunStats stats =
        run_input(random_input(seed, 24, FuzzTarget::Gfx1250), FuzzTarget::Gfx1250, resources);
    EXPECT_EQ(stats.async_capable_inputs, 1u);
    EXPECT_EQ(stats.async_enabled_inputs, async);
    EXPECT_EQ(stats.concurrent_inputs, concurrent);
    EXPECT_EQ(stats.issuer_executions, concurrent ? 2u : 1u);
    if (async)
      EXPECT_GT(stats.async_executed, 0u);
    else
      EXPECT_EQ(stats.async_executed, 0u);
  }
}

TEST(ExecutionFuzzInputTest, ModeByteDoesNotRewriteAsyncWmmaPrefix) {
  for (uint64_t seed : {1u, 17u, 97u, 113u}) {
    const auto input = random_input(seed, 24, FuzzTarget::Gfx1250);
    ASSERT_GE(input.size(), 18u);
    EXPECT_EQ(input[17], 0x1du);
  }
}

TEST(ExecutionFuzzInputTest, TargetOverrideKeepsInputLayout) {
  constexpr uint64_t kGfx1250Selector = static_cast<uint8_t>(FuzzTarget::Gfx1250);
  const auto automatic = random_input(kGfx1250Selector, 24);
  const auto pinned = random_input(kGfx1250Selector, 24, FuzzTarget::Gfx1250);
  EXPECT_EQ(automatic, pinned);
  EXPECT_EQ(run_input(automatic).executed, run_input(pinned, FuzzTarget::Gfx1250).executed);
}

TEST(ExecutionFuzzInputTest, PinnedGeneratorEncodesPinnedTarget) {
  const auto input = random_input(5, 24, FuzzTarget::Gfx90a);
  ASSERT_GE(input.size(), kHeaderSize);
  EXPECT_EQ(input[0], static_cast<uint8_t>(FuzzTarget::Gfx90a));
  EXPECT_EQ(indexed_target(input[0]), FuzzTarget::Gfx90a);
}

TEST(ExecutionFuzzInputTest, ExplicitModesOverrideInputSelection) {
  auto resources = make_async_execution_resources(2);
  const auto async_input = random_input(17, 24, FuzzTarget::Gfx1250);
  const RunStats synchronous =
      run_input(async_input, FuzzTarget::Gfx1250, resources, AsyncMode::Disabled, 2);
  EXPECT_EQ(synchronous.async_enabled_inputs, 0u);
  EXPECT_EQ(synchronous.async_executed, 0u);
  EXPECT_EQ(synchronous.concurrent_inputs, 1u);
  EXPECT_EQ(synchronous.issuer_executions, 2u);

  const auto synchronous_input = random_input(1, 24, FuzzTarget::Gfx1250);
  const RunStats asynchronous =
      run_input(synchronous_input, FuzzTarget::Gfx1250, resources, AsyncMode::Enabled, 1);
  EXPECT_EQ(asynchronous.async_enabled_inputs, 1u);
  EXPECT_GT(asynchronous.async_executed, 0u);
  EXPECT_EQ(asynchronous.concurrent_inputs, 0u);
  EXPECT_EQ(asynchronous.issuer_executions, 1u);
}

TEST(ExecutionFuzzInputTest, AsyncOverridePreservesInitialScalarCondition) {
  const auto resources = make_async_execution_resources(1);
  for (uint8_t marker = 0; marker < 64; ++marker) {
    std::vector<uint8_t> input{static_cast<uint8_t>(FuzzTarget::Gfx1250), 0};
    const auto append = [&](auto words) {
      for (const uint32_t word : words)
        for (unsigned byte = 0; byte < sizeof(word); ++byte)
          input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
    };
    append(cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 1}));
    append(cdna5::build_sopp(cdna5::kSNopSopp));
    append(cdna5::build_sopp(cdna5::kSEndpgmSopp));
    input.push_back(marker);
    const RunStats synchronous = run_input(input, std::nullopt, resources, AsyncMode::Disabled, 1);
    const RunStats asynchronous = run_input(input, std::nullopt, resources, AsyncMode::Enabled, 1);
    EXPECT_EQ(synchronous.executed, asynchronous.executed) << unsigned(marker);
    EXPECT_EQ(synchronous.inputs_halted, 1u);
    EXPECT_EQ(asynchronous.inputs_halted, 1u);
  }
}

TEST(ExecutionFuzzInputTest, ThreadedInputsShareAsyncWmmaHelpers) {
  constexpr uint64_t kThreads = 8;
  constexpr uint64_t kInputsPerThread = 2;
  auto resources = make_async_execution_resources(4);
  std::vector<RunStats> results(kThreads);
  std::vector<std::jthread> workers;
  workers.reserve(kThreads);
  for (uint64_t worker = 0; worker != kThreads; ++worker) {
    workers.emplace_back([&, worker] {
      for (uint64_t input = 0; input != kInputsPerThread; ++input) {
        const uint64_t seed = 1 + 16 * (worker + input * kThreads);
        const RunStats stats =
            run_input(random_input(seed, 24, FuzzTarget::Gfx1250), FuzzTarget::Gfx1250, resources);
        results[worker].executed += stats.executed;
        results[worker].async_executed += stats.async_executed;
        results[worker].inputs += stats.inputs;
        results[worker].inputs_executed += stats.inputs_executed;
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  uint64_t async_executed = 0;
  for (const RunStats &stats : results) {
    EXPECT_EQ(stats.inputs, kInputsPerThread);
    EXPECT_EQ(stats.inputs_executed, kInputsPerThread);
    async_executed += stats.async_executed;
  }
  EXPECT_GT(async_executed, 0u);
}

} // namespace
} // namespace rocjitsu::execution_fuzz
