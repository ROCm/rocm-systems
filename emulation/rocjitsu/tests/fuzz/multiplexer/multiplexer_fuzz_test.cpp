// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "multiplexer_fuzz_core.h"

#include "code_object_fuzz_core.h"
#include "concurrent_dispatch_fuzz_core.h"
#include "decode_fuzz_core.h"
#include "execution_fuzz_core.h"

#include "rocjitsu/isa/decoder.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace rocjitsu::multiplexer_fuzz {
namespace {

uint64_t focused_work_units(FuzzTarget target, std::span<const uint8_t> payload) {
  switch (target) {
  case FuzzTarget::Decode: {
    std::unique_ptr<Decoder> decoder = decode_fuzz::create_decoder("gfx1250");
    return decode_fuzz::decode_window(*decoder, payload).valid;
  }
  case FuzzTarget::Execution: {
    const auto resources = execution_fuzz::make_async_execution_resources(1);
    const execution_fuzz::RunStats stats =
        execution_fuzz::run_input(payload, std::nullopt, resources);
    return stats.executed;
  }
  case FuzzTarget::CodeObject: {
    const code_object_fuzz::RunStats stats = code_object_fuzz::run_input(payload);
    return stats.code_objects_valid + stats.executables_valid;
  }
  case FuzzTarget::ConcurrentDispatch:
    return concurrent_dispatch_fuzz::run_input(payload).instructions;
  }
  throw std::logic_error("multiplexer fuzz target registry is inconsistent");
}

TEST(MultiplexerFuzzTest, RegistryHasEveryFocusedHarnessInStableOrder) {
  const std::span<const TargetDescriptor> registered = targets();
  ASSERT_EQ(registered.size(), 4u);
  EXPECT_EQ(registered[0].id, FuzzTarget::Decode);
  EXPECT_EQ(registered[0].name, "decode");
  EXPECT_EQ(registered[1].id, FuzzTarget::Execution);
  EXPECT_EQ(registered[1].name, "execution");
  EXPECT_EQ(registered[2].id, FuzzTarget::CodeObject);
  EXPECT_EQ(registered[2].name, "code-object");
  EXPECT_EQ(registered[3].id, FuzzTarget::ConcurrentDispatch);
  EXPECT_EQ(registered[3].name, "concurrent-dispatch");
}

TEST(MultiplexerFuzzTest, SelectorIdsRemainStableAsTargetsAreAdded) {
  EXPECT_EQ(selected_target(0), FuzzTarget::Decode);
  EXPECT_EQ(selected_target(1), FuzzTarget::Execution);
  EXPECT_EQ(selected_target(2), FuzzTarget::CodeObject);
  EXPECT_EQ(selected_target(3), FuzzTarget::ConcurrentDispatch);

  std::array<uint64_t, 4> selections{};
  uint64_t unassigned = 0;
  for (unsigned selector = 0; selector <= std::numeric_limits<uint8_t>::max(); ++selector) {
    const std::optional<FuzzTarget> target = selected_target(static_cast<uint8_t>(selector));
    if (target)
      ++selections[static_cast<std::size_t>(*target)];
    else
      ++unassigned;
  }

  EXPECT_EQ(selections[0], 1u);
  EXPECT_EQ(selections[1], 1u);
  EXPECT_EQ(selections[2], 1u);
  EXPECT_EQ(selections[3], 1u);
  EXPECT_EQ(unassigned, 252u);
}

TEST(MultiplexerFuzzTest, StructuredSeedsReachEveryHarness) {
  const std::vector<std::vector<uint8_t>> seeds = seed_corpus();
  ASSERT_EQ(seeds.size(), 7u);
  std::array<uint64_t, 4> seeds_per_target{};
  std::array<bool, 4> target_did_work{};
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    SCOPED_TRACE(index);
    const std::optional<DispatchResult> result = run_input(seeds[index]);
    ASSERT_TRUE(result);
    const std::span<const uint8_t> payload(seeds[index].data() + 1, seeds[index].size() - 1);
    EXPECT_EQ(result->work_units, focused_work_units(result->target, payload));
    const std::size_t target_index = static_cast<std::size_t>(result->target);
    ++seeds_per_target[target_index];
    target_did_work[target_index] |= result->work_units != 0;
  }
  EXPECT_EQ(seeds_per_target, (std::array<uint64_t, 4>{1, 1, 4, 1}));
  EXPECT_EQ(target_did_work, (std::array<bool, 4>{true, true, true, true}));
}

TEST(MultiplexerFuzzTest, ExecutionWorkUnitsCountAsyncInstructionsOnce) {
  const std::vector<uint8_t> input = seed_corpus()[1];
  ASSERT_EQ(input.front(), static_cast<uint8_t>(FuzzTarget::Execution));
  const std::span<const uint8_t> payload(input.data() + 1, input.size() - 1);
  const auto resources = execution_fuzz::make_async_execution_resources(1);
  const execution_fuzz::RunStats focused =
      execution_fuzz::run_input(payload, std::nullopt, resources);
  ASSERT_EQ(focused.executed, 6u);
  ASSERT_EQ(focused.async_executed, 1u);
  const std::optional<DispatchResult> dispatched = run_input(input);
  ASSERT_TRUE(dispatched);
  EXPECT_EQ(dispatched->work_units, 6u);
}

TEST(MultiplexerFuzzTest, ConcurrentDispatchWorkUnitsCountExecutedInstructions) {
  const std::vector<uint8_t> input = seed_corpus().back();
  ASSERT_EQ(input.front(), static_cast<uint8_t>(FuzzTarget::ConcurrentDispatch));
  const std::span<const uint8_t> payload(input.data() + 1, input.size() - 1);
  const concurrent_dispatch_fuzz::RunStats focused = concurrent_dispatch_fuzz::run_input(payload);
  const std::optional<DispatchResult> dispatched = run_input(input);
  ASSERT_TRUE(dispatched);
  EXPECT_GT(focused.instructions, 0u);
  EXPECT_EQ(dispatched->work_units, focused.instructions);
}

TEST(MultiplexerFuzzTest, PreservesFocusedPayloadSizeContracts) {
  constexpr uint32_t kSNop = 0xBF800000u;
  for (const std::size_t payload_size : {15u, 16u, 17u}) {
    std::vector<uint8_t> input(payload_size + 1, 0);
    input.front() = static_cast<uint8_t>(FuzzTarget::Decode);
    for (unsigned byte = 0; byte < sizeof(kSNop); ++byte)
      input[1 + byte] = static_cast<uint8_t>(kSNop >> (8 * byte));
    const std::optional<DispatchResult> result = run_input(input);
    ASSERT_TRUE(result);
    if (payload_size == decode_fuzz::kWindowSize)
      EXPECT_GT(result->work_units, 0u);
    else
      EXPECT_EQ(result->work_units, 0u);
  }

  const std::optional<DispatchResult> empty_execution =
      run_input(std::array<uint8_t, 1>{static_cast<uint8_t>(FuzzTarget::Execution)});
  ASSERT_TRUE(empty_execution);
  EXPECT_EQ(empty_execution->work_units, 0u);

  std::vector<uint8_t> maximum_execution = execution_fuzz::random_input(
      17, execution_fuzz::kMaxInputSize, execution_fuzz::FuzzTarget::Gfx1250);
  maximum_execution.insert(maximum_execution.begin(), static_cast<uint8_t>(FuzzTarget::Execution));
  const std::optional<DispatchResult> maximum_result = run_input(maximum_execution);
  ASSERT_TRUE(maximum_result);
  EXPECT_GT(maximum_result->work_units, 0u);

  std::vector<uint8_t> oversized_execution(execution_fuzz::kMaxInputSize + 2, 0);
  oversized_execution.front() = static_cast<uint8_t>(FuzzTarget::Execution);
  const std::optional<DispatchResult> oversized_result = run_input(oversized_execution);
  ASSERT_TRUE(oversized_result);
  EXPECT_EQ(oversized_result->work_units, 0u);

  std::vector<uint8_t> maximum_concurrent(concurrent_dispatch_fuzz::kMaxInputSize + 1, 0);
  maximum_concurrent.front() = static_cast<uint8_t>(FuzzTarget::ConcurrentDispatch);
  maximum_concurrent[1] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx908);
  for (std::size_t offset = 1 + concurrent_dispatch_fuzz::kHeaderSize;
       offset < maximum_concurrent.size(); offset += sizeof(kSNop))
    for (unsigned byte = 0; byte < sizeof(kSNop); ++byte)
      maximum_concurrent[offset + byte] = static_cast<uint8_t>(kSNop >> (8 * byte));
  const std::optional<DispatchResult> maximum_concurrent_result = run_input(maximum_concurrent);
  ASSERT_TRUE(maximum_concurrent_result);
  EXPECT_GT(maximum_concurrent_result->work_units, 0u);

  maximum_concurrent.push_back(0xff);
  const std::optional<DispatchResult> oversized_concurrent_result = run_input(maximum_concurrent);
  ASSERT_TRUE(oversized_concurrent_result);
  EXPECT_EQ(oversized_concurrent_result->work_units, 0u);
}

TEST(MultiplexerFuzzTest, UnassignedSelectorIsAnOrdinaryNoOp) {
  EXPECT_FALSE(run_input(std::array<uint8_t, 1>{0xff}));
}

TEST(MultiplexerFuzzTest, RejectsMissingSelectorAndOversizedInput) {
  EXPECT_THROW((void)run_input({}), std::invalid_argument);
  const std::vector<uint8_t> oversized(kMaxInputSize + 1, 0);
  EXPECT_THROW((void)run_input(oversized), std::invalid_argument);
}

} // namespace
} // namespace rocjitsu::multiplexer_fuzz
