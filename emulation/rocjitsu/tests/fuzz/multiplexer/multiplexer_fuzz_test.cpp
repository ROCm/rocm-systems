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
#include <vector>

namespace rocjitsu::multiplexer_fuzz {
namespace {

util::FailureOr<uint64_t> focused_work_units(FuzzTarget target, std::span<const uint8_t> payload) {
  switch (target) {
  case FuzzTarget::Decode: {
    const util::FailureOr<std::unique_ptr<Decoder>> decoder =
        decode_fuzz::create_decoder("gfx1250");
    EXPECT_TRUE(decoder.succeeded());
    if (decoder.failed())
      return util::Result::failure();
    const util::FailureOr<decode_fuzz::DecodeRecord> record =
        decode_fuzz::decode_window(*decoder.value(), payload);
    EXPECT_TRUE(record.succeeded());
    if (record.failed())
      return util::Result::failure();
    return record.value().valid;
  }
  case FuzzTarget::Execution: {
    const util::FailureOr<std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>>
        maybe_resources = execution_fuzz::make_async_execution_resources(1);
    EXPECT_TRUE(maybe_resources.succeeded());
    if (maybe_resources.failed())
      return util::Result::failure();
    const auto resources = maybe_resources.value();
    const util::FailureOr<execution_fuzz::RunStats> maybe_stats =
        execution_fuzz::run_input(payload, std::nullopt, resources);
    EXPECT_TRUE(maybe_stats.succeeded());
    if (maybe_stats.failed())
      return util::Result::failure();
    const execution_fuzz::RunStats stats = maybe_stats.value();
    return stats.executed;
  }
  case FuzzTarget::CodeObject: {
    const util::FailureOr<code_object_fuzz::RunStats> result = code_object_fuzz::run_input(payload);
    EXPECT_TRUE(result.succeeded());
    if (result.failed())
      return util::Result::failure();
    const code_object_fuzz::RunStats &stats = result.value();
    return stats.code_objects_valid + stats.executables_valid;
  }
  case FuzzTarget::ConcurrentDispatch:
    return concurrent_dispatch_fuzz::run_input(payload).instructions;
  }
  return util::Result::failure();
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
  const util::FailureOr<std::vector<std::vector<uint8_t>>> maybe_seeds = seed_corpus();
  ASSERT_TRUE(maybe_seeds.succeeded());
  const std::vector<std::vector<uint8_t>> seeds = maybe_seeds.value();
  ASSERT_EQ(seeds.size(), 8u);
  std::array<uint64_t, 4> seeds_per_target{};
  std::array<bool, 4> target_did_work{};
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    SCOPED_TRACE(index);
    const util::FailureOr<std::optional<DispatchResult>> maybe_result = run_input(seeds[index]);
    ASSERT_TRUE(maybe_result.succeeded());
    const std::optional<DispatchResult> result = maybe_result.value();
    ASSERT_TRUE(result);
    const std::span<const uint8_t> payload(seeds[index].data() + 1, seeds[index].size() - 1);
    const util::FailureOr<uint64_t> maybe_focused = focused_work_units(result->target, payload);
    ASSERT_TRUE(maybe_focused.succeeded());
    EXPECT_EQ(result->work_units, maybe_focused.value());
    const std::size_t target_index = static_cast<std::size_t>(result->target);
    ++seeds_per_target[target_index];
    target_did_work[target_index] |= result->work_units != 0;
  }
  EXPECT_EQ(seeds_per_target, (std::array<uint64_t, 4>{1, 2, 4, 1}));
  EXPECT_EQ(target_did_work, (std::array<bool, 4>{true, true, true, true}));
}

TEST(MultiplexerFuzzTest, ExecutionWorkUnitsCountMigratedInstructionsOnce) {
  const util::FailureOr<std::vector<std::vector<uint8_t>>> maybe_input = seed_corpus();
  ASSERT_TRUE(maybe_input.succeeded());
  const std::vector<uint8_t> input = maybe_input.value()[1];
  ASSERT_EQ(input.front(), static_cast<uint8_t>(FuzzTarget::Execution));
  const std::span<const uint8_t> payload(input.data() + 1, input.size() - 1);
  const util::FailureOr<std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>>
      maybe_resources = execution_fuzz::make_async_execution_resources(1);
  ASSERT_TRUE(maybe_resources.succeeded());
  const auto resources = maybe_resources.value();
  const util::FailureOr<execution_fuzz::RunStats> maybe_focused =
      execution_fuzz::run_input(payload, std::nullopt, resources);
  ASSERT_TRUE(maybe_focused.succeeded());
  const execution_fuzz::RunStats focused = maybe_focused.value();
  ASSERT_EQ(focused.executed, 6u);
  // All matrix operations now execute on the issuer.
  ASSERT_EQ(focused.async_executed, 0u);
  const util::FailureOr<std::optional<DispatchResult>> maybe_dispatched = run_input(input);
  ASSERT_TRUE(maybe_dispatched.succeeded());
  const std::optional<DispatchResult> dispatched = maybe_dispatched.value();
  ASSERT_TRUE(dispatched);
  EXPECT_EQ(dispatched->work_units, 6u);
}

TEST(MultiplexerFuzzTest, ConcurrentDispatchWorkUnitsCountExecutedInstructions) {
  const util::FailureOr<std::vector<std::vector<uint8_t>>> maybe_seeds = seed_corpus();
  ASSERT_TRUE(maybe_seeds.succeeded());
  const std::vector<uint8_t> input = maybe_seeds.value().back();
  ASSERT_EQ(input.front(), static_cast<uint8_t>(FuzzTarget::ConcurrentDispatch));
  const std::span<const uint8_t> payload(input.data() + 1, input.size() - 1);
  const concurrent_dispatch_fuzz::RunStats focused = concurrent_dispatch_fuzz::run_input(payload);
  const util::FailureOr<std::optional<DispatchResult>> maybe_dispatched = run_input(input);
  ASSERT_TRUE(maybe_dispatched.succeeded());
  const std::optional<DispatchResult> dispatched = maybe_dispatched.value();
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
    const util::FailureOr<std::optional<DispatchResult>> maybe_result = run_input(input);
    ASSERT_TRUE(maybe_result.succeeded());
    const std::optional<DispatchResult> result = maybe_result.value();
    ASSERT_TRUE(result);
    if (payload_size == decode_fuzz::kWindowSize)
      EXPECT_GT(result->work_units, 0u);
    else
      EXPECT_EQ(result->work_units, 0u);
  }

  const util::FailureOr<std::optional<DispatchResult>> maybe_empty_execution =
      run_input(std::array<uint8_t, 1>{static_cast<uint8_t>(FuzzTarget::Execution)});
  ASSERT_TRUE(maybe_empty_execution.succeeded());
  const std::optional<DispatchResult> empty_execution = maybe_empty_execution.value();
  ASSERT_TRUE(empty_execution);
  EXPECT_EQ(empty_execution->work_units, 0u);

  const util::FailureOr<std::vector<uint8_t>> maybe_maximum_execution =
      execution_fuzz::random_input(17, execution_fuzz::kMaxInputSize,
                                   execution_fuzz::FuzzTarget::Gfx1250);
  ASSERT_TRUE(maybe_maximum_execution.succeeded());
  std::vector<uint8_t> maximum_execution = maybe_maximum_execution.value();
  maximum_execution.insert(maximum_execution.begin(), static_cast<uint8_t>(FuzzTarget::Execution));
  const util::FailureOr<std::optional<DispatchResult>> maybe_maximum_result =
      run_input(maximum_execution);
  ASSERT_TRUE(maybe_maximum_result.succeeded());
  const std::optional<DispatchResult> maximum_result = maybe_maximum_result.value();
  ASSERT_TRUE(maximum_result);
  EXPECT_GT(maximum_result->work_units, 0u);

  std::vector<uint8_t> oversized_execution(execution_fuzz::kMaxInputSize + 2, 0);
  oversized_execution.front() = static_cast<uint8_t>(FuzzTarget::Execution);
  const util::FailureOr<std::optional<DispatchResult>> maybe_oversized_result =
      run_input(oversized_execution);
  ASSERT_TRUE(maybe_oversized_result.succeeded());
  const std::optional<DispatchResult> oversized_result = maybe_oversized_result.value();
  ASSERT_TRUE(oversized_result);
  EXPECT_EQ(oversized_result->work_units, 0u);

  std::vector<uint8_t> maximum_concurrent(concurrent_dispatch_fuzz::kMaxInputSize + 1, 0);
  maximum_concurrent.front() = static_cast<uint8_t>(FuzzTarget::ConcurrentDispatch);
  maximum_concurrent[1] = static_cast<uint8_t>(execution_fuzz::FuzzTarget::Gfx908);
  for (std::size_t offset = 1 + concurrent_dispatch_fuzz::kHeaderSize;
       offset < maximum_concurrent.size(); offset += sizeof(kSNop))
    for (unsigned byte = 0; byte < sizeof(kSNop); ++byte)
      maximum_concurrent[offset + byte] = static_cast<uint8_t>(kSNop >> (8 * byte));
  const util::FailureOr<std::optional<DispatchResult>> maybe_maximum_concurrent_result =
      run_input(maximum_concurrent);
  ASSERT_TRUE(maybe_maximum_concurrent_result.succeeded());
  const std::optional<DispatchResult> maximum_concurrent_result =
      maybe_maximum_concurrent_result.value();
  ASSERT_TRUE(maximum_concurrent_result);
  EXPECT_GT(maximum_concurrent_result->work_units, 0u);

  maximum_concurrent.push_back(0xff);
  const util::FailureOr<std::optional<DispatchResult>> maybe_oversized_concurrent_result =
      run_input(maximum_concurrent);
  ASSERT_TRUE(maybe_oversized_concurrent_result.succeeded());
  const std::optional<DispatchResult> oversized_concurrent_result =
      maybe_oversized_concurrent_result.value();
  ASSERT_TRUE(oversized_concurrent_result);
  EXPECT_EQ(oversized_concurrent_result->work_units, 0u);
}

TEST(MultiplexerFuzzTest, UnassignedSelectorIsAnOrdinaryNoOp) {
  const util::FailureOr<std::optional<DispatchResult>> maybe_dispatch =
      run_input(std::array<uint8_t, 1>{0xff});
  ASSERT_TRUE(maybe_dispatch.succeeded());
  EXPECT_FALSE(maybe_dispatch.value());
}

TEST(MultiplexerFuzzTest, RejectsMissingSelectorAndOversizedInput) {
  util::StringDiagnostic missing;
  EXPECT_TRUE(run_input({}, missing.emitter()).failed());
  EXPECT_FALSE(missing.message().empty());
  const std::vector<uint8_t> oversized(kMaxInputSize + 1, 0);
  util::StringDiagnostic large;
  EXPECT_TRUE(run_input(oversized, large.emitter()).failed());
  EXPECT_FALSE(large.message().empty());
}

} // namespace
} // namespace rocjitsu::multiplexer_fuzz
