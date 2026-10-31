// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "capture_fixture.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/hooks/consan/runtime_core.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <thread>

namespace rocjitsu::consan {
namespace {

class ConSanRuntimeCore : public ::testing::Test {
protected:
  void SetUp() override {
    header_ = test::capture_header();
    facts_ = std::make_shared<const Manifest>(header_.manifest);
    raw_ = test::execution(*facts_, 1).report;
  }

  CapturedExecution envelope(uint64_t serial = 1, uint32_t generation = 1) {
    CapturedExecution result = test::execution(*facts_, serial, generation);
    result.report.clear();
    result.completion = 0;
    return result;
  }

  CaptureSink sink(size_t maximum_write = SIZE_MAX) {
    return {
        .write = [this, maximum_write](std::span<const uint8_t> bytes) -> int64_t {
          const size_t count = std::min(maximum_write, bytes.size());
          output_.insert(output_.end(), bytes.begin(), bytes.begin() + count);
          return static_cast<int64_t>(count);
        },
        .close =
            [this] {
              ++closes_;
              return close_succeeds_;
            },
        .publish =
            [this] {
              ++publishes_;
              return publish_succeeds_;
            },
    };
  }

  OwnedSnapshot copied_snapshot(uint64_t serial = 1, uint32_t generation = 1) {
    DispatchLifecycle lifecycle(facts_, envelope(serial, generation));
    EXPECT_EQ(lifecycle.submitted(), RuntimeStatus::Success);
    EXPECT_EQ(lifecycle.observe(true, true, [] { return true; }), RuntimeStatus::Success);
    OwnedSnapshot snapshot;
    EXPECT_EQ(lifecycle.snapshot(
                  [this](std::span<uint8_t> bytes) {
                    std::copy(raw_.begin(), raw_.end(), bytes.begin());
                    return CopyResult{RuntimeStatus::Success, raw_.size()};
                  },
                  snapshot),
              RuntimeStatus::Success);
    EXPECT_EQ(lifecycle.snapshot({}, snapshot), RuntimeStatus::InvalidTransition);
    return snapshot;
  }

  CaptureHeader header_;
  std::shared_ptr<const Manifest> facts_;
  std::vector<uint8_t> raw_;
  std::vector<uint8_t> output_;
  uint32_t closes_ = 0;
  uint32_t publishes_ = 0;
  bool close_succeeds_ = true;
  bool publish_succeeds_ = true;
};

TEST_F(ConSanRuntimeCore, CompletionAndSystemVisibilityGateCopyReuseAndExactlyOnceForwarding) {
  DispatchLifecycle lifecycle(facts_, envelope());
  uint32_t forwards = 0;
  const auto forward = [&] {
    ++forwards;
    return true;
  };
  OwnedSnapshot snapshot;
  EXPECT_EQ(lifecycle.observe(true, true, forward), RuntimeStatus::InvalidTransition);
  EXPECT_EQ(lifecycle.submitted(), RuntimeStatus::Success);
  EXPECT_EQ(lifecycle.submitted(), RuntimeStatus::InvalidTransition);
  EXPECT_EQ(lifecycle.observe(false, true, forward), RuntimeStatus::PendingCompletion);
  EXPECT_EQ(lifecycle.snapshot({}, snapshot), RuntimeStatus::InvalidTransition);
  EXPECT_EQ(snapshot.facts, nullptr);
  EXPECT_EQ(forwards, 0u);
  EXPECT_EQ(lifecycle.observe(true, false, forward), RuntimeStatus::PendingVisibility);
  EXPECT_EQ(lifecycle.snapshot({}, snapshot), RuntimeStatus::InvalidTransition);
  EXPECT_EQ(forwards, 0u);
  EXPECT_EQ(lifecycle.observe(false, true, forward), RuntimeStatus::Success);
  EXPECT_EQ(lifecycle.observe(true, true, forward), RuntimeStatus::Success);
  EXPECT_EQ(forwards, 1u);
  EXPECT_EQ(snapshot.facts, nullptr);
}

TEST_F(ConSanRuntimeCore, InvalidLayoutEnvelopeAndContextCannotArmAnExecution) {
  CapturedExecution invalid_layout = envelope();
  invalid_layout.dimensions.grid[0] += 1;
  CapturedExecution populated_report = envelope();
  populated_report.report = raw_;
  CapturedExecution wrong_context = envelope();
  wire::write_u64(wrong_context.context, wire::kContextSerial, wrong_context.serial + 1);
  for (const CapturedExecution &invalid : {invalid_layout, populated_report, wrong_context}) {
    DispatchLifecycle lifecycle(facts_, invalid);
    EXPECT_EQ(lifecycle.submitted(), RuntimeStatus::InvalidFacts);
    uint32_t forwards = 0;
    EXPECT_EQ(lifecycle.observe(true, true,
                                [&] {
                                  ++forwards;
                                  return true;
                                }),
              RuntimeStatus::InvalidTransition);
    EXPECT_EQ(forwards, 0u);
    OwnedSnapshot snapshot;
    EXPECT_EQ(lifecycle.snapshot({}, snapshot), RuntimeStatus::InvalidTransition);
    EXPECT_EQ(snapshot.facts, nullptr);
  }
}

TEST_F(ConSanRuntimeCore, FailedForwardingIsNeverRetriedAndCannotCertifyAReusableSlice) {
  DispatchLifecycle lifecycle(facts_, envelope());
  ASSERT_EQ(lifecycle.submitted(), RuntimeStatus::Success);
  uint32_t forwards = 0;
  const auto failure = [&] {
    ++forwards;
    return false;
  };
  EXPECT_EQ(lifecycle.observe(true, true, failure), RuntimeStatus::ForwardFailed);
  EXPECT_EQ(lifecycle.observe(true, true, failure), RuntimeStatus::InvalidTransition);
  EXPECT_EQ(forwards, 1u);
  OwnedSnapshot snapshot;
  EXPECT_EQ(lifecycle.snapshot({}, snapshot), RuntimeStatus::InvalidTransition);
  EXPECT_EQ(snapshot.facts, nullptr);
}

TEST_F(ConSanRuntimeCore, MissingFailedAndShortCopiesRemainDistinctAndDoNotPermitReuse) {
  for (const RuntimeStatus expected :
       {RuntimeStatus::CopyUnavailable, RuntimeStatus::CopyFailed, RuntimeStatus::ShortCopy}) {
    DispatchLifecycle lifecycle(facts_, envelope());
    ASSERT_EQ(lifecycle.submitted(), RuntimeStatus::Success);
    uint32_t forwards = 0;
    const auto forward = [&] {
      ++forwards;
      return true;
    };
    ASSERT_EQ(lifecycle.observe(true, true, forward), RuntimeStatus::Success);
    std::function<CopyResult(std::span<uint8_t>)> copy;
    if (expected != RuntimeStatus::CopyUnavailable) {
      copy = [expected](std::span<uint8_t> bytes) {
        return CopyResult{expected == RuntimeStatus::CopyFailed ? RuntimeStatus::CopyFailed
                                                                : RuntimeStatus::Success,
                          bytes.size() - 1};
      };
    }
    OwnedSnapshot result;
    EXPECT_EQ(lifecycle.snapshot(copy, result), expected);
    EXPECT_EQ(lifecycle.snapshot(copy, result), RuntimeStatus::InvalidTransition);
    EXPECT_EQ(result.facts, nullptr);
    EXPECT_TRUE(result.execution.report.empty());
    EXPECT_EQ(lifecycle.observe(true, true, forward), RuntimeStatus::InvalidTransition);
    EXPECT_EQ(forwards, 1u);
  }
}

TEST_F(ConSanRuntimeCore, IndependentSnapshotSurvivesSourcePoisonAndFactsUnloading) {
  const std::vector<uint8_t> original = raw_;
  OwnedSnapshot snapshot = copied_snapshot();
  std::ranges::fill(raw_, 0xcd);
  facts_.reset();
  header_.manifest.entry.name = "replaced_registry_entry";
  EXPECT_EQ(snapshot.execution.report, original);
  ASSERT_NE(snapshot.facts, nullptr);
  EXPECT_EQ(snapshot.facts->entry.name, "synthetic_codec_fixture");
}

TEST_F(ConSanRuntimeCore, SlowCollectorDoesNotHoldUpTheObserverOrAnotherCompletion) {
  DispatchLifecycle first(facts_, envelope());
  DispatchLifecycle second(facts_, envelope(2, 2));
  ASSERT_EQ(first.submitted(), RuntimeStatus::Success);
  ASSERT_EQ(second.submitted(), RuntimeStatus::Success);
  uint32_t forwarded = 0;
  ASSERT_EQ(first.observe(true, true,
                          [&] {
                            ++forwarded;
                            return true;
                          }),
            RuntimeStatus::Success);
  std::promise<void> copying;
  std::promise<void> permit_copy;
  std::future<void> entered = copying.get_future();
  std::future<void> proceed = permit_copy.get_future();
  OwnedSnapshot snapshot;
  std::future<RuntimeStatus> collector = std::async(std::launch::async, [&] {
    return first.snapshot(
        [&](std::span<uint8_t> bytes) {
          copying.set_value();
          proceed.wait();
          std::copy(raw_.begin(), raw_.end(), bytes.begin());
          return CopyResult{RuntimeStatus::Success, bytes.size()};
        },
        snapshot);
  });
  const std::future_status ready = entered.wait_for(std::chrono::seconds(5));
  if (ready != std::future_status::ready) {
    permit_copy.set_value();
    (void)collector.get();
    FAIL() << "collector did not enter the production copy callback";
  }
  EXPECT_EQ(snapshot.facts, nullptr);
  EXPECT_TRUE(snapshot.execution.report.empty());
  EXPECT_EQ(second.observe(true, true,
                           [&] {
                             ++forwarded;
                             return true;
                           }),
            RuntimeStatus::Success);
  EXPECT_EQ(forwarded, 2u);
  permit_copy.set_value();
  EXPECT_EQ(collector.get(), RuntimeStatus::Success);
  EXPECT_EQ(snapshot.facts, facts_);
  EXPECT_EQ(snapshot.execution.report, raw_);
}

TEST_F(ConSanRuntimeCore, ShortWritesAreRetriedAndMalformedRawRowsAreSavedUnchanged) {
  wire::write_u32(raw_, kReportHeaderBytes + wire::kRowStatus, UINT32_MAX);
  OwnedSnapshot snapshot = copied_snapshot();
  SnapshotWriter writer;
  ASSERT_EQ(writer.start(header_, sink(13)), RuntimeStatus::Success);
  ASSERT_EQ(writer.append(snapshot), RuntimeStatus::Success);
  EXPECT_EQ(publishes_, 0u);
  EXPECT_EQ(writer.finish(), RuntimeStatus::Success);
  EXPECT_EQ(writer.finish(), RuntimeStatus::Success);
  EXPECT_EQ(closes_, 1u);
  EXPECT_EQ(publishes_, 1u);
  const FailureOr<Capture> decoded = decode_capture(output_);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value().executions.front().report, raw_);
}

TEST_F(ConSanRuntimeCore, WriterFailureCannotPublishAnEndOrUndoIndependentCopyOwnership) {
  const OwnedSnapshot snapshot = copied_snapshot();
  for (const bool zero_progress : {false, true}) {
    output_.clear();
    closes_ = 0;
    publishes_ = 0;
    CaptureSink callbacks = sink();
    callbacks.write = [zero_progress](std::span<const uint8_t>) -> int64_t {
      return zero_progress ? 0 : -1;
    };
    // The sink function captures this seam, preserving the exact production
    // write_all loop while injecting the next underlying write result.
    bool failing = false;
    CaptureSink controlled = sink();
    const std::function<int64_t(std::span<const uint8_t>)> normal = controlled.write;
    controlled.write = [&](std::span<const uint8_t> bytes) {
      return failing ? callbacks.write(bytes) : normal(bytes);
    };
    SnapshotWriter actual;
    ASSERT_EQ(actual.start(header_, controlled), RuntimeStatus::Success);
    failing = true;
    const RuntimeStatus expected =
        zero_progress ? RuntimeStatus::ZeroWriteProgress : RuntimeStatus::WriteFailed;
    EXPECT_EQ(actual.append(snapshot), expected);
    EXPECT_EQ(actual.finish(), expected);
    EXPECT_EQ(actual.finish(), expected);
    EXPECT_EQ(publishes_, 0u);
    EXPECT_TRUE(decode_capture(output_).failed());
    EXPECT_EQ(snapshot.execution.report, raw_);
  }
}

TEST_F(ConSanRuntimeCore, CloseAndPublicationFailuresNeverReturnSavedSuccess) {
  const OwnedSnapshot snapshot = copied_snapshot();
  for (const bool close_failure : {true, false}) {
    close_succeeds_ = !close_failure;
    publish_succeeds_ = close_failure;
    closes_ = publishes_ = 0;
    output_.clear();
    SnapshotWriter writer;
    ASSERT_EQ(writer.start(header_, sink()), RuntimeStatus::Success);
    ASSERT_EQ(writer.append(snapshot), RuntimeStatus::Success);
    EXPECT_EQ(writer.finish(),
              close_failure ? RuntimeStatus::CloseFailed : RuntimeStatus::PublishFailed);
    EXPECT_EQ(closes_, 1u);
    EXPECT_EQ(publishes_, close_failure ? 0u : 1u);
  }
}

TEST_F(ConSanRuntimeCore, AdmissionFailureCancelsEndEvenAfterAnEarlierSuccessfulSnapshot) {
  SnapshotWriter writer;
  ASSERT_EQ(writer.start(header_, sink()), RuntimeStatus::Success);
  ASSERT_EQ(writer.append(copied_snapshot()), RuntimeStatus::Success);
  writer.cancel(RuntimeStatus::InvalidFacts);
  EXPECT_EQ(writer.finish(), RuntimeStatus::InvalidFacts);
  EXPECT_EQ(writer.finish(), RuntimeStatus::InvalidFacts);
  EXPECT_EQ(closes_, 1u);
  EXPECT_EQ(publishes_, 0u);
  EXPECT_TRUE(decode_capture(output_).failed());
}

TEST_F(ConSanRuntimeCore, WriterRequiresTheOriginalExecutionsOwningFacts) {
  OwnedSnapshot snapshot = copied_snapshot();
  auto other = std::make_shared<Manifest>(*snapshot.facts);
  other->entry.name = "other_execution";
  snapshot.facts = other;
  SnapshotWriter writer;
  ASSERT_EQ(writer.start(header_, sink()), RuntimeStatus::Success);
  EXPECT_EQ(writer.append(snapshot), RuntimeStatus::InvalidFacts);
  EXPECT_EQ(writer.finish(), RuntimeStatus::InvalidFacts);
  EXPECT_EQ(publishes_, 0u);
  EXPECT_TRUE(decode_capture(output_).failed());
}

} // namespace
} // namespace rocjitsu::consan
