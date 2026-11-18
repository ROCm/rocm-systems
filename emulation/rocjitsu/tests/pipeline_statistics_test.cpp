// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"

#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "util/statistics.h"

#include <gtest/gtest.h>

#include <charconv>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::test::cdna5;

std::shared_ptr<stats::Scope> make_pipeline_scope(std::vector<std::string> include = {}) {
  auto config = std::make_shared<stats::Config>();
  config->enabled = true;
  config->default_domain = "model_events";
  config->include = std::move(include);
  return std::make_shared<stats::Scope>("cu", "cu", std::move(config));
}

int64_t read_pipeline_statistic(const stats::Scope &scope, std::string_view name) {
  std::ostringstream output;
  scope.write_samples(output);
  std::istringstream lines(output.str());
  std::string line;
  while (std::getline(lines, line)) {
    flexbuffers::Builder builder;
    if (!plugin_detail::flexbuffer_from_json(line, builder)) {
      ADD_FAILURE() << "invalid statistics JSON: " << line;
      continue;
    }
    const auto record = flexbuffers::GetRoot(builder.GetBuffer()).AsMap();
    if (record["metric_id"].AsString().str() != scope.name() + ":" + std::string(name))
      continue;
    const std::string value = record["value"].AsString().str();
    int64_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    EXPECT_EQ(result.ec, std::errc{});
    EXPECT_EQ(result.ptr, value.data() + value.size());
    return parsed;
  }
  ADD_FAILURE() << "missing statistic " << name;
  return 0;
}

class StatisticsMemoryInstruction final : public Instruction {
public:
  StatisticsMemoryInstruction() : Instruction("test_statistics_memory", nullptr) {
    flags_ |= MEMORY_OP;
    auto state = std::make_unique<ScalarMemState>();
    state->is_load = true;
    state->wait_counter_type = WaitCounterType::KMCNT;
    set_data(std::move(state));
  }
};

class StatisticsTestPipeline final : public MemoryPipeline {
public:
  StatisticsTestPipeline() : MemoryPipeline(WaitCounterType::KMCNT) {}

  VmAccessOutcome access_outcome = VmAccessOutcome::Complete;
  MemoryAccessCompletion completion_outcome = MemoryAccessCompletion::Complete;
  uint32_t access_attempts = 0;

  void complete_next() {
    ASSERT_FALSE(completions_.empty());
    auto completion = std::move(completions_.front());
    completions_.erase(completions_.begin());
    completion();
  }

protected:
  VmAccessOutcome initiate_access(Instruction &, Wavefront &) override {
    ++access_attempts;
    return access_outcome;
  }
  MemoryAccessCompletion complete_access(Instruction &, Wavefront &,
                                         MemoryAccessDeferredCompletion complete) override {
    if (completion_outcome == MemoryAccessCompletion::Deferred)
      completions_.push_back(std::move(complete));
    return completion_outcome;
  }

private:
  std::vector<MemoryAccessDeferredCompletion> completions_;
};

class PipelineStatisticsTest : public testing::Test {
protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(sim_.initialize());
    cu_ = sim_.cu();
    wave_ = sim_.dispatch_scratch_wf();
    ASSERT_NE(wave_, nullptr);
    scope_ = make_pipeline_scope();
    pipeline_.set_statistics(scope_, "test");
  }

  int64_t metric(std::string_view name) const {
    return read_pipeline_statistic(*scope_, "memory.test." + std::string(name));
  }

  Gfx1250Sim sim_;
  ComputeUnitCore *cu_ = nullptr;
  Wavefront *wave_ = nullptr;
  std::shared_ptr<stats::Scope> scope_;
  StatisticsTestPipeline pipeline_;
};

TEST_F(PipelineStatisticsTest, ImmediateOutcomesSettleOwnershipAndWaitCounters) {
  EXPECT_EQ(pipeline_.issue(new StatisticsMemoryInstruction, *wave_), VmAccessOutcome::Complete);
  pipeline_.access_outcome = VmAccessOutcome::Faulted;
  EXPECT_EQ(pipeline_.issue(new StatisticsMemoryInstruction, *wave_), VmAccessOutcome::Faulted);
  pipeline_.access_outcome = VmAccessOutcome::Complete;
  pipeline_.completion_outcome = MemoryAccessCompletion::Failed;
  EXPECT_EQ(pipeline_.issue(new StatisticsMemoryInstruction, *wave_), VmAccessOutcome::Malformed);
  pipeline_.access_outcome = VmAccessOutcome::Unavailable;
  EXPECT_EQ(pipeline_.issue(new StatisticsMemoryInstruction, *wave_), VmAccessOutcome::Unavailable);
  EXPECT_EQ(metric("accepted"), 4);
  EXPECT_EQ(metric("completed"), 1);
  EXPECT_EQ(metric("failed"), 2);
  EXPECT_EQ(metric("cancelled"), 1);
  EXPECT_EQ(metric("pending"), 0);
  EXPECT_EQ(metric("retry_depth"), 0);
  EXPECT_EQ(metric("retry_attempts"), 0);
  EXPECT_EQ(metric("unavailable_attempts"), 1);
  EXPECT_TRUE(wave_->wait_counters().empty());
}

TEST_F(PipelineStatisticsTest, RetriedUnavailableOperationHasOneAcceptedAndTerminalOutcome) {
  pipeline_.access_outcome = VmAccessOutcome::Unavailable;
  ASSERT_EQ(pipeline_.issue_deferred(new StatisticsMemoryInstruction, *wave_),
            VmAccessOutcome::Complete);
  EXPECT_EQ(metric("accepted"), 1);
  EXPECT_EQ(metric("pending"), 1);
  EXPECT_EQ(metric("retry_depth"), 1);
  pipeline_.tick();
  EXPECT_EQ(metric("retry_attempts"), 1);
  EXPECT_EQ(metric("unavailable_attempts"), 2);
  EXPECT_EQ(metric("pending"), 1);
  pipeline_.access_outcome = VmAccessOutcome::Complete;
  pipeline_.tick();
  EXPECT_EQ(metric("accepted"), 1);
  EXPECT_EQ(metric("completed"), 1);
  EXPECT_EQ(metric("pending"), 0);
  EXPECT_EQ(metric("retry_depth"), 0);
  EXPECT_EQ(metric("retry_attempts"), 2);
  EXPECT_EQ(wave_->state(), WfState::RUNNING);
  EXPECT_TRUE(wave_->wait_counters().empty());
}

TEST_F(PipelineStatisticsTest, RewindingExternallyAttemptedUnavailableLoadCancelsItOnce) {
  constexpr uint64_t kIssuePc = 0x1234;
  wave_->pc = kIssuePc;
  pipeline_.defer_unavailable(new StatisticsMemoryInstruction, *wave_);
  wave_->pc += 4;
  ASSERT_TRUE(pipeline_.rewind_unavailable_load(*wave_));
  EXPECT_EQ(wave_->pc, kIssuePc);
  EXPECT_EQ(metric("accepted"), 1);
  EXPECT_EQ(metric("cancelled"), 1);
  EXPECT_EQ(metric("unavailable_attempts"), 1);
  EXPECT_EQ(metric("retry_attempts"), 0);
  EXPECT_EQ(metric("pending"), 0);
  EXPECT_EQ(metric("retry_depth"), 0);
  EXPECT_TRUE(wave_->wait_counters().empty());
  EXPECT_FALSE(pipeline_.rewind_unavailable_load(*wave_));
  EXPECT_EQ(metric("cancelled"), 1);
}

TEST_F(PipelineStatisticsTest, FaultCallbackCanCancelAnotherQueuedOperation) {
  auto *peer = cu_->dispatch_wf(1, 0, kGfx1250ScalarSlots, 32);
  ASSERT_NE(peer, nullptr);
  pipeline_.access_outcome = VmAccessOutcome::Unavailable;
  ASSERT_EQ(pipeline_.issue_deferred(new StatisticsMemoryInstruction, *wave_),
            VmAccessOutcome::Complete);
  ASSERT_EQ(pipeline_.issue_deferred(new StatisticsMemoryInstruction, *peer),
            VmAccessOutcome::Complete);
  pipeline_.set_fault_handler(
      [&](Wavefront &, VmAccessOutcome, uint64_t) { pipeline_.cancel(*peer); });
  pipeline_.access_outcome = VmAccessOutcome::Faulted;
  pipeline_.tick();
  EXPECT_EQ(metric("accepted"), 2);
  EXPECT_EQ(metric("failed"), 1);
  EXPECT_EQ(metric("cancelled"), 1);
  EXPECT_EQ(metric("pending"), 0);
  EXPECT_EQ(metric("retry_depth"), 0);
  EXPECT_EQ(metric("retry_attempts"), 1);
  EXPECT_TRUE(pipeline_.empty());
}

TEST_F(PipelineStatisticsTest, CancelledDeferredCompletionCannotSettleReusedWaveSlot) {
  pipeline_.completion_outcome = MemoryAccessCompletion::Deferred;
  ASSERT_EQ(pipeline_.issue(new StatisticsMemoryInstruction, *wave_), VmAccessOutcome::Complete);
  EXPECT_EQ(metric("pending"), 1);
  EXPECT_EQ(metric("retry_depth"), 0);
  const uint64_t generation = wave_->dispatch_generation();
  pipeline_.cancel(*wave_);
  EXPECT_EQ(metric("cancelled"), 1);
  EXPECT_EQ(metric("pending"), 0);
  cu_->free_wavefront_resources(*wave_);
  auto *replacement = cu_->dispatch_wf(1, 0, kGfx1250ScalarSlots, 32);
  ASSERT_EQ(replacement, wave_);
  ASSERT_NE(replacement->dispatch_generation(), generation);
  ASSERT_EQ(pipeline_.issue(new StatisticsMemoryInstruction, *replacement),
            VmAccessOutcome::Complete);
  const auto remaining = replacement->wait_counters().kmcnt;
  pipeline_.complete_next();
  EXPECT_EQ(metric("pending"), 1);
  EXPECT_EQ(metric("cancelled"), 1);
  EXPECT_EQ(metric("completed"), 0);
  EXPECT_EQ(replacement->wait_counters().kmcnt, remaining);
  pipeline_.complete_next();
  EXPECT_EQ(metric("accepted"), 2);
  EXPECT_EQ(metric("completed"), 1);
  EXPECT_EQ(metric("pending"), 0);
  EXPECT_TRUE(replacement->wait_counters().empty());
}

TEST_F(PipelineStatisticsTest, DeferredSlotReuseCancelsOnlyTheOldGeneration) {
  pipeline_.completion_outcome = MemoryAccessCompletion::Deferred;
  ASSERT_EQ(pipeline_.issue(new StatisticsMemoryInstruction, *wave_), VmAccessOutcome::Complete);
  const uint64_t generation = wave_->dispatch_generation();
  cu_->free_wavefront_resources(*wave_);
  auto *replacement = cu_->dispatch_wf(1, 0, kGfx1250ScalarSlots, 32);
  ASSERT_EQ(replacement, wave_);
  ASSERT_NE(replacement->dispatch_generation(), generation);
  ASSERT_EQ(pipeline_.issue(new StatisticsMemoryInstruction, *replacement),
            VmAccessOutcome::Complete);
  EXPECT_EQ(metric("pending"), 2);
  pipeline_.complete_next();
  EXPECT_EQ(metric("pending"), 1);
  EXPECT_EQ(metric("cancelled"), 1);
  EXPECT_EQ(replacement->wait_counters().kmcnt, 1);
  pipeline_.complete_next();
  EXPECT_EQ(metric("completed"), 1);
  EXPECT_EQ(metric("pending"), 0);
  EXPECT_TRUE(replacement->wait_counters().empty());
}

TEST_F(PipelineStatisticsTest, StaleRetryCancelsWithoutRetryingTheReplacementGeneration) {
  pipeline_.access_outcome = VmAccessOutcome::Unavailable;
  ASSERT_EQ(pipeline_.issue_deferred(new StatisticsMemoryInstruction, *wave_),
            VmAccessOutcome::Complete);
  const uint64_t generation = wave_->dispatch_generation();
  cu_->free_wavefront_resources(*wave_);
  auto *replacement = cu_->dispatch_wf(1, 0, kGfx1250ScalarSlots, 32);
  ASSERT_EQ(replacement, wave_);
  ASSERT_NE(replacement->dispatch_generation(), generation);
  pipeline_.tick();
  EXPECT_EQ(pipeline_.access_attempts, 1u);
  EXPECT_EQ(metric("accepted"), 1);
  EXPECT_EQ(metric("cancelled"), 1);
  EXPECT_EQ(metric("pending"), 0);
  EXPECT_EQ(metric("retry_depth"), 0);
  EXPECT_EQ(metric("retry_attempts"), 0);
  EXPECT_TRUE(replacement->wait_counters().empty());
}

TEST_F(PipelineStatisticsTest, ExplicitFinalizationSettlesDirectWaveOperationsExactlyOnce) {
  pipeline_.completion_outcome = MemoryAccessCompletion::Deferred;
  ASSERT_EQ(pipeline_.issue(new StatisticsMemoryInstruction, *wave_), VmAccessOutcome::Complete);
  pipeline_.access_outcome = VmAccessOutcome::Unavailable;
  ASSERT_EQ(pipeline_.issue_deferred(new StatisticsMemoryInstruction, *wave_),
            VmAccessOutcome::Complete);
  EXPECT_EQ(metric("accepted"), 2);
  EXPECT_EQ(metric("pending"), 2);
  EXPECT_EQ(metric("retry_depth"), 1);
  pipeline_.finish_statistics();
  pipeline_.finish_statistics();
  EXPECT_EQ(metric("cancelled"), 2);
  EXPECT_EQ(metric("completed"), 0);
  EXPECT_EQ(metric("pending"), 0);
  EXPECT_EQ(metric("retry_depth"), 0);

  const uint64_t generation = wave_->dispatch_generation();
  cu_->free_wavefront_resources(*wave_);
  auto *replacement = cu_->dispatch_wf(1, 0, kGfx1250ScalarSlots, 32);
  ASSERT_EQ(replacement, wave_);
  ASSERT_NE(replacement->dispatch_generation(), generation);
  pipeline_.complete_next();
  pipeline_.tick();
  EXPECT_TRUE(pipeline_.empty());
  EXPECT_TRUE(replacement->wait_counters().empty());
  EXPECT_EQ(metric("cancelled"), 2);
  EXPECT_EQ(metric("completed"), 0);
  EXPECT_EQ(metric("pending"), 0);
  EXPECT_EQ(metric("retry_depth"), 0);
}

TEST_F(PipelineStatisticsTest, DestructionSettlesPendingRetry) {
  auto scope = make_pipeline_scope();
  {
    StatisticsTestPipeline pipeline;
    pipeline.set_statistics(scope, "test");
    pipeline.access_outcome = VmAccessOutcome::Unavailable;
    ASSERT_EQ(pipeline.issue_deferred(new StatisticsMemoryInstruction, *wave_),
              VmAccessOutcome::Complete);
    EXPECT_EQ(read_pipeline_statistic(*scope, "memory.test.pending"), 1);
  }
  EXPECT_EQ(read_pipeline_statistic(*scope, "memory.test.cancelled"), 1);
  EXPECT_EQ(read_pipeline_statistic(*scope, "memory.test.pending"), 0);
  EXPECT_EQ(read_pipeline_statistic(*scope, "memory.test.retry_depth"), 0);
}

class SchedulerStatisticsTest : public testing::Test {
protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(sim_.initialize());
    cu_ = sim_.cu();
    scope_ = make_pipeline_scope();
    cu_->set_statistics(scope_);
    sim_.memory->write32(kCode, cdna5::build_sopp(cdna5::kSNop)[0]);
    sim_.memory->write32(kCode + 4, cdna5::build_sopp(cdna5::kSEndpgm)[0]);
    cu_->begin_workgroup(0, 0, 1);
    wave_ = cu_->dispatch_wf(0, kCode, kGfx1250ScalarSlots, 32);
    ASSERT_NE(wave_, nullptr);
  }
  int64_t metric(std::string_view name) const {
    return read_pipeline_statistic(*scope_, "scheduler." + std::string(name));
  }
  void begin_wait() {
    wave_->wait_counters().increment(WaitCounterType::KMCNT);
    wave_->wait_for({.kmcnt = 0});
  }

  static constexpr uint64_t kCode = 0x1000;
  Gfx1250Sim sim_;
  ComputeUnitCore *cu_ = nullptr;
  Wavefront *wave_ = nullptr;
  std::shared_ptr<stats::Scope> scope_;
};

TEST_F(SchedulerStatisticsTest, ResolvedWaitcntIsObservedAfterStateUpdate) {
  begin_wait();
  cu_->step();
  EXPECT_EQ(wave_->pc, kCode);
  EXPECT_EQ(metric("waitcnt_wave_steps"), 1);
  wave_->wait_counters().decrement(WaitCounterType::KMCNT);
  ASSERT_EQ(wave_->state(), WfState::WAITCNT);
  cu_->step();
  EXPECT_EQ(wave_->pc, kCode + 4);
  EXPECT_EQ(metric("waitcnt_wave_steps"), 1);
  EXPECT_EQ(metric("active_wave_steps"), 2);
  EXPECT_EQ(metric("steps"), 2);
}

TEST_F(SchedulerStatisticsTest, DebuggerAndRuntimePausesDoNotChargeWaitcnt) {
  begin_wait();
  wave_->set_debug_halted(true);
  cu_->step();
  EXPECT_EQ(metric("debug_paused_wave_steps"), 1);
  EXPECT_EQ(metric("waitcnt_wave_steps"), 0);
  wave_->set_debug_halted(false);
  wave_->set_runtime_suspended(true);
  cu_->step();
  EXPECT_EQ(metric("runtime_paused_wave_steps"), 1);
  EXPECT_EQ(metric("waitcnt_wave_steps"), 0);
  wave_->set_runtime_suspended(false);
  cu_->step();
  EXPECT_EQ(metric("waitcnt_wave_steps"), 1);
  EXPECT_EQ(metric("active_wave_steps"), 3);
}

TEST_F(SchedulerStatisticsTest, ExportAndParameterWaitsStopCountingWhenReady) {
  wave_->set_status_raw(wave_->status_raw() & ~Wavefront::kStatusExportReadyMask);
  wave_->set_state(WfState::EXPORT_WAIT);
  cu_->step();
  EXPECT_EQ(wave_->pc, kCode);
  EXPECT_EQ(metric("export_wait_wave_steps"), 1);
  EXPECT_EQ(metric("param_wait_wave_steps"), 0);
  wave_->set_status_raw(wave_->status_raw() | Wavefront::kStatusExportReadyMask);
  cu_->step();
  EXPECT_EQ(wave_->pc, kCode + 4);
  EXPECT_EQ(metric("export_wait_wave_steps"), 1);

  wave_->set_status_raw(wave_->status_raw() & ~Wavefront::kStatusParamReadyMask);
  wave_->set_state(WfState::PARAM_WAIT);
  cu_->step();
  EXPECT_EQ(wave_->pc, kCode + 4);
  EXPECT_EQ(metric("param_wait_wave_steps"), 1);
  EXPECT_EQ(metric("export_wait_wave_steps"), 1);
  wave_->set_status_raw(wave_->status_raw() | Wavefront::kStatusParamReadyMask);
  cu_->step();
  EXPECT_TRUE(wave_->is_halted());
  EXPECT_EQ(metric("param_wait_wave_steps"), 1);
}

TEST_F(SchedulerStatisticsTest, SleepAndInstructionReadinessAreSeparate) {
  wave_->set_sleep_cycles(2);
  wave_->delay_nop_issue(2);
  cu_->step();
  cu_->step();
  EXPECT_EQ(metric("not_ready_wave_steps"), 2);
  EXPECT_EQ(metric("sleep_wave_steps"), 0);
  EXPECT_EQ(wave_->sleep_cycles(), 2u);
  cu_->step();
  cu_->step();
  EXPECT_EQ(metric("sleep_wave_steps"), 2);
  EXPECT_EQ(wave_->pc, kCode);
  cu_->step();
  EXPECT_EQ(wave_->pc, kCode + 4);
  EXPECT_EQ(metric("active_wave_steps"), 5);
}

TEST_F(SchedulerStatisticsTest, EndingWaveIsExcludedOnceItRetiresDuringStateUpdate) {
  wave_->wait_counters().increment(WaitCounterType::KMCNT);
  wave_->end();
  ASSERT_EQ(wave_->state(), WfState::ENDING);
  cu_->step();
  EXPECT_EQ(metric("ending_wave_steps"), 1);
  wave_->wait_counters().decrement(WaitCounterType::KMCNT);
  cu_->step();
  EXPECT_TRUE(wave_->is_halted());
  EXPECT_EQ(metric("ending_wave_steps"), 1);
  EXPECT_EQ(metric("active_wave_steps"), 1);
  EXPECT_EQ(metric("steps"), 2);
}

TEST_F(SchedulerStatisticsTest, GwsAndUnavailableMemoryHaveDistinctReasons) {
  cu_->set_gws_wait_limit(0);
  cu_->gws_sema_p(*wave_, 0);
  ASSERT_EQ(wave_->state(), WfState::GWS_WAIT);
  cu_->step();
  EXPECT_EQ(metric("gws_wait_wave_steps"), 1);
  cu_->gws_sema_v(*wave_, 0);
  StatisticsTestPipeline pipeline;
  pipeline.access_outcome = VmAccessOutcome::Unavailable;
  ASSERT_EQ(pipeline.issue_deferred(new StatisticsMemoryInstruction, *wave_),
            VmAccessOutcome::Complete);
  cu_->step();
  EXPECT_EQ(metric("vm_retry_wave_steps"), 1);
  EXPECT_EQ(metric("gws_wait_wave_steps"), 1);
  pipeline.access_outcome = VmAccessOutcome::Complete;
  pipeline.tick();
  cu_->step();
  EXPECT_EQ(wave_->pc, kCode + 4);
  EXPECT_EQ(metric("vm_retry_wave_steps"), 1);
}

TEST_F(SchedulerStatisticsTest, SelectionDoesNotBecomeAStallAndSingleStepOverridesDelays) {
  cu_->run_issue_cycle(0);
  EXPECT_EQ(wave_->pc, kCode);
  EXPECT_EQ(metric("active_wave_steps"), 1);
  EXPECT_EQ(metric("not_ready_wave_steps"), 0);
  wave_->delay_nop_issue(10);
  wave_->set_sleep_cycles(10);
  wave_->set_debug_single_step(true);
  cu_->step();
  EXPECT_EQ(wave_->pc, kCode + 4);
  EXPECT_EQ(wave_->sleep_cycles(), 0u);
  EXPECT_EQ(metric("not_ready_wave_steps"), 0);
  EXPECT_EQ(metric("sleep_wave_steps"), 0);
}

TEST_F(SchedulerStatisticsTest, BarrierObservationExcludesPausedPeer) {
  cu_->begin_workgroup(0, 0, 2);
  auto *peer = cu_->dispatch_wf(0, kCode, kGfx1250ScalarSlots, 32);
  ASSERT_NE(peer, nullptr);
  peer->set_debug_halted(true);
  wave_->barrier_wait(kWorkgroupBarrierId);
  ASSERT_EQ(wave_->state(), WfState::BARRIER);
  cu_->step();
  EXPECT_EQ(metric("barrier_wave_steps"), 1);
  EXPECT_EQ(metric("debug_paused_wave_steps"), 1);
  EXPECT_EQ(metric("active_wave_steps"), 2);
}

} // namespace
