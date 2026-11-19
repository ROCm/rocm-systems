// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "execution_plugin_test_util.h"
#include "rocjitsu/stats/execution_statistics.h"
#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "scoped_temp.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <thread>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::test::execution_plugin;

uint64_t execution_value(const stats::Scope &scope, std::string_view name) {
  std::ostringstream output;
  scope.write_samples(output);
  std::istringstream lines(output.str());
  std::string line;
  while (std::getline(lines, line)) {
    flexbuffers::Builder builder;
    if (!plugin_detail::flexbuffer_from_json(line, builder)) {
      ADD_FAILURE() << line;
      continue;
    }
    const auto record = flexbuffers::GetRoot(builder.GetBuffer()).AsMap();
    if (record["record"].AsString().str() == "sample" &&
        record["metric_id"].AsString().str().ends_with(":" + std::string(name)))
      return std::stoull(record["value"].AsString().str());
  }
  ADD_FAILURE() << "missing metric " << name;
  return 0;
}

class ExecutionStatisticsTest : public testing::Test {
protected:
  void SetUp() override { ASSERT_TRUE(temporary_.initialize("rocjitsu-execution-statistics-")); }

  void initialize(std::vector<std::string> include = {},
                  std::vector<std::string> scopes = {"component", "dispatch"}) {
    StatisticsConfig config;
    config.enabled = true;
    config.directory = temporary_.path();
    config.include = std::move(include);
    config.scopes = std::move(scopes);
    auto result = StatisticsSession::create(config, "node", "functional");
    ASSERT_TRUE(result.succeeded());
    session_ = std::move(result).value();
  }

  std::string finish() {
    session_->mark_run_complete();
    session_->finish();
    std::ifstream input(session_->output_path());
    return {std::istreambuf_iterator<char>(input), {}};
  }

  test::ScopedTempDirectory temporary_;
  std::shared_ptr<StatisticsSession> session_;
};

TEST_F(ExecutionStatisticsTest, CountsActualExecutionAfterPluginReplacement) {
  ASSERT_NO_FATAL_FAILURE(initialize());
  PluginFixture fixture;
  ASSERT_NO_FATAL_FAILURE(fixture.initialize(1));
  fixture.soc->set_statistics(session_, 0);
  ASSERT_EQ(fixture.soc->plugin_group().num_plugins(), 1u);
  // The shared empty singleton must remain immutable, including explicit replacement.
  fixture.soc->set_plugin_group(ExecutionPluginGroup::empty_group());
  EXPECT_TRUE(ExecutionPluginGroup::empty_group()->empty());
  fixture.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<OrderingPlugin>();
  auto *observer_ptr = observer.get();
  ASSERT_TRUE(fixture.plugin_group_->add(std::move(observer)));
  fixture.soc->set_plugin_group(fixture.plugin_group_);
  fixture.soc->set_plugin_group(fixture.plugin_group_);
  ASSERT_EQ(fixture.plugin_group_->num_plugins(), 2u);
  fixture.plugin_group_->onInit();
  const uint32_t code[] = {vop1_encode(1, 0, 128), S_NOP, S_ENDPGM};
  fixture.run_kernel(code, std::size(code));
  fixture.shutdown();
  const auto scope = session_->scope("execution", 0);
  ASSERT_NE(scope, nullptr);
  EXPECT_EQ(execution_value(*scope, "execution.wave_instructions.vector"), 1u);
  EXPECT_EQ(execution_value(*scope, "execution.wave_instructions.control"), 2u);
  EXPECT_EQ(execution_value(*scope, "execution.dispatches_started"), 1u);
  EXPECT_EQ(execution_value(*scope, "execution.dispatches_ended"), 1u);
  EXPECT_EQ(execution_value(*scope, "host.dispatch.duration.count"), 1u);
  EXPECT_EQ(execution_value(*scope, "execution.dispatches_incomplete"), 0u);
  const std::vector<HookEvent>::const_iterator dispatch = std::ranges::find(
      observer_ptr->events, HookEvent::DISPATCH_PACKET_PROCESSED, &HookEvent::kind);
  ASSERT_NE(dispatch, observer_ptr->events.end());
  const std::shared_ptr<stats::Scope> dispatch_scope =
      session_->scope("dispatch/" + std::to_string(dispatch->dispatch_id), 0, "dispatch");
  ASSERT_NE(dispatch_scope, nullptr);
  EXPECT_EQ(execution_value(*dispatch_scope, "execution.wave_instructions.vector"), 1u);
  EXPECT_EQ(execution_value(*dispatch_scope, "execution.wave_instructions.control"), 2u);
  EXPECT_EQ(std::count_if(observer_ptr->events.begin(), observer_ptr->events.end(),
                          [](const HookEvent &event) { return event.kind == HookEvent::SHUTDOWN; }),
            1);
  fixture.soc->release_vm_resources();
  EXPECT_NE(finish().find("\"session_end\",\"complete\":true"), std::string::npos);
}

TEST_F(ExecutionStatisticsTest, RebindingSessionPreservesUserPluginsAndChangesSelection) {
  ASSERT_NO_FATAL_FAILURE(initialize({"execution.dispatches_started"}, {"component"}));
  const std::shared_ptr<StatisticsSession> original = session_;
  PluginFixture fixture;
  ASSERT_NO_FATAL_FAILURE(fixture.initialize(1));
  OrderingPlugin *observer = fixture.attach_ordering_plugin();
  fixture.soc->set_statistics(original, 0);
  ASSERT_EQ(fixture.plugin_group_->num_plugins(), 2u);
  const std::shared_ptr<stats::Scope> original_scope = original->scope("execution", 0);
  fixture.soc->set_statistics(nullptr, 0);
  ASSERT_EQ(fixture.plugin_group_->num_plugins(), 1u);
  EXPECT_EQ(fixture.soc->statistics(), nullptr);
  const uint32_t code[] = {vop1_encode(1, 0, 128), S_NOP, S_ENDPGM};
  const uint64_t kernel = fixture.write_kernel(0x1000, code, std::size(code));
  rocjitsu::test::AqlQueue queue(fixture.mem, fixture.cp());
  queue.dispatch(kernel, 64);
  // Stop at durable retirement: stepping the drained fixture again terminates
  // its engine before the second dispatch can be submitted.
  for (uint32_t step = 0;
       step < 10000 && fixture.mem->read64(rocjitsu::test::AqlQueue::DEFAULT_READ_PTR_ADDR) == 0;
       ++step)
    ASSERT_TRUE(fixture.engine->step());
  ASSERT_EQ(fixture.mem->read64(rocjitsu::test::AqlQueue::DEFAULT_READ_PTR_ADDR), 1u);
  ASSERT_FALSE(fixture.cu()->has_active_wfs());
  ASSERT_FALSE(fixture.cp()->has_pending_aql_dispatch_work());
  EXPECT_EQ(execution_value(*original_scope, "execution.dispatches_started"), 0u);

  ASSERT_NO_FATAL_FAILURE(initialize({"execution.wave_instructions.*"}, {"component"}));
  fixture.soc->set_statistics(session_, 0);
  fixture.soc->set_statistics(session_, 0);
  ASSERT_EQ(fixture.plugin_group_->num_plugins(), 2u);
  queue.dispatch(kernel, 64);
  fixture.run_until_idle();
  fixture.shutdown();
  const std::shared_ptr<stats::Scope> replacement_scope = session_->scope("execution", 0);
  EXPECT_EQ(execution_value(*replacement_scope, "execution.wave_instructions.vector"), 1u);
  EXPECT_EQ(execution_value(*replacement_scope, "execution.wave_instructions.control"), 2u);
  EXPECT_EQ(execution_value(*original_scope, "execution.dispatches_started"), 0u);
  EXPECT_EQ(
      std::ranges::count(observer->events, HookEvent::DISPATCH_PACKET_PROCESSED, &HookEvent::kind),
      2);
  EXPECT_EQ(std::ranges::count(observer->events, HookEvent::SHUTDOWN, &HookEvent::kind), 1);
}

TEST_F(ExecutionStatisticsTest, SecondaryGpuFinalizesUnfinishedDispatchWithoutUserPlugins) {
  ASSERT_NO_FATAL_FAILURE(initialize({"execution.dispatches_incomplete"}, {"component"}));
  PluginFixture fixture;
  ASSERT_NO_FATAL_FAILURE(fixture.initialize(1));
  fixture.soc->set_statistics(session_, 1);
  KernelDispatchInfo info{};
  info.dispatch_id = 9;
  info.workgroup_count = 1;
  fixture.soc->plugin_group().onAmdgpuDispatchPacketProcessed(info);
  fixture.soc->plugin_group().onAmdgpuDispatchExecutionBegin(9);
  fixture.soc->release_vm_resources();
  fixture.soc->release_vm_resources();
  const auto scope = session_->scope("execution", 1);
  EXPECT_EQ(execution_value(*scope, "execution.dispatches_incomplete"), 1u);
  EXPECT_NE(finish().find("\"session_end\",\"complete\":false"), std::string::npos);
}

TEST_F(ExecutionStatisticsTest, ZeroGridPacketDoesNotLeaveUnfinishedExecution) {
  ASSERT_NO_FATAL_FAILURE(initialize());
  ExecutionStatistics adapter(session_, 0);
  KernelDispatchInfo info{};
  info.dispatch_id = 1;
  adapter.onAmdgpuDispatchPacketProcessed(info);
  adapter.onShutdown();
  EXPECT_EQ(execution_value(*session_->scope("execution", 0), "execution.dispatches_incomplete"),
            0u);
  EXPECT_NE(finish().find("\"session_end\",\"complete\":true"), std::string::npos);
}

TEST_F(ExecutionStatisticsTest, InstructionOnlyHooksCountConcurrentAcceptedIssues) {
  ASSERT_NO_FATAL_FAILURE(initialize({"execution.wave_instructions.*"}, {"component"}));
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::make_unique<ExecutionStatistics>(session_, 0)));
  EXPECT_TRUE(group->supports_async_instructions());
  EXPECT_FALSE(group->requires_serial_hot_hooks());
  EXPECT_FALSE(group->observes_after_execute_instruction());
  EXPECT_FALSE(group->observes_vgpr_reads());
  EXPECT_FALSE(group->observes_vgpr_writes());
  PluginFixture fixture;
  ASSERT_NO_FATAL_FAILURE(fixture.initialize(4));
  fixture.soc->set_plugin_group(group);
  std::array<std::jthread, 4> threads;
  for (auto &thread : threads) {
    auto *wave = fixture.cu()->dispatch_wf(1, 0, 32, 32);
    ASSERT_NE(wave, nullptr);
    if (&thread == &threads.front())
      wave->set_plugin_state(0, std::make_unique<WavefrontState>());
    thread = std::jthread([&, wave] {
      Instruction instruction("v_mfma_test", nullptr);
      for (uint32_t index = 0; index < 1000; ++index)
        group->onAmdgpuAsyncInstructionIssued(0, instruction, *wave);
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(execution_value(*session_->scope("execution", 0), "execution.wave_instructions.matrix"),
            4000u);
  group->onShutdown();
}

TEST_F(ExecutionStatisticsTest, CacheOnlySelectionDoesNotRequestExecutionAdapter) {
  ASSERT_NO_FATAL_FAILURE(initialize({"cache.*"}));
  EXPECT_FALSE(ExecutionStatistics::requested(*session_));
}

} // namespace
