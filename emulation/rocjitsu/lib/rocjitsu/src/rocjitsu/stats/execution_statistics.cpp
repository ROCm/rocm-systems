// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/stats/execution_statistics.h"

#include <format>

namespace rocjitsu {

bool ExecutionStatistics::requested(const StatisticsSession &session) {
  for (size_t index = 0; index < plugins::kInstructionFamilyCount; ++index)
    if (session.selected(std::string("execution.wave_instructions.") +
                         std::string(plugins::instruction_family_name(
                             static_cast<plugins::InstructionFamily>(index)))))
      return true;
  for (const char *name : {"execution.dispatches_started", "execution.dispatches_ended",
                           "execution.dispatches_incomplete", "execution.waves_cancelled",
                           "execution.waves_halted", "host.dispatch.duration"})
    if (session.selected(name))
      return true;
  return false;
}

ExecutionStatistics::Counts
ExecutionStatistics::bind_counts(const std::shared_ptr<stats::Scope> &scope) {
  Counts counts{};
  if (!scope)
    return counts;
  for (size_t index = 0; index < counts.size(); ++index) {
    const std::string name = std::string("execution.wave_instructions.") +
                             std::string(plugins::instruction_family_name(
                                 static_cast<plugins::InstructionFamily>(index)));
    counts[index] =
        scope->counter(name, "wave_instructions",
                       "Synchronous before-execute and accepted async-issue observations");
  }
  return counts;
}

std::shared_ptr<stats::RegisteredHistogram>
ExecutionStatistics::bind_duration(const std::shared_ptr<stats::Scope> &scope) {
  if (!scope)
    return {};
  constexpr std::array bounds{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10., 100.};
  return scope->histogram(
      "host.dispatch.duration", "seconds",
      "Host steady-clock interval from dispatch execution begin to execution end", bounds,
      "host_wall_time");
}

ExecutionStatistics::ExecutionStatistics(std::shared_ptr<StatisticsSession> session, uint32_t gpu)
    : ExecutionPlugin("rocjitsu_statistics"), session_(std::move(session)), gpu_(gpu) {
  const auto scope = session_->scope("execution", gpu_);
  counts_ = bind_counts(scope);
  duration_ = bind_duration(scope);
  if (scope) {
    started_ = scope->counter("execution.dispatches_started", "dispatches", "Execution began");
    ended_ = scope->counter("execution.dispatches_ended", "dispatches",
                            "Execution ended; durable completion publication may still retry");
    incomplete_ = scope->counter("execution.dispatches_incomplete", "dispatches",
                                 "Observed nonempty dispatches lacking execution end");
    cancelled_waves_ = scope->counter("execution.waves_cancelled", "waves", "Cancelled waves");
    halted_waves_ = scope->counter("execution.waves_halted", "waves", "Halted waves");
  }
  for (size_t index = 0; index < counts_.size(); ++index)
    count_instructions_ |= session_->selected(std::string("execution.wave_instructions.") +
                                              std::string(plugins::instruction_family_name(
                                                  static_cast<plugins::InstructionFamily>(index))));
}

void ExecutionStatistics::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  // PM4 zero-grid packets do not enter execution and have no execution-end callback.
  if (info.workgroup_count == 0)
    return;
  DispatchState &state = dispatches_[info.dispatch_id];
  state.scope = session_->scope(std::format("dispatch/{}", info.dispatch_id), gpu_, "dispatch");
  state.counts = bind_counts(state.scope);
  state.duration = bind_duration(state.scope);
  if (state.scope) {
    state.scope->set_attribute("kernel_name", info.kernel_name);
    state.scope->set_attribute("kernel_symbol", info.kernel_symbol);
    state.scope->set_attribute("dispatch_id", std::to_string(info.dispatch_id));
    state.scope->set_attribute("wave_size", std::to_string(info.wave_size));
    state.scope->set_attribute("code_target", std::to_string(info.code_target));
    state.started =
        state.scope->counter("execution.dispatches_started", "dispatches", "Execution began");
    state.ended = state.scope->counter("execution.dispatches_ended", "dispatches",
                                       "Execution ended before durable completion publication");
    state.incomplete = state.scope->counter("execution.dispatches_incomplete", "dispatches",
                                            "Observed dispatches lacking execution end");
  }
}

void ExecutionStatistics::onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
  DispatchState &state = dispatches_[dispatch_id];
  if (state.begun)
    return;
  state.begun = true;
  if (duration_ || state.duration)
    state.begin = Clock::now();
  if (started_)
    started_->increment();
  if (state.started)
    state.started->increment();
}

void ExecutionStatistics::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  const auto found = dispatches_.find(dispatch_id);
  if (found == dispatches_.end())
    return;
  DispatchState &state = found->second;
  if (state.begun && (duration_ || state.duration)) {
    const double seconds = std::chrono::duration<double>(Clock::now() - state.begin).count();
    if (duration_)
      duration_->observe(seconds);
    if (state.duration)
      state.duration->observe(seconds);
  }
  if (ended_)
    ended_->increment();
  if (state.ended)
    state.ended->increment();
  dispatches_.erase(found);
}

void ExecutionStatistics::onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wave) {
  if (!count_instructions_)
    return;
  auto state = std::make_unique<WaveState>();
  state->owner = this;
  const auto found = dispatches_.find(wave.dispatch_id());
  if (found != dispatches_.end())
    state->counts = found->second.counts;
  wave.set_plugin_state(slot_index(), std::move(state));
}

void ExecutionStatistics::count(const Instruction &instruction, amdgpu::Wavefront &wave) {
  const size_t family = static_cast<size_t>(plugins::classify_instruction(instruction));
  if (counts_[family])
    counts_[family]->increment();
  // Low-level plugin-group replacement can leave another plugin's state in this slot.
  const auto *state = wave.has_plugin_state(slot_index())
                          ? dynamic_cast<const WaveState *>(wave.plugin_state(slot_index()))
                          : nullptr;
  if (state && state->owner == this && state->counts[family])
    state->counts[family]->increment();
}

void ExecutionStatistics::onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &instruction,
                                                           amdgpu::Wavefront &wave) {
  count(instruction, wave);
}

void ExecutionStatistics::onAmdgpuAsyncInstructionIssued(uint64_t, const Instruction &instruction,
                                                         amdgpu::Wavefront &wave) {
  count(instruction, wave);
}

void ExecutionStatistics::onAmdgpuWavefrontHalted(amdgpu::Wavefront &) {
  if (halted_waves_)
    halted_waves_->increment();
}

void ExecutionStatistics::onAmdgpuWavefrontCancelled(amdgpu::Wavefront &) {
  if (cancelled_waves_)
    cancelled_waves_->increment();
}

void ExecutionStatistics::onShutdown() {
  if (shutdown_)
    return;
  shutdown_ = true;
  if (!dispatches_.empty())
    session_->mark_incomplete();
  for (auto &[dispatch_id, state] : dispatches_) {
    if (incomplete_)
      incomplete_->increment();
    if (state.incomplete)
      state.incomplete->increment();
  }
  dispatches_.clear();
}

} // namespace rocjitsu
