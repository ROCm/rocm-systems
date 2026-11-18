// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/stats/statistics.h"
#include "rocjitsu/base/rj_version.h"

#include <algorithm>
#include <format>
#include <utility>

namespace rocjitsu {

StatisticsReset::StatisticsReset(std::shared_ptr<State> state, Kind kind)
    : state_(std::move(state)), kind_(kind) {
  std::lock_guard lock(state_->mutex);
  state_->kinds[static_cast<size_t>(kind_)].attempts.increment();
  ++state_->active;
}

StatisticsReset::~StatisticsReset() {
  if (!state_)
    return;
  std::lock_guard lock(state_->mutex);
  auto &counts = state_->kinds[static_cast<size_t>(kind_)];
  if (completed_)
    counts.completed.increment();
  else
    counts.failed.increment();
  --state_->active;
}

void StatisticsSession::register_gpu(uint32_t gpu_id) {
  std::lock_guard lock(mutex_);
  if (!finished_ && !gpu_resets_.contains(gpu_id))
    gpu_resets_.emplace(gpu_id, std::make_shared<StatisticsReset::State>());
}

StatisticsReset StatisticsSession::observe_reset(uint32_t gpu_id, StatisticsReset::Kind kind) {
  std::lock_guard lock(mutex_);
  if (finished_)
    return {};
  const auto state = gpu_resets_.find(gpu_id);
  if (state == gpu_resets_.end() || static_cast<size_t>(kind) >= 3) {
    incomplete_.store(true, std::memory_order_relaxed);
    return {};
  }
  return StatisticsReset(state->second, kind);
}

void StatisticsSession::write_gpu_metadata(std::ostream &output) const {
  output << ",\"accumulation_scope\":\"session\",\"devices\":[";
  bool first = true;
  constexpr std::array names{"function_level", "bus", "lost_connection"};
  for (const auto &[gpu_id, state] : gpu_resets_) {
    std::lock_guard lock(state->mutex);
    if (!first)
      output << ',';
    first = false;
    output << std::format("{{\"gpu_id\":{},\"resets\":{{", gpu_id);
    for (size_t index = 0; index < names.size(); ++index) {
      if (index != 0)
        output << ',';
      const auto &counts = state->kinds[index];
      output << std::format("\"{}\":{{\"attempts\":\"{}\",\"completed\":\"{}\",\"failed\":\"{}\","
                            "\"overflow\":{}}}",
                            names[index], counts.attempts.value(), counts.completed.value(),
                            counts.failed.value(),
                            counts.attempts.overflowed() || counts.completed.overflowed() ||
                                counts.failed.overflowed());
    }
    output << std::format("}},\"resets_in_progress\":\"{}\"}}", state->active);
  }
  output << ']';
}

StatisticsSession::StatisticsSession(const StatisticsConfig &config, std::string node,
                                     std::string execution_model,
                                     std::shared_ptr<stats::Session> collection)
    : config_(config), node_(std::move(node)), execution_model_(std::move(execution_model)),
      collection_(std::move(collection)) {}

util::FailureOr<std::shared_ptr<StatisticsSession>>
StatisticsSession::create(const StatisticsConfig &config, std::string node,
                          std::string execution_model, util::DiagnosticEmitter emit_error) {
  if (!config.enabled)
    return std::shared_ptr<StatisticsSession>{};
  if (config.max_dispatch_scopes == 0)
    return emit_error.emit() << "statistics.max_dispatch_scopes must be positive";
  for (const std::string &scope : config.scopes)
    if (scope != "component" && scope != "dispatch")
      return emit_error.emit() << "statistics.scopes accepts component or dispatch";
  stats::Config collection_config{.enabled = true,
                                  .include = config.include,
                                  .directory = config.directory,
                                  .format = config.format,
                                  .schema = "rocjitsu.statistics.v1",
                                  .default_domain = "model_events"};
  auto collection = stats::Session::create(collection_config, emit_error);
  if (collection.failed())
    return util::Result::failure();
  return std::shared_ptr<StatisticsSession>(new StatisticsSession(
      config, std::move(node), std::move(execution_model), std::move(collection).value()));
}

std::shared_ptr<stats::Scope> StatisticsSession::scope(std::string path, uint32_t gpu_id,
                                                       std::string_view kind) {
  std::lock_guard lock(mutex_);
  if (finished_ ||
      std::find(config_.scopes.begin(), config_.scopes.end(), kind) == config_.scopes.end())
    return {};
  const std::string key = std::format("{}:{}", gpu_id, path);
  if (auto existing = collection_->find_scope(key))
    return existing;
  if (kind == "dispatch") {
    if (dispatch_scopes_ == config_.max_dispatch_scopes) {
      dropped_dispatch_scopes_.increment();
      incomplete_.store(true, std::memory_order_relaxed);
      return {};
    }
    ++dispatch_scopes_;
  }
  return collection_->scope(key, std::move(path), [gpu_id](std::ostream &output) {
    output << std::format(",\"gpu_id\":{}", gpu_id);
  });
}

void StatisticsSession::write_run_metadata(std::ostream &output) const {
  output << std::format(",\"node\":{},\"execution_model\":{},\"build\":{}",
                        stats::json_string(node_), stats::json_string(execution_model_),
                        stats::json_string(rj_get_version_string()));
  write_gpu_metadata(output);
  output << ",\"selection\":{\"include\":[";
  for (size_t index = 0; index < config_.include.size(); ++index) {
    if (index)
      output << ',';
    output << stats::json_string(config_.include[index]);
  }
  output << "],\"scopes\":[";
  for (size_t index = 0; index < config_.scopes.size(); ++index) {
    if (index)
      output << ',';
    output << stats::json_string(config_.scopes[index]);
  }
  output << std::format("],\"max_dispatch_scopes\":{}}},\"dropped_dispatch_scopes\":\"{}\"",
                        config_.max_dispatch_scopes, dropped_dispatch_scopes_.value());
}

void StatisticsSession::finish() {
  std::lock_guard lock(mutex_);
  if (finished_)
    return;
  finished_ = true;
  for (const auto &[gpu_id, state] : gpu_resets_) {
    std::lock_guard reset_lock(state->mutex);
    if (state->active != 0)
      incomplete_.store(true, std::memory_order_relaxed);
  }
  const bool complete =
      ran_.load(std::memory_order_relaxed) && !incomplete_.load(std::memory_order_relaxed);
  collection_->finish(
      complete, [this](std::ostream &output) { write_run_metadata(output); },
      [this](std::ostream &output) {
        output << std::format(",\"dropped_dispatch_scopes\":\"{}\"",
                              dropped_dispatch_scopes_.value());
      });
}

} // namespace rocjitsu
