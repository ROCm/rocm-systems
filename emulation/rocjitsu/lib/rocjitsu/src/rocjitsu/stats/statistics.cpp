// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/stats/statistics.h"
#include "rocjitsu/base/rj_version.h"

#include <algorithm>
#include <format>
#include <utility>

namespace rocjitsu {

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
}

void StatisticsSession::finish() {
  std::lock_guard lock(mutex_);
  if (finished_)
    return;
  finished_ = true;
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
