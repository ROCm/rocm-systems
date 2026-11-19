// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/diagnostic.h"
#include "util/statistics.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

struct StatisticsConfig {
  bool enabled = false;
  std::vector<std::string> include{};
  std::vector<std::string> scopes = {"component", "dispatch"};
  std::string directory = "stats";
  std::string format = "jsonl";
  uint32_t max_dispatch_scopes = 1024;
};

/// @brief VM policy and provenance around the reusable util statistics collection.
/// @details finish() requires all core, frontend and plugin producers stopped.
class StatisticsSession {
public:
  static util::FailureOr<std::shared_ptr<StatisticsSession>>
  create(const StatisticsConfig &config, std::string node, std::string execution_model,
         util::DiagnosticEmitter emit_error = {});
  StatisticsSession(const StatisticsSession &) = delete;
  StatisticsSession &operator=(const StatisticsSession &) = delete;
  std::shared_ptr<stats::Scope> scope(std::string path, uint32_t gpu_id,
                                      std::string_view kind = "component");
  void mark_incomplete() noexcept { incomplete_.store(true, std::memory_order_relaxed); }
  void mark_run_complete() noexcept { ran_.store(true, std::memory_order_relaxed); }
  void finish();
  const std::string &output_path() const { return collection_->output_path(); }
  bool selected(std::string_view metric_name) const { return collection_->selected(metric_name); }

private:
  StatisticsSession(const StatisticsConfig &config, std::string node, std::string execution_model,
                    std::shared_ptr<stats::Session> collection);
  void write_run_metadata(std::ostream &output) const;
  StatisticsConfig config_;
  std::string node_;
  std::string execution_model_;
  std::shared_ptr<stats::Session> collection_;
  std::mutex mutex_;
  uint32_t dispatch_scopes_ = 0;
  stats::Counter dropped_dispatch_scopes_;
  std::atomic<bool> incomplete_{false};
  std::atomic<bool> ran_{false};
  bool finished_ = false;
};

} // namespace rocjitsu
