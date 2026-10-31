// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hazard_record.h
/// @brief Shared hazard report used by the core memory-wait checker and the race plugin.
///
/// @details The race plugin bundles these objects but publishes through the simulator's
/// writer, so one lock covers the output of every checker.

#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu::diagnostics {

/// @brief The config's `diagnostics` block.
struct Settings {
  std::string report_path;
  uint32_t max_sites_per_dispatch = 64;
};

struct HazardSite {
  uint64_t pc = 0;
  std::string disasm;
  bool marked = false;
  std::string note;
};

struct HazardRecord {
  std::string checker;
  std::string type_name;
  std::string kernel;
  std::string symbol;
  /// Ordinal of the simulated GPU. Dispatch ids are unique only within one GPU.
  uint32_t gpu = 0;
  uint32_t dispatch = 0;
  uint32_t process = 0;
  std::array<uint32_t, 3> wg{};
  uint32_t wave = 0;
  int32_t lane = -1;
  std::string cu;
  uint32_t reg = 0;
  bool write = false;
  HazardSite producer;
  HazardSite consumer;
  std::optional<std::string> counter;
  std::optional<uint32_t> required_wait;
  std::vector<HazardSite> trace;
  std::string trace_block;
  bool trace_executed = false;
  std::string trace_note;
  std::string summary;
};

/// @brief Human-readable block. `tag` is `HAZARD` or `RACE`.
/// @details Race records keep `conflict=unknown` and add `producer_pc` after it so
/// a parser that keys on the existing prefix still matches.
std::string format_text(const HazardRecord &record, std::string_view tag);

/// @brief One `site` line of the report file.
std::string format_jsonl(const HazardRecord &record);

/// @brief Serializes the hazard output of every checker in the process.
/// @details One lock covers the text a checker prints and the JSON line appended to
/// the report file, so blocks from the core checker and the race plugin never
/// interleave, and each line reaches the file in a single write.
class ReportWriter {
public:
  ReportWriter() = default;
  ~ReportWriter();
  ReportWriter(const ReportWriter &) = delete;
  ReportWriter &operator=(const ReportWriter &) = delete;

  /// @brief Run @p print_text, then append @p json_line to @p path, under one lock.
  template <typename PrintText>
  void publish(PrintText &&print_text, const std::string &path, std::string_view json_line) {
    std::lock_guard lock(mutex_);
    print_text();
    if (!path.empty() && !json_line.empty())
      append(path, json_line);
  }

private:
  void append(const std::string &path, std::string_view json_line);

  std::mutex mutex_;
  /// Descriptor of each report path, or -1 once opening or writing it failed.
  std::unordered_map<std::string, int> files_;
};

struct DispatchIdentity {
  uint32_t dispatch_id = 0;
  std::string kernel;
  std::string symbol;
};

} // namespace rocjitsu::diagnostics
