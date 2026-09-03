// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file report.h
/// @brief Collection, deduplication and serialization of hazard findings.
///
/// The engine reports findings through ::hazard_core::HazardWarningSink. This
/// file implements that sink on top of a collector that folds logically
/// identical hazards together, and emits the result both as human-readable
/// lines through the plugin sink and as a JSON document.

#pragma once

#include "data_hazard_state.h"
#include "hazard_events.h"
#include "simulator_api.h"

#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu::plugins::data_hazard {

/// @brief One reported hazard, retained until the report is written.
struct HazardWarning {
  hazard_core::EntityId instruction_id = 0;
  hazard_core::EntityId wave_id = 0;
  uint64_t pc = 0;
  std::string message;
  std::string suggestion;
  hazard_core::EntityId dispatch_id = 0;
  hazard_core::EntityId cluster_id = 0;
  hazard_core::EntityId workgroup_id = 0;
  std::string instruction_text;
  std::array<uint32_t, 4> raw_isa{};
  uint64_t suppressed_occurrences = 0;
  std::string source_instruction_text;
  std::array<uint32_t, 4> source_raw_isa{};
  uint64_t source_pc = 0;
  hazard_core::EntityId source_instruction_id = 0;
  /// Where the producing instruction ran. A cross-wave or cross-workgroup race
  /// is only actionable once the other side of it is identified, so these are
  /// carried in their own right rather than left in the message text. They stay
  /// zero when the producer is known by raw words alone.
  hazard_core::EntityId source_dispatch_id = 0;
  hazard_core::EntityId source_cluster_id = 0;
  hazard_core::EntityId source_workgroup_id = 0;
  hazard_core::EntityId source_wave_id = 0;
  bool has_source_instruction = false;
};

/// @brief Formats four ISA words as space-separated zero-padded hex.
std::string format_raw_isa_hex(const std::array<uint32_t, 4> &raw_isa);

/// @brief Strips runtime-varying values from a message so that logically
/// identical hazards from different waves collapse to one entry.
///
/// Per-lane addresses, racing wave IDs, and the trailing runtime instruction
/// ordinal are all execution artifacts of the same shader-level hazard.
std::string dedup_message_key(const std::string &message);

/// @brief Renders one hazard as a single human-readable sink line.
std::string format_warning_line(const HazardWarning &warning);

/// @brief Accumulates hazards, optionally folding duplicates.
class WarningCollector {
public:
  /// @param verbose Retain every occurrence instead of deduplicating.
  explicit WarningCollector(bool verbose = false) : verbose_(verbose) {}

  void add(const HazardWarning &warning);
  void clear();

  /// @brief Installs a callback invoked once per distinct hazard, as it is
  /// collected.
  ///
  /// The JSON report and the summary are only written once the run finishes,
  /// which for a launched application is process exit. This hook lets the
  /// plugin stream findings out while the run is still in progress. Duplicates
  /// folded into an existing entry do not fire it.
  void set_on_new_warning(std::function<void(const HazardWarning &)> callback);

  /// @returns A copy of the collected hazards in first-seen order.
  std::vector<HazardWarning> snapshot() const;

  /// @brief Serializes the collected hazards as a JSON array.
  std::string to_json() const;

  /// @brief Renders a short human-readable summary for the plugin sink.
  std::string to_summary() const;

  /// @brief Writes to_json() to @p path.
  /// @returns false when the file could not be opened or written.
  bool write_json_file(const std::string &path) const;

private:
  /// The static instruction pair a hazard belongs to. Both PCs are named: a
  /// race message keeps neither once dedup_message_key() has taken the waves
  /// and addresses out of it, so without the producer's PC two instructions
  /// racing with one consumer would fold into whichever was found first.
  struct DedupKey {
    hazard_core::EntityId dispatch_id;
    uint64_t pc;
    uint64_t source_pc;
    std::string message;
    std::string suggestion;

    bool operator==(const DedupKey &other) const noexcept {
      return dispatch_id == other.dispatch_id && pc == other.pc && source_pc == other.source_pc &&
             message == other.message && suggestion == other.suggestion;
    }
  };

  struct DedupKeyHash {
    size_t operator()(const DedupKey &key) const noexcept;
  };

  mutable std::mutex mutex_;
  bool verbose_ = false;
  std::vector<HazardWarning> warnings_;
  std::unordered_map<DedupKey, size_t, DedupKeyHash> dedup_index_;
  std::function<void(const HazardWarning &)> on_new_warning_;
};

/// @brief Adapts engine findings into HazardWarning entries.
///
/// The formatter supplies the disassembly-like text for the offending and
/// producing instructions; it may be null, in which case only raw ISA words
/// are recorded.
class CollectingWarningSink final : public hazard_core::HazardWarningSink {
public:
  CollectingWarningSink(WarningCollector &collector,
                        const hazard_core::SimulatorInstructionFormatter *formatter)
      : collector_(collector), formatter_(formatter) {}

  void emit_warning(const hazard_core::EngineWarning &warning) const override;

private:
  WarningCollector &collector_;
  const hazard_core::SimulatorInstructionFormatter *formatter_ = nullptr;
};

} // namespace rocjitsu::plugins::data_hazard
