// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_HOOKS_RJ_HSA_DBI_REPLAY_PROVENANCE_H_
#define ROCJITSU_HOOKS_RJ_HSA_DBI_REPLAY_PROVENANCE_H_

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

struct ConSanMoiReplayProvenanceRepair {
  uint32_t repaired_diagnostic_count = 0;
  uint32_t unresolved_diagnostic_count = 0;
};

/// Restores host-only provenance which cannot fit in the packed exact shadow.
///
/// Replay itself remains authoritative for conflict and ordering decisions.
/// This companion follows the same per-workgroup/per-cell replacement order,
/// consumes the already emitted diagnostics in event order, and copies lane
/// and byte-range evidence only from the exact cell whose packed identity
/// matches the reported prior access. Missing or inconsistent provenance is
/// left unknown rather than inferred from another cell or access.
[[nodiscard]] inline ConSanMoiReplayProvenanceRepair repair_consan_moi_record_replay_provenance(
    std::span<const ConSanMoiAccessRecord> access_records,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records) {
  constexpr uint64_t kMaxCompanionCells = 1u << 20u;
  ConSanMoiReplayProvenanceRepair result;

  struct WorkgroupState {
    std::unordered_map<uint32_t, const ConSanMoiAccessRecord *> cells;
    bool complete = true;
  };
  using WorkgroupKey = std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>;
  std::map<WorkgroupKey, WorkgroupState> workgroups;
  uint64_t allocated_cells = 0;
  auto workgroup_for = [&](const ConSanMoiAccessRecord &record) -> WorkgroupState & {
    return workgroups[{record.generation, record.workgroup_x, record.workgroup_y,
                       record.workgroup_z}];
  };

  const auto range_for =
      [](const ConSanMoiAccessRecord &record) -> std::optional<ConSanMoiLdsCellRange> {
    uint64_t start = record.start_cell;
    uint64_t count = record.cell_count;
    if (count == 0 && record.lds_byte_count != 0) {
      const uint64_t byte_end = static_cast<uint64_t>(record.lds_byte_offset) +
                                static_cast<uint64_t>(record.lds_byte_count);
      start = record.lds_byte_offset >> consan_moi_exact_shadow::granule_shift;
      const uint64_t end = (byte_end + consan_moi_exact_shadow::granule_bytes - 1u) >>
                           consan_moi_exact_shadow::granule_shift;
      count = end - start;
    }
    const uint64_t end = start + count;
    if (count == 0 || end < start || end > kMaxCompanionCells)
      return std::nullopt;
    return ConSanMoiLdsCellRange{static_cast<uint32_t>(start), static_cast<uint32_t>(count)};
  };
  const auto valid_kind = [](uint32_t value) {
    const auto kind = static_cast<ConSanMoiShadowAccessKind>(value);
    return kind == ConSanMoiShadowAccessKind::Read || kind == ConSanMoiShadowAccessKind::Write ||
           kind == ConSanMoiShadowAccessKind::ReadWrite ||
           kind == ConSanMoiShadowAccessKind::Atomic;
  };
  const auto diagnostic_matches_current = [](const ConSanMoiDiagnosticRecord &diagnostic,
                                             const ConSanMoiAccessRecord &record) {
    return diagnostic.backend == static_cast<uint32_t>(ConSanMoiEngine::RecordReplay) &&
           diagnostic.reserved == record.event_index &&
           diagnostic.second_owner_id == record.wave_id &&
           diagnostic.second_instruction_offset == record.instruction_offset &&
           diagnostic.second_access_kind == record.access_kind;
  };
  const auto diagnostic_matches_prior = [](const ConSanMoiDiagnosticRecord &diagnostic,
                                           const ConSanMoiAccessRecord &record) {
    return diagnostic.first_owner_id == record.wave_id &&
           diagnostic.first_instruction_offset ==
               (record.instruction_offset & consan_moi_exact_shadow::max_instruction_offset) &&
           diagnostic.first_access_kind == record.access_kind;
  };

  std::vector<size_t> order(access_records.size());
  for (size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    return access_records[lhs].event_index < access_records[rhs].event_index;
  });

  size_t diagnostic_index = 0;
  for (size_t record_index : order) {
    const ConSanMoiAccessRecord &record = access_records[record_index];
    if (!valid_kind(record.access_kind) || record.wave_id > consan_moi_exact_shadow::max_owner)
      continue;
    ConSanMoiDiagnosticRecord *diagnostic = nullptr;
    if (diagnostic_index < diagnostic_records.size() &&
        diagnostic_matches_current(diagnostic_records[diagnostic_index], record)) {
      diagnostic = &diagnostic_records[diagnostic_index++];
    }
    const std::optional<ConSanMoiLdsCellRange> range = range_for(record);
    if (!range) {
      if (diagnostic != nullptr &&
          diagnostic->kind == static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict))
        ++result.unresolved_diagnostic_count;
      continue;
    }
    WorkgroupState &state = workgroup_for(record);
    if (!state.complete) {
      if (diagnostic != nullptr &&
          diagnostic->kind == static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict))
        ++result.unresolved_diagnostic_count;
      continue;
    }
    if (diagnostic != nullptr) {
      if (diagnostic->kind != static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict))
        continue;
      const ConSanMoiAccessRecord *prior = nullptr;
      const uint32_t end = range->start_cell + range->cell_count;
      for (uint32_t cell = range->start_cell; cell < end; ++cell) {
        const auto candidate = state.cells.find(cell);
        if (candidate != state.cells.end() &&
            diagnostic_matches_prior(*diagnostic, *candidate->second)) {
          prior = candidate->second;
          break;
        }
      }
      if (prior == nullptr) {
        ++result.unresolved_diagnostic_count;
        continue;
      }
      diagnostic->first_lane_mask = prior->lane_mask;
      diagnostic->first_lds_byte_offset = prior->lds_byte_offset;
      diagnostic->first_lds_byte_count = prior->lds_byte_count;
      diagnostic->second_lane_mask = record.lane_mask;
      diagnostic->second_lds_byte_offset = record.lds_byte_offset;
      diagnostic->second_lds_byte_count = record.lds_byte_count;
      ++result.repaired_diagnostic_count;
      continue;
    }

    const uint32_t end = range->start_cell + range->cell_count;
    uint64_t growth = 0;
    for (uint32_t cell = range->start_cell; cell < end; ++cell) {
      if (!state.cells.contains(cell))
        ++growth;
    }
    if (growth > kMaxCompanionCells - allocated_cells) {
      // The cap is global, but a miss poisons only this workgroup: a partial
      // shadow must never be mistaken for exact provenance after exhaustion.
      state.complete = false;
      continue;
    }
    for (uint32_t cell = range->start_cell; cell < end; ++cell)
      state.cells[cell] = &record;
    allocated_cells += growth;
  }

  for (; diagnostic_index < diagnostic_records.size(); ++diagnostic_index) {
    if (diagnostic_records[diagnostic_index].kind ==
        static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict))
      ++result.unresolved_diagnostic_count;
  }
  return result;
}

} // namespace rocjitsu

#endif // ROCJITSU_HOOKS_RJ_HSA_DBI_REPLAY_PROVENANCE_H_
