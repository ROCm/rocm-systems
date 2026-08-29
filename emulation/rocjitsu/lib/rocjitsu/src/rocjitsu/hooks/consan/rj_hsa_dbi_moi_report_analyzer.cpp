// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_analyzer.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <ranges>
#include <tuple>

namespace rocjitsu::consan_hook {

AutoMoiSampledConflictAnalysis
analyze_auto_moi_sampled_conflicts(std::span<const AutoMoiSampledEvidence> evidence,
                                   bool synchronization_evidence_complete) {
  AutoMoiSampledConflictAnalysis result;
  for (size_t index = 0; index < evidence.size(); ++index) {
    const AutoMoiSampledEvidence &current = evidence[index];
    for (size_t prior_index = 0; prior_index < index; ++prior_index) {
      const AutoMoiSampledEvidence &prior = evidence[prior_index];
      if (current.dispatch_id != prior.dispatch_id || current.workgroup_x != prior.workgroup_x ||
          current.workgroup_y != prior.workgroup_y || current.workgroup_z != prior.workgroup_z ||
          current.cluster_workgroup_id != prior.cluster_workgroup_id ||
          current.epoch != prior.epoch)
        continue;
      if (current.static_mapping != nullptr && prior.static_mapping != nullptr &&
          current.static_mapping->owner_provenance_complete &&
          prior.static_mapping->owner_provenance_complete &&
          std::ranges::none_of(
              current.static_mapping->owner_descriptor_file_offsets, [&](uint64_t owner) {
                return std::ranges::find(prior.static_mapping->owner_descriptor_file_offsets,
                                         owner) !=
                       prior.static_mapping->owner_descriptor_file_offsets.end();
              }))
        continue;
      if (!consan_moi_sampled_watchpoints_conflict(current.entry, prior.entry))
        continue;
      if (synchronization_evidence_complete &&
          consan_moi_sampled_atomic_pair_orders_same_workgroup(prior.sync, current.sync))
        continue;
      if (result.conflict_count != std::numeric_limits<uint32_t>::max())
        ++result.conflict_count;
      if (!result.first_conflict)
        result.first_conflict = std::make_pair(prior, current);
    }
  }
  return result;
}

uint64_t record_replay_bank_saturation_count(const ConSanMoiReportHeader &header,
                                             ConSanMoiEngine engine) {
  return engine == ConSanMoiEngine::RecordReplay &&
                 (header.flags & kConSanMoiReportFlagRecordReplayBankSaturated) != 0u
             ? 1u
             : 0u;
}

std::string_view record_replay_pressure_unavailable_reason_name(
    RecordReplayPressureTelemetry::UnavailableReason reason) {
  using Reason = RecordReplayPressureTelemetry::UnavailableReason;
  switch (reason) {
  case Reason::None:
    return "none";
  case Reason::NotRecordReplay:
    return "not_record_replay";
  case Reason::NoDispatchDirectory:
    return "no_dispatch_directory";
  case Reason::NoAccessTable:
    return "no_access_table";
  case Reason::NoLogicalAccessRanges:
    return "no_logical_access_ranges";
  }
  return "unknown";
}

RecordReplayPressureTelemetry
record_replay_pressure_telemetry(const ConSanMoiReportHeader &header, ConSanMoiEngine engine,
                                 std::span<const ConSanMoiAccessRecord> records,
                                 uint32_t logical_access_range_count,
                                 uint32_t address_group_headroom) {
  RecordReplayPressureTelemetry result;
  result.saturated = record_replay_bank_saturation_count(header, engine) != 0;
  if (engine != ConSanMoiEngine::RecordReplay) {
    result.unavailable_reason = RecordReplayPressureTelemetry::UnavailableReason::NotRecordReplay;
    return result;
  }
  if (header.record_replay_dispatch_token_capacity == 0) {
    result.unavailable_reason =
        RecordReplayPressureTelemetry::UnavailableReason::NoDispatchDirectory;
    return result;
  }
  if (header.access_record_capacity == 0) {
    result.unavailable_reason = RecordReplayPressureTelemetry::UnavailableReason::NoAccessTable;
    return result;
  }
  if (logical_access_range_count == 0) {
    result.unavailable_reason =
        RecordReplayPressureTelemetry::UnavailableReason::NoLogicalAccessRanges;
    return result;
  }

  result.available = true;
  result.access_record_capacity = header.access_record_capacity;
  result.address_group_headroom = address_group_headroom;
  result.logical_access_range_count = logical_access_range_count;
  std::vector<uint32_t> committed_record_indices(records.size());
  size_t committed_record_count = 0;
  for (size_t index = 0; index < records.size(); ++index) {
    const ConSanMoiAccessRecord &record = records[index];
    if (record.access_kind == static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty))
      continue;
    ++result.occupied_access_record_count;
    if (record.site_token >= logical_access_range_count) {
      ++result.invalid_site_token_count;
      continue;
    }
    committed_record_indices[committed_record_count++] = static_cast<uint32_t>(index);
  }

  const auto owner_key = [&](uint32_t index) {
    const ConSanMoiAccessRecord &record = records[index];
    return std::tuple{record.site_token,  record.generation,  record.workgroup_x,
                      record.workgroup_y, record.workgroup_z, record.wave_id};
  };
  const auto address_group_key = [&](uint32_t index) {
    const ConSanMoiAccessRecord &record = records[index];
    return std::tuple{record.site_token,     record.generation,  record.workgroup_x,
                      record.workgroup_y,    record.workgroup_z, record.wave_id,
                      record.lds_byte_offset};
  };
  std::sort(committed_record_indices.begin(),
            committed_record_indices.begin() + committed_record_count,
            [&](uint32_t left, uint32_t right) {
              return address_group_key(left) < address_group_key(right);
            });
  std::optional<decltype(owner_key(0u))> prior_owner;
  std::optional<uint32_t> prior_site;
  std::optional<uint32_t> prior_address;
  uint64_t owner_address_group_count = 0;
  for (size_t index = 0; index < committed_record_count; ++index) {
    const uint32_t record_index = committed_record_indices[index];
    const ConSanMoiAccessRecord &record = records[record_index];
    const auto current_owner = owner_key(record_index);
    if (!prior_site || *prior_site != record.site_token) {
      ++result.observed_site_count;
      prior_site = record.site_token;
    }
    if (!prior_owner || *prior_owner != current_owner) {
      owner_address_group_count = 1;
    } else if (!prior_address || *prior_address != record.lds_byte_offset) {
      ++owner_address_group_count;
    }
    prior_owner = current_owner;
    prior_address = record.lds_byte_offset;
    if (owner_address_group_count > result.maximum_site_owner_address_group_count) {
      result.maximum_site_owner_address_group_count = owner_address_group_count;
      result.maximum_site_token = record.site_token;
    }
  }
  return result;
}

AutoMoiRecordReplayAnalysis
analyze_auto_moi_record_replay(const ConSanMoiReportHeader &header, ConSanMoiEngine engine,
                               std::span<const ConSanMoiAccessRecord> access_records,
                               std::span<const ConSanMoiBarrierRecord> barrier_records,
                               std::span<const ConSanMoiAtomicRecord> atomic_records,
                               std::span<const ConSanMoiFenceRecord> fence_records,
                               std::span<const AutoMoiRecordReplayStaticMapping> static_mappings,
                               bool static_mapping_malformed, uint32_t logical_access_range_count,
                               uint32_t address_group_headroom) {
  AutoMoiRecordReplayAnalysis result;
  result.pressure = record_replay_pressure_telemetry(
      header, engine, access_records, logical_access_range_count, address_group_headroom);
  if (access_records.empty() && barrier_records.empty() && atomic_records.empty() &&
      fence_records.empty())
    return result;

  result.replay_performed = true;
  for (const ConSanMoiAccessRecord &record : access_records) {
    uint64_t record_end = static_cast<uint64_t>(record.start_cell) + record.cell_count;
    if (record_end == 0 && record.lds_byte_count != 0) {
      const ConSanMoiLdsCellRange range =
          consan_moi_lds_cell_range_for_bytes(record.lds_byte_offset, record.lds_byte_count);
      record_end = static_cast<uint64_t>(range.start_cell) + range.cell_count;
    }
    result.required_shadow_entry_count = std::max(result.required_shadow_entry_count, record_end);
  }
  constexpr uint64_t kMaxAutoReplayShadowEntries = 1u << 20u;
  result.replay_shadow_entry_count = std::max<uint64_t>(
      std::min(result.required_shadow_entry_count, kMaxAutoReplayShadowEntries), 1u);
  result.shadow_bounded = result.required_shadow_entry_count > kMaxAutoReplayShadowEntries;

  result.replay_header = header;
  result.replay_header.access_record_count = static_cast<uint32_t>(access_records.size());
  result.replay_header.access_record_capacity = result.replay_header.access_record_count;
  result.replay_header.diagnostic_count = 0;
  result.replay_header.diagnostic_capacity =
      std::min(header.diagnostic_capacity, result.replay_header.access_record_count);
  result.diagnostics.resize(result.replay_header.diagnostic_capacity);
  std::vector<uint64_t> exact_shadow_entries(static_cast<size_t>(result.replay_shadow_entry_count));
  result.replay = consan_moi_record_replay_access_records(
      result.replay_header, access_records, barrier_records,
      std::span<const ConSanMoiRecordReplayAtomicEvent>(atomic_records.data(),
                                                        atomic_records.size()),
      std::span<const ConSanMoiRecordReplayFenceEvent>(fence_records.data(), fence_records.size()),
      result.diagnostics, exact_shadow_entries);

  const uint32_t raw_visible_diagnostics =
      std::min<uint32_t>(result.replay.emitted_diagnostic_count, result.diagnostics.size());
  result.provenance = repair_consan_moi_record_replay_provenance(
      access_records,
      std::span<ConSanMoiDiagnosticRecord>(result.diagnostics.data(), raw_visible_diagnostics));
  const auto static_mapping_for_instruction = [&](uint32_t instruction_offset) {
    const auto mapping = std::ranges::find_if(
        static_mappings, [instruction_offset](const AutoMoiRecordReplayStaticMapping &candidate) {
          return candidate.instruction_offset == instruction_offset;
        });
    return mapping == static_mappings.end() ? nullptr : &*mapping;
  };
  const auto has_disjoint_kernel_owners = [&](const ConSanMoiDiagnosticRecord &diagnostic) {
    if (diagnostic.kind != static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict))
      return false;
    const AutoMoiRecordReplayStaticMapping *first =
        static_mapping_for_instruction(diagnostic.first_instruction_offset);
    const AutoMoiRecordReplayStaticMapping *second =
        static_mapping_for_instruction(diagnostic.second_instruction_offset);
    if (first == nullptr || second == nullptr || !first->owner_provenance_complete ||
        !second->owner_provenance_complete || static_mapping_malformed)
      return false;
    return std::ranges::none_of(first->owner_descriptor_file_offsets, [&](uint64_t owner) {
      return std::ranges::find(second->owner_descriptor_file_offsets, owner) !=
             second->owner_descriptor_file_offsets.end();
    });
  };
  const auto visible_end = std::remove_if(result.diagnostics.begin(),
                                          result.diagnostics.begin() + raw_visible_diagnostics,
                                          has_disjoint_kernel_owners);
  const uint32_t visible_diagnostics =
      static_cast<uint32_t>(visible_end - result.diagnostics.begin());
  result.disjoint_owner_suppressed_count = raw_visible_diagnostics - visible_diagnostics;
  result.effective_diagnostic_count =
      result.replay.emitted_diagnostic_count >= result.disjoint_owner_suppressed_count
          ? result.replay.emitted_diagnostic_count - result.disjoint_owner_suppressed_count
          : 0u;
  result.diagnostics.resize(visible_diagnostics);
  result.effective_conflict =
      result.effective_diagnostic_count != 0u || result.replay.metadata_full ||
      result.replay.diagnostic_capacity_exhausted || result.replay.dropped_access_count != 0u ||
      result.replay.dropped_barrier_count != 0u || result.replay.unsupported_access_count != 0u ||
      result.replay.unsupported_atomic_count != 0u || result.replay.unsupported_fence_count != 0u;
  return result;
}

} // namespace rocjitsu::consan_hook
