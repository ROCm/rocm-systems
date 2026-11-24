// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/render_helpers.h"
#include "rocjitsu/code/consan/capture.h"
#include "rocjitsu/code/consan/encoding.h"
#include "util/diagnostic.h"

#include <array>
#include <fstream>
#include <iostream>
#include <string_view>

int main(int argc, char **argv) {
  using namespace rocjitsu::consan;
  using rocjitsu::FailureOr;
  const bool summary = argc == 3 && std::string_view(argv[1]) == "--summary";
  if (argc != 2 && !summary) {
    return 2;
  }
  std::ifstream file(argv[summary ? 2 : 1], std::ios::binary | std::ios::ate);
  if (!file || file.tellg() < 0 || uint64_t(file.tellg()) > CaptureLimits{}.file_bytes) {
    return 2;
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (!file) {
    return 2;
  }
  util::StringDiagnostic error;
  const FailureOr<Capture> captured = decode_capture(bytes, error.emitter());
  if (captured.failed()) {
    std::cerr << "capture input error: " << error.message() << '\n';
    return 2;
  }
  const Capture &capture = captured.value();
  if (summary) {
    std::cout << "{\"original_sha256\":\""
              << render_detail::hex_bytes(capture.header.image.original.sha256)
              << "\",\"entries\":[";
    bool first = true;
    for (const ImageEntryFacts &entry : capture.header.image.entries) {
      if (!first)
        std::cout << ',';
      first = false;
      std::cout << "{\"entry_key\":" << entry.key << ",\"entry\":";
      render_detail::write_json_string(std::cout, entry.binding.name);
      std::cout << ",\"selected\":" << (entry.selected ? "true" : "false");
      if (entry.selected) {
        const Manifest &manifest = *entry.selected;
        std::cout << ",\"lds_access_sites\":"
                  << (manifest.scope == ScopeKind::ExecutionTrace ? 0 : manifest.sites.size())
                  << ",\"barrier_completion_sites\":" << manifest.barriers.size()
                  << ",\"trace_boundary_sites\":" << manifest.boundaries.size();
      } else {
        std::cout << ",\"lds_access_sites\":null,\"barrier_completion_sites\":null,"
                     "\"trace_boundary_sites\":null";
      }
      std::cout << '}';
    }
    std::cout << "]}\n";
    return 0;
  }
  const auto write_sites = [](const Manifest &manifest) {
    bool first_site = true;
    for (const SiteFacts &site : manifest.sites) {
      if (!first_site) {
        std::cout << ',';
      }
      first_site = false;
      std::cout << "{\"id\":" << site.id << ",\"access\":" << uint32_t(site.access)
                << ",\"width\":" << site.width << ",\"original_offset\":" << site.original_offset
                << ",\"guest_offset\":" << site.prepared_guest_offset;
      if (is_dual(site.lds_ranges.shape)) {
        std::cout << ",\"range_offsets\":[" << site.lds_ranges.byte_offset(0, site.width) << ','
                  << site.lds_ranges.byte_offset(1, site.width) << ']';
      }
      if (is_transpose(site.lds_ranges.shape)) {
        constexpr std::array<uint32_t, 4> bits{4, 6, 8, 16};
        std::cout << ",\"transpose_bits\":"
                  << bits[uint32_t(site.lds_ranges.shape) - uint32_t(LdsRangeShape::Transpose4)];
      }
      std::cout << '}';
    }
  };
  std::cout << "{\"execution_conditions\":" << capture.header.execution_conditions
            << ",\"sites\":[";
  if (capture.header.image.entries.size() == 1 && capture.header.image.entries.front().selected) {
    write_sites(*capture.header.image.entries.front().selected);
  }
  std::cout << "],\"entries\":[";
  bool first = true;
  for (const ImageEntryFacts &entry : capture.header.image.entries) {
    if (!first) {
      std::cout << ',';
    }
    first = false;
    std::cout << "{\"entry_key\":" << entry.key
              << ",\"selected\":" << (entry.selected ? "true" : "false") << ",\"sites\":[";
    if (entry.selected) {
      write_sites(*entry.selected);
    }
    std::cout << "]}";
  }
  std::cout << "],\"executions\":[";
  first = true;
  for (const CapturedExecution &execution : capture.executions) {
    if (!first) {
      std::cout << ',';
    }
    first = false;
    const Manifest *manifest = selected_manifest(capture.header.image, execution.entry_key);
    if (manifest == nullptr) {
      return 2;
    }
    const FailureOr<LaunchLayout> layout = plan_launch(
        *manifest, execution.dimensions, execution.group_bytes, execution.private_bytes);
    if (layout.failed()) {
      return 2;
    }
    std::cout << "{\"serial\":" << execution.serial << ",\"slot\":" << execution.storage_slot
              << ",\"completion\":" << execution.completion
              << ",\"entry_key\":" << execution.entry_key
              << ",\"generation\":" << execution.storage_generation << ",\"report_address\":"
              << *wire::read_u64(execution.context, wire::kContextReportPointer)
              << ",\"grid_x\":" << execution.dimensions.grid[0]
              << ",\"workgroup_size_x\":" << execution.dimensions.workgroup[0]
              << ",\"argument_u32\":";
    if (const std::optional<uint32_t> argument = wire::read_u32(execution.arguments, 0)) {
      std::cout << *argument;
    } else {
      std::cout << "null";
    }
    std::cout << ",\"rows\":[";
    for (uint32_t wave = 0; wave < layout.value().wave_count; ++wave) {
      if (wave != 0) {
        std::cout << ',';
      }
      const size_t row = kReportHeaderBytes + wave * layout.value().row_stride;
      if (manifest->scope == ScopeKind::ExecutionTrace) {
        std::cout << "{\"wave\":" << *wire::read_u32(execution.report, row + wire::kRowWave)
                  << ",\"workgroup_x\":"
                  << *wire::read_u32(execution.report, row + wire::kTraceRowWorkgroup)
                  << ",\"local_wave\":"
                  << *wire::read_u32(execution.report, row + wire::kTraceRowLocalWave)
                  << ",\"serial\":" << *wire::read_u64(execution.report, row + wire::kRowSerial)
                  << ",\"status\":" << *wire::read_u32(execution.report, row + wire::kRowStatus)
                  << ",\"entries\":"
                  << *wire::read_u32(execution.report, row + wire::kTraceRowEntries)
                  << ",\"exits\":" << *wire::read_u32(execution.report, row + wire::kTraceRowExits)
                  << '}';
        continue;
      }
      std::cout << "{\"wave\":" << *wire::read_u32(execution.report, row + wire::kRowWave)
                << ",\"workgroup_x\":"
                << *wire::read_u32(execution.report, row + wire::kRowWorkgroupX)
                << ",\"serial\":" << *wire::read_u64(execution.report, row + wire::kRowSerial)
                << ",\"status\":" << *wire::read_u32(execution.report, row + wire::kRowStatus)
                << ",\"loss\":" << *wire::read_u32(execution.report, row + wire::kRowLoss)
                << ",\"final_sequence\":"
                << *wire::read_u32(execution.report, row + wire::kRowFinalSequence)
                << ",\"sync_loss\":" << *wire::read_u32(execution.report, row + wire::kRowSyncLoss)
                << ",\"counter_exhausted\":"
                << *wire::read_u32(execution.report, row + wire::kRowCounterExhausted)
                << ",\"final_phase\":"
                << *wire::read_u32(execution.report, row + wire::kRowFinalPhase)
                << ",\"completions\":[";
      bool first_completion = true;
      if (!manifest->barriers.empty()) {
        for (uint32_t slot = 0; slot < kSyncCapacity; ++slot) {
          const size_t offset = row + sync_offset(manifest->sites.size()) + slot * kSyncRecordBytes;
          if (wire::read_u32(execution.report, offset + wire::kSyncCommit) == 0) {
            continue;
          }
          if (!first_completion) {
            std::cout << ',';
          }
          first_completion = false;
          std::cout << "{\"commit\":"
                    << *wire::read_u32(execution.report, offset + wire::kSyncCommit)
                    << ",\"group\":" << *wire::read_u32(execution.report, offset + wire::kSyncGroup)
                    << ",\"occurrence\":"
                    << *wire::read_u32(execution.report, offset + wire::kSyncOccurrence)
                    << ",\"participants\":"
                    << *wire::read_u32(execution.report, offset + wire::kSyncParticipants)
                    << ",\"sequence\":"
                    << *wire::read_u32(execution.report, offset + wire::kSyncSequence)
                    << ",\"phase\":" << *wire::read_u32(execution.report, offset + wire::kSyncPhase)
                    << '}';
        }
      }
      std::cout << ']' << ",\"records\":[";
      bool first_record = true;
      for (const SiteFacts &site : manifest->sites) {
        for (uint32_t address_group = 0; address_group < kAddressGroupsPerSite; ++address_group) {
          const size_t record =
              row + kRowHeaderBytes + (site.first_slot + address_group) * kAccessRecordBytes;
          const uint32_t commit = *wire::read_u32(execution.report, record + wire::kRecordCommit);
          if (commit == 0) {
            continue;
          }
          if (!first_record) {
            std::cout << ',';
          }
          first_record = false;
          std::cout
              << "{\"commit\":" << commit
              << ",\"site\":" << *wire::read_u32(execution.report, record + wire::kRecordSite)
              << ",\"sequence\":"
              << *wire::read_u32(execution.report, record + wire::kRecordSequence)
              << ",\"phase\":" << *wire::read_u32(execution.report, record + wire::kRecordPhase)
              << ",\"offset\":"
              << *wire::read_u32(execution.report, record + wire::kRecordLdsOffset)
              << ",\"width\":" << *wire::read_u32(execution.report, record + wire::kRecordWidth)
              << ",\"mask\":" << *wire::read_u64(execution.report, record + wire::kRecordGuestMask)
              << ",\"flags\":" << *wire::read_u32(execution.report, record + wire::kRecordFlags);
          if (is_transpose(site.lds_ranges.shape)) {
            std::cout << ",\"anchor_exec\":"
                      << *wire::read_u64(execution.report, record + wire::kRecordAnchorExec);
          }
          std::cout << '}';
        }
      }
      std::cout << "]}";
    }
    std::cout << "]}";
  }
  std::cout << "]}\n";
  return 0;
}
