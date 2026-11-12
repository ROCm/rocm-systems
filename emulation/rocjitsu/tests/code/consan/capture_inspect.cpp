// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/capture.h"
#include "rocjitsu/code/consan/encoding.h"

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  using namespace rocjitsu::consan;
  using rocjitsu::FailureOr;
  if (argc != 2)
    return 2;
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  if (!file || file.tellg() < 0 || uint64_t(file.tellg()) > CaptureLimits{}.file_bytes)
    return 2;
  std::vector<uint8_t> bytes(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (!file)
    return 2;
  ContractError error;
  const FailureOr<Capture> captured = decode_capture(bytes, &error);
  if (captured.failed()) {
    std::cerr << "capture input error: " << error.message << '\n';
    return 2;
  }
  const Capture &capture = captured.value();
  std::cout << "{\"execution_conditions\":" << capture.header.execution_conditions
            << ",\"sites\":[";
  bool first = true;
  for (const SiteFacts &site : capture.header.manifest.sites) {
    if (!first)
      std::cout << ',';
    first = false;
    std::cout << "{\"id\":" << site.id << ",\"access\":" << uint32_t(site.access)
              << ",\"original_offset\":" << site.original_offset
              << ",\"guest_offset\":" << site.prepared_guest_offset << '}';
  }
  std::cout << "],\"executions\":[";
  first = true;
  for (const CapturedExecution &execution : capture.executions) {
    if (!first)
      std::cout << ',';
    first = false;
    const FailureOr<LaunchLayout> layout =
        plan_launch(capture.header.manifest, execution.dimensions, execution.group_bytes,
                    execution.private_bytes);
    if (layout.failed())
      return 2;
    std::cout << "{\"serial\":" << execution.serial << ",\"slot\":" << execution.storage_slot
              << ",\"generation\":" << execution.storage_generation << ",\"report_address\":"
              << *wire::read_u64(execution.context, wire::kContextReportPointer)
              << ",\"argument_u32\":";
    if (const std::optional<uint32_t> argument = wire::read_u32(execution.arguments, 0))
      std::cout << *argument;
    else
      std::cout << "null";
    std::cout << ",\"rows\":[";
    for (uint32_t wave = 0; wave < layout.value().wave_count; ++wave) {
      if (wave != 0)
        std::cout << ',';
      const size_t row = kReportHeaderBytes + wave * layout.value().row_stride;
      std::cout << "{\"wave\":" << *wire::read_u32(execution.report, row + wire::kRowWave)
                << ",\"serial\":" << *wire::read_u64(execution.report, row + wire::kRowSerial)
                << ",\"status\":" << *wire::read_u32(execution.report, row + wire::kRowStatus)
                << ",\"loss\":" << *wire::read_u32(execution.report, row + wire::kRowLoss)
                << ",\"final_sequence\":"
                << *wire::read_u32(execution.report, row + wire::kRowFinalSequence)
                << ",\"records\":[";
      bool first_record = true;
      for (const SiteFacts &site : capture.header.manifest.sites) {
        for (uint32_t address_group = 0; address_group < kAddressGroupsPerSite; ++address_group) {
          const size_t record =
              row + kRowHeaderBytes + (site.first_slot + address_group) * kAccessRecordBytes;
          const uint32_t commit = *wire::read_u32(execution.report, record + wire::kRecordCommit);
          if (commit == 0)
            continue;
          if (!first_record)
            std::cout << ',';
          first_record = false;
          std::cout << "{\"commit\":" << commit
                    << ",\"site\":" << *wire::read_u32(execution.report, record + wire::kRecordSite)
                    << ",\"sequence\":"
                    << *wire::read_u32(execution.report, record + wire::kRecordSequence)
                    << ",\"offset\":"
                    << *wire::read_u32(execution.report, record + wire::kRecordLdsOffset)
                    << ",\"width\":"
                    << *wire::read_u32(execution.report, record + wire::kRecordWidth)
                    << ",\"mask\":"
                    << *wire::read_u64(execution.report, record + wire::kRecordGuestMask)
                    << ",\"flags\":"
                    << *wire::read_u32(execution.report, record + wire::kRecordFlags) << '}';
        }
      }
      std::cout << "]}";
    }
    std::cout << "]}";
  }
  std::cout << "]}\n";
  return 0;
}
