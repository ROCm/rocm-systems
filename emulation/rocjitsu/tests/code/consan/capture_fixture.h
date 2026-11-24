// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/capture.h"

namespace rocjitsu::consan::test {

// Synthetic evidence exercises shared codecs/core branches. It is never an
// input to the physical runner or a substitute for compiler/GPU observations.
inline CaptureHeader capture_header() {
  CaptureHeader header;
  header.run_identity[0] = 1;
  Manifest manifest;
  manifest.original = {digest(std::array<uint8_t, 256>{}), 256};
  manifest.prepared = {digest(std::array<uint8_t, 1024>{}), 1024};
  manifest.options = mvp_options_digest();
  manifest.target = 1100;
  manifest.wave_size = 32;
  manifest.code_object_abi = 5;
  manifest.original_text_bytes = 128;
  manifest.prepared_text_bytes = 768;
  manifest.entry = {.name = "synthetic_codec_fixture",
                    .original_descriptor_offset = 128,
                    .prepared_descriptor_offset = 800,
                    .prepared_entry_offset = 512,
                    .original_kernarg_bytes = 16,
                    .kernarg_alignment = 8,
                    .original_group_bytes = 8,
                    .prepared_group_bytes = 8,
                    .prepared_sgprs = 40,
                    .prepared_vgprs = 16};
  manifest.sites = {{.analysis = AnalysisKind::LdsRaceDetection,
                     .access = AccessKind::Write,
                     .width = 4,
                     .original_size = 8,
                     .prepared_guest_offset = 32,
                     .prepared_guest_size = 8,
                     .probe_offset = 64,
                     .probe_size = 64},
                    {.id = 1,
                     .analysis = AnalysisKind::LdsRaceDetection,
                     .access = AccessKind::Read,
                     .width = 4,
                     .original_offset = 8,
                     .original_size = 8,
                     .prepared_guest_offset = 160,
                     .prepared_guest_size = 8,
                     .probe_offset = 192,
                     .probe_size = 64,
                     .first_slot = 4}};
  header.image = single_entry_image_facts(manifest, 16, 8);
  return header;
}

inline Manifest &manifest(CaptureHeader &header) { return *header.image.entries.front().selected; }
inline const Manifest &manifest(const CaptureHeader &header) {
  return *header.image.entries.front().selected;
}

inline void refresh_image(CaptureHeader &header) {
  const Manifest &first = manifest(header);
  header.image.original = first.original;
  header.image.prepared = first.prepared;
  header.image.target = first.target;
  header.image.wave_size = first.wave_size;
  header.image.code_object_abi = first.code_object_abi;
  header.image.original_text_bytes = first.original_text_bytes;
  header.image.prepared_text_bytes = first.prepared_text_bytes;
  std::vector<EntryKey> keys;
  for (ImageEntryFacts &entry : header.image.entries) {
    if (!entry.selected) {
      continue;
    }
    entry.key = entry.selected->entry.original_descriptor_offset;
    entry.binding = entry.selected->entry;
    keys.push_back(entry.key);
  }
  header.image.selection_profile = selection_profile_digest(keys, first.scope, first.options);
}

inline CaptureHeader trace_capture_header() {
  CaptureHeader header = capture_header();
  Manifest &facts = manifest(header);
  facts.scope = ScopeKind::ExecutionTrace;
  facts.options = profile_options_digest(facts.scope);
  facts.target = 1250;
  facts.entry.workgroup_id_scalar = kWorkgroupIdTtmp9Selector;
  facts.entry.original_group_bytes = facts.entry.prepared_group_bytes = 0;
  facts.sites.clear();
  facts.boundaries = {{.kind = BoundaryKind::Entry,
                       .original_offset = 0,
                       .prepared_guest_offset = 512,
                       .prepared_guest_size = 0,
                       .probe_offset = 64,
                       .probe_size = 64},
                      {.kind = BoundaryKind::Exit,
                       .original_offset = 64,
                       .prepared_guest_offset = 640,
                       .prepared_guest_size = 4,
                       .probe_offset = 192,
                       .probe_size = 64}};
  refresh_image(header);
  header.execution_conditions = capture_execution_conditions(ExecutionMode::Application);
  return header;
}

inline CaptureHeader multi_entry_header() {
  CaptureHeader header = capture_header();
  const Manifest first = manifest(header);
  Manifest second = first;
  second.entry.name = "second_codec_fixture";
  second.entry.original_descriptor_offset = 192;
  second.entry.prepared_descriptor_offset = 864;
  second.entry.original_entry_offset = 64;
  second.entry.prepared_entry_offset = 640;
  second.entry.original_kernarg_bytes = 32;
  second.entry.original_group_bytes = second.entry.prepared_group_bytes = 128;
  second.entry.prepared_sgprs = 48;
  second.entry.prepared_vgprs = 32;
  for (SiteFacts &site : second.sites) {
    site.original_offset += 64;
    site.prepared_guest_offset += 256;
    site.probe_offset += 256;
    site.access = AccessKind::Read;
  }
  const ImageFacts sibling = single_entry_image_facts(second, 32, 24);
  header.image.entries.push_back(sibling.entries.front());
  const std::array<EntryKey, 2> keys = {128, 192};
  header.image.selection_profile = selection_profile_digest(keys);
  return header;
}

inline CapturedExecution execution(const Manifest &manifest, uint64_t serial,
                                   uint32_t generation = 1,
                                   LaunchDimensions dimensions = {{64, 1, 1}, {64, 1, 1}},
                                   ExecutionMode mode = ExecutionMode::Controlled) {
  CapturedExecution execution;
  execution.serial = serial;
  execution.entry_key = manifest.entry.original_descriptor_offset;
  execution.dimensions = dimensions;
  execution.group_bytes = manifest.entry.prepared_group_bytes;
  execution.completion = completion_evidence(mode);
  execution.storage_slot = 1;
  execution.storage_generation = generation;
  execution.arguments.resize(manifest.entry.original_kernarg_bytes);
  const FailureOr<TransportImage> transport = initialize_transport(
      manifest, execution.dimensions, execution.group_bytes, 0, serial, 0x10000);
  if (transport.succeeded()) {
    execution.context = transport.value().context;
    execution.report = transport.value().report;
  }
  return execution;
}

} // namespace rocjitsu::consan::test
