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
  Manifest &manifest = header.manifest;
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
                     .original_size = 8,
                     .prepared_guest_offset = 32,
                     .prepared_guest_size = 8,
                     .probe_offset = 64,
                     .probe_size = 64},
                    {.id = 1,
                     .analysis = AnalysisKind::LdsRaceDetection,
                     .access = AccessKind::Read,
                     .original_offset = 8,
                     .original_size = 8,
                     .prepared_guest_offset = 160,
                     .prepared_guest_size = 8,
                     .probe_offset = 192,
                     .probe_size = 64,
                     .first_slot = 4}};
  return header;
}

inline CapturedExecution execution(const Manifest &manifest, uint64_t serial,
                                   uint32_t generation = 1,
                                   LaunchDimensions dimensions = {{64, 1, 1}, {64, 1, 1}}) {
  CapturedExecution execution;
  execution.serial = serial;
  execution.dimensions = dimensions;
  execution.group_bytes = manifest.entry.prepared_group_bytes;
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
