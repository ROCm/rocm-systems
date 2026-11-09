/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * hip_capture.h — In-tree HIP capture layer public API.
 *
 * Captures HIP API calls to a .hrr archive for later replay.
 * Independent of the profiler layer — no profiler headers included.
 *
 * Binary format is compatible with hrr_reader.h / hrr_replay.cpp.
 * Format constants (HRR_MAGIC, HRR_VERSION, hrr_file_header) are in
 * hrr_api_args.h — included by hip_capture.cpp and hip_capture_writer.cpp.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace hrr_cap {

struct Hash128 {
  uint64_t lo;
  uint64_t hi;
};

}  // namespace hrr_cap


// True when HIP_HRR_CAPTURE_OUTPUT is set and not blank, and the process did not start in secure-execution mode.
bool hip_capture_enabled();

const char* hip_capture_output_dir();

// Call while the live table still holds the real pointers.
void hip_capture_build_table();

void hip_capture_install();

void hip_capture_uninstall();

void hip_capture_build_compiler_table();

struct HipCompilerDispatchTable;

// Call only from hip_capture_build_compiler_table(), after it snapshots g_real_compiler_table.
void hip_capture_install_compiler_table(const HipCompilerDispatchTable& shims);

void hip_capture_init();

void hip_capture_shutdown();
