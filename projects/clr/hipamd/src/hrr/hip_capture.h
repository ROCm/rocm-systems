/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

/*
 * hip_capture.h — In-tree HIP capture layer public API.
 *
 * Captures HIP API calls to a .hrr archive for later replay.
 * Independent of the profiler layer — no profiler headers included.
 *
 * Binary format is compatible with hrr_reader.h / hrr_replay.cpp.
 * Format constants (HRR_MAGIC, HRR_VERSION, hrr_file_header) are in
 * hrr_api_args.h — included by hip_capture.cpp and hip_capture_writer.cpp.
 */

#include <cstddef>
#include <cstdint>

namespace hrr_cap {

// 128-bit hash (FNV-1a variant) — return type of write_blob / write_code_object
struct Hash128 {
  uint64_t lo;
  uint64_t hi;
};

}  // namespace hrr_cap

struct HipDispatchTable;

// ---------------------------------------------------------------------------
// Public API — called from hip_context.cpp and hip_capture.cpp
// ---------------------------------------------------------------------------

// Check if capture is enabled (HIP_HRR_CAPTURE_OUTPUT env var set and non-empty)
bool hip_capture_enabled();

// Return the output directory from the env var
const char* hip_capture_output_dir();

// Snapshot real fn ptrs (must be called while live table holds real ptrs).
// Pass the table explicitly when it is not yet reachable through
// hip::GetHipDispatchTable() — see hip_capture_install_early().
void hip_capture_build_table(const HipDispatchTable* live = nullptr);

// Install capture shims into live dispatch tables
void hip_capture_install(HipDispatchTable* target = nullptr);

// Called from UpdateDispatchTable(HipDispatchTable*) once every slot holds its
// real function pointer, which is the last moment before a caller can load a
// slot and dispatch through it. Installing here rather than from
// hip_capture_init() is what lets the process's very first HIP call be
// recorded; see the ordering note in hip_capture.cpp.
void hip_capture_install_early(HipDispatchTable* table);

// Restore real dispatch tables
void hip_capture_uninstall();

// Hook compiler dispatch table for <<<>>> launch path
void hip_capture_build_compiler_table();

// Called from hip_context.cpp init() — performs build + conditional install
void hip_capture_init();

// Called at atexit — uninstalls shims and flushes the archive to disk
void hip_capture_shutdown();
