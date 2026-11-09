/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
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
