// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <iosfwd>

namespace rocprofsys
{
class output_file_registry;
}

namespace rocprofsys::output
{

// Records the calling process into `registry`, captures run metadata
// against `load_baseline`, and writes the Output Summary to `os`.
// Single entry point used by both the library finalize path and the
// rocprofiler-sdk attach finalize path.
void
emit_summary(std::ostream& os, output_file_registry& registry,
             std::chrono::steady_clock::time_point load_baseline);

}  // namespace rocprofsys::output
