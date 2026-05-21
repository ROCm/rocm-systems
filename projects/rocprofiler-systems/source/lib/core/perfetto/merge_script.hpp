// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace rocprofsys
{
namespace core
{
namespace perfetto
{
// Spawns rocprof-sys-merge-output.sh on `output_folder` via fork+execv so
// any character in the folder path (including a single quote) is passed
// verbatim to the script — the shell-escaping footprint of std::system()
// is not safe when output_folder is user-controlled. Resolves the script
// from ROCPROFSYS_SCRIPT_PATH (or PATH if that env var is empty, but only
// after verifying the path contains a '/' so an attacker cannot win via
// PATH lookup). No-op on ranks other than 0, and on missing script.
void
run_merge_script(const std::string& output_folder);
}  // namespace perfetto
}  // namespace core
}  // namespace rocprofsys
