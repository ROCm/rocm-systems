// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rocprofiler_compute_tool
{

// Returns the source path encoded in an ISA comment: the substring BEFORE the LAST ':'.
// Returns std::nullopt for empty comments, comments with no ':', or comments whose
// path part (before the last ':') is empty.
std::optional<std::string> parse_source_path(const std::string& comment);

// Copies each DISTINCT existing file in source_paths into dest_root, reproducing the
// file's path beneath dest_root. Best-effort: warns and skips on any error and never throws.
void copy_source_files(const std::vector<std::string>& source_paths,
                       const std::filesystem::path&    dest_root);

}  // namespace rocprofiler_compute_tool
